"""Frozen visual features, precomputed once (plan §7.4).

Two backends:

* ``pose`` (default) — MediaPipe Holistic upper-body + both hands, re-expressed in
  a **signer-normalised frame**: origin at the shoulder midpoint, unit = shoulder
  width, so a keypoint coordinate *is* a position in signing space. This matters
  more here than raw accuracy: a spatial locus is a location in that space, and a
  representation that throws the coordinate frame away cannot encode one. It also
  makes the features comparable across the 24 signers and across the two halves
  of a dialogue frame.

* ``dino`` — DINOv2 ViT-S/14 CLS token per frame. Appearance rather than
  geometry; used as the modality ablation ("does the gain need explicit
  geometry?") and as the fallback if MediaPipe will not install.

One film is decoded exactly once and every utterance span is sliced out of it, so
the cost is 407 decodes rather than 5,282 seeks. Dialogue films carry two signers
side by side, so both halves are extracted and the `speaker` field selects one.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

FPS = 15
FRAME_H = 432  # after cropping; keeps hands ~60 px wide, enough for MediaPipe

# --------------------------------------------------------------- rtmpose (main)
#
# COCO-WholeBody 133 keypoints via rtmlib, which is exactly what Uni-Sign's
# released checkpoints were trained on. We store the RAW 133 points plus their
# confidences — the same content as Uni-Sign's own .pkl files — and leave every
# normalisation to load time, because that is where their `load_part_kp` does it.
# Storing anything else, including an extra channel of our own, would put the
# features off-distribution for the pretrained weights and quietly invalidate the
# whole transfer comparison.
#
# The four groups Uni-Sign reads out of the 133, confirmed twice over: once in
# their `datasets.py`, and once in the shapes of the GCN adjacency matrices in
# the published checkpoint (body [2,9,9], left/right [2,21,21], face_all
# [2,18,18]).
N_WHOLEBODY = 133
UNISIGN_PARTS = {
    "body": [0] + list(range(3, 11)),                       # nose, ears, shoulders, elbows, wrists
    "left": list(range(91, 112)),                           # left hand
    "right": list(range(112, 133)),                         # right hand
    "face_all": list(range(23, 40))[::2] + list(range(83, 91)) + [53],  # jaw, mouth, nose
}
RTMPOSE_DIM = N_WHOLEBODY * 3  # x, y, confidence

# ------------------------------------------------------------- mediapipe (alt)
#
# Kept as an independent, dependency-light alternative and as the from-scratch
# ablation. Not interchangeable with the rtmpose features: different keypoint
# set, different normalisation, no transfer path.
POSE_KEEP = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24]
N_POINTS = len(POSE_KEEP) + 21 + 21  # 15 + both hands = 57
POSE_DIM = N_POINTS * 3 + N_POINTS   # xyz + per-point presence
DINO_DIM = 384

# Which side of the studio a dialogue signer occupies is NOT stored here. It is
# already on every record as `speaker`, so the dataset supplies it downstream;
# baking it into the feature vector would break the dimension the pretrained
# pose encoder expects.
SIDE_CODE = {"L": -1.0, "R": 1.0, None: 0.0}


@dataclass
class Crop:
    """Pixel window into the source frame, before scaling."""

    x: int
    y: int
    w: int
    h: int

    @property
    def ffmpeg(self) -> str:
        return f"crop={self.w}:{self.h}:{self.x}:{self.y}"


@lru_cache(maxsize=512)
def probe(film: str) -> tuple[int, int]:
    """Frame size. Four of the 407 films are 720p, not 1080p, so this cannot be
    assumed — a hardcoded 960-wide crop fails outright on those."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(film)],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split("x")
    return (int(out[0]), int(out[1]))


def crops_for(para_type: str, film: Path | str | None = None,
              width: int | None = None, height: int | None = None
              ) -> dict[str | None, Crop]:
    """Dialogue films are split down the middle; monologues use the full frame.

    The mapping L -> screen-left is asserted by `scripts/03_extract_features.py
    --check-sides`, which measures which half moves while each speaker holds the
    floor rather than trusting the convention. It holds 10/10 with a 4-6x motion
    ratio, so it is safe — but it is checked, not assumed.
    """
    if width is None or height is None:
        if film is None:
            raise ValueError("crops_for needs either a film path or width/height")
        width, height = probe(str(film))
    if para_type == "2":
        half = width // 2
        return {"L": Crop(0, 0, half, height), "R": Crop(half, 0, half, height)}
    return {None: Crop(0, 0, width, height)}


# ------------------------------------------------------------------- decoding


def decode(film: Path, crop: Crop, fps: int = FPS, out_h: int = FRAME_H,
           pix_fmt: str = "rgb24"):
    """Yield uint8 frames. ffmpeg does the crop/scale/resample.

    `pix_fmt` follows what the backend wants: MediaPipe and DINOv2 take RGB,
    rtmlib takes BGR (its own extractor feeds it straight from cv2).
    """
    out_w = int(round(crop.w * out_h / crop.h)) // 2 * 2
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(film),
        "-vf", f"{crop.ffmpeg},fps={fps},scale={out_w}:{out_h}",
        "-f", "rawvideo", "-pix_fmt", pix_fmt, "-",
    ]
    n = out_w * out_h * 3
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         bufsize=n * 4)
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(out_h, out_w, 3)
    finally:
        # The probe pass stops early on purpose, so ffmpeg is usually still
        # writing. Kill it rather than closing the pipe under it, which would
        # otherwise spew "Broken pipe" for every clip.
        if p.poll() is None:
            p.kill()
        p.stdout.close()
        p.wait()


# -------------------------------------------------------------- rtmpose backend


def limit_onnx_threads(n: int) -> None:
    """Cap ONNX Runtime's per-session thread pool.

    rtmlib builds its sessions as `ort.InferenceSession(path, providers=[...])`
    with no `sess_options`, so ORT defaults to one intra-op thread per core. ORT
    does not read OMP_NUM_THREADS for this, so setting env vars looks like it
    works and does nothing: fourteen workers each opened ~95 threads on a 32-core
    box and drove the load average to 889 on a machine with other users on it.
    The only reliable lever is `sess_options`, so inject one.
    """
    import onnxruntime as ort

    if getattr(ort.InferenceSession, "_wslp_thread_capped", False):
        return
    original = ort.InferenceSession

    def capped(*args, **kwargs):
        if "sess_options" not in kwargs:
            so = ort.SessionOptions()
            so.intra_op_num_threads = n
            so.inter_op_num_threads = 1
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            kwargs["sess_options"] = so
        return original(*args, **kwargs)

    capped._wslp_thread_capped = True
    ort.InferenceSession = capped


class RTMPoseBackend:
    """COCO-WholeBody 133 keypoints, the format Uni-Sign's checkpoints expect.

    Emits raw normalised (x, y, confidence) per keypoint and nothing else, so the
    arrays are drop-in for Uni-Sign's own `load_part_kp` — which is where the
    wrist-relative hand coordinates, the nose-relative face and the body-derived
    scale all get computed. Normalising here instead would double-normalise.

    Two departures from Uni-Sign's `demo/pose_extraction.py`, both deliberate:

    * It reads the entire video into RAM before posing. Our films are 1080p and up
      to 210 s, so we stream frames instead.
    * It runs the person detector on every frame. Our crops are one signer, fixed
      camera, green screen — so the detector runs on a handful of probe frames, the
      median box is taken, and the pose model is then given that box directly. That
      is roughly twice as fast and removes per-frame box jitter, which would
      otherwise show up as spurious motion in a representation whose whole content
      is coordinates.
    """

    dim = RTMPOSE_DIM
    name = "rtmpose"
    wants = "bgr24"

    def __init__(self, mode: str = "lightweight", device: str = "cpu",
                 backend: str = "onnxruntime", det_probe: int = 12,
                 threads: int = 2):
        limit_onnx_threads(threads)
        # CPU by default, and not as a fallback. Measured on this machine, one
        # process reaches 31.7 fps on CPU against 21.9 fps on the GPU while another
        # user's job holds it — and CPU scales across 32 cores whereas the GPU is
        # a single contended resource. ffmpeg decode runs at 328 fps, so the pose
        # forward is the only thing that matters here.
        from rtmlib import Wholebody

        # Uni-Sign's demo defaults: mmpose-style indexing (NOT openpose) and
        # `lightweight`. Indexing must match or every slice in load_part_kp is
        # wrong; the mode only affects accuracy, but keeping it equal keeps our
        # features on the distribution the checkpoints were trained on.
        self.wb = Wholebody(mode=mode, backend=backend, device=device,
                            to_openpose=False)
        self.det_probe = det_probe

    def close(self) -> None:
        pass

    def _bbox(self, make_iter, probe_fps: int = 2) -> np.ndarray | None:
        """Median detection box from a cheap low-rate pass over the same clip.

        A separate low-fps decode costs a few seconds and keeps memory flat; the
        alternative — buffering the clip to pick probe frames — is the 3 GB-per-
        worker mistake this class exists to avoid.
        """
        boxes = []
        for i, f in enumerate(make_iter(fps=probe_fps)):
            if len(boxes) >= self.det_probe:
                break
            det = self.wb.det_model(f)
            if det is not None and len(det):
                # largest box: the signer, not a stray detection on the backdrop
                det = np.asarray(det)
                areas = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                boxes.append(det[int(np.argmax(areas))][:4])
        if not boxes:
            return None
        return np.median(np.stack(boxes), axis=0)

    def __call__(self, make_iter) -> np.ndarray:
        box = self._bbox(make_iter)
        bboxes = None if box is None else np.asarray([box])

        rows = []
        for f in make_iter():
            h, w = f.shape[:2]
            row = np.zeros((N_WHOLEBODY, 3), np.float32)
            kp, sc = (self.wb(f) if bboxes is None
                      else self.wb.pose_model(f, bboxes=bboxes))
            if kp is not None and len(kp):
                row[:, :2] = np.asarray(kp[0]) / np.array([w, h])[None]
                row[:, 2] = np.asarray(sc[0])
            rows.append(row.reshape(-1))
        if not rows:
            return np.zeros((0, self.dim), np.float32)
        return np.stack(rows)


def unisign_parts(feat: np.ndarray) -> dict[str, np.ndarray]:
    """(T, 133*3) -> {'body','left','right','face_all'}, each (T, N, 3).

    Uni-Sign's `load_part_kp` takes it from here. Provided so the split is stated
    once, and can be asserted against the checkpoint's GCN shapes.
    """
    kp = feat.reshape(len(feat), N_WHOLEBODY, 3)
    return {name: kp[:, idx, :] for name, idx in UNISIGN_PARTS.items()}


# --------------------------------------------------------------- pose backend


class PoseBackend:
    dim = POSE_DIM
    name = "pose"
    wants = "rgb24"

    def __init__(self) -> None:
        import mediapipe as mp

        self.h = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=False,
        )

    def close(self) -> None:
        self.h.close()

    def __call__(self, make_iter) -> np.ndarray:
        rows = [self._one(f) for f in make_iter()]
        return np.asarray(rows, np.float32) if rows else np.zeros((0, self.dim), np.float32)

    def _one(self, frame: np.ndarray) -> np.ndarray:
        res = self.h.process(frame)
        xyz = np.zeros((N_POINTS, 3), np.float32)
        pres = np.zeros(N_POINTS, np.float32)

        pl = res.pose_landmarks
        if pl is not None:
            for j, i in enumerate(POSE_KEEP):
                lm = pl.landmark[i]
                xyz[j] = (lm.x, lm.y, lm.z)
                pres[j] = getattr(lm, "visibility", 1.0)
        for off, hand in ((len(POSE_KEEP), res.left_hand_landmarks),
                          (len(POSE_KEEP) + 21, res.right_hand_landmarks)):
            if hand is None:
                continue
            for j, lm in enumerate(hand.landmark):
                xyz[off + j] = (lm.x, lm.y, lm.z)
                pres[off + j] = 1.0

        xyz = normalise(xyz, pres)
        return np.concatenate([xyz.reshape(-1), pres])


def normalise(xyz: np.ndarray, pres: np.ndarray) -> np.ndarray:
    """Shoulder-midpoint origin, shoulder-width unit.

    POSE_KEEP indices 7 and 8 are the shoulders (source indices 11, 12). If they
    are missing the frame is left at raw normalised-image coordinates minus 0.5,
    which keeps the scale roughly comparable rather than blowing up.
    """
    ls, rs = 7, 8
    if pres[ls] > 0.3 and pres[rs] > 0.3:
        origin = (xyz[ls] + xyz[rs]) / 2.0
        width = float(np.linalg.norm(xyz[ls, :2] - xyz[rs, :2]))
        scale = width if width > 1e-3 else 0.25
    else:
        origin = np.array([0.5, 0.5, 0.0], np.float32)
        scale = 0.25
    out = (xyz - origin) / scale
    out[pres <= 0.0] = 0.0
    return out


# ----------------------------------------------------------------- dino backend


class DinoBackend:
    dim = DINO_DIM
    name = "dino"
    wants = "rgb24"

    def __init__(self, batch: int = 64, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = device
        self.batch = batch
        self.m = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def close(self) -> None:
        pass

    def __call__(self, make_iter) -> np.ndarray:
        torch = self.torch
        out, buf = [], []
        for f in make_iter():
            buf.append(f)
            if len(buf) == self.batch:
                out.append(self._flush(buf))
                buf = []
        if buf:
            out.append(self._flush(buf))
        if not out:
            return np.zeros((0, self.dim), np.float32)
        return np.concatenate(out, 0)

    def _flush(self, buf) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(np.stack(buf)).to(self.device)
            x = x.permute(0, 3, 1, 2).float().div_(255.0)
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear",
                                                align_corners=False)
            x = (x - self.mean) / self.std
            y = self.m(pixel_values=x).last_hidden_state[:, 0]
        return y.float().cpu().numpy()


BACKENDS = {"rtmpose": RTMPoseBackend, "pose": PoseBackend, "dino": DinoBackend}


# ------------------------------------------------------------------- film pass


def extract_film(film: Path, para_type: str, spans: dict[str, tuple[int, int, str | None]],
                 backend, fps: int = FPS) -> dict[str, np.ndarray]:
    """Decode `film` once per required crop and slice out every span.

    `spans` maps a record key to (t1_ms, t2_ms, speaker). Returns key -> (T, D).
    """
    needed = {sp for _, _, sp in spans.values()}
    pix_fmt = getattr(backend, "wants", "rgb24")
    per_side: dict[str | None, np.ndarray] = {}
    for side, crop in crops_for(para_type, film).items():
        if side not in needed:
            continue
        # A factory, not an iterator: RTMPose needs a second, low-rate pass to fix
        # the detection box, and nothing may be buffered between the two.
        def make_iter(fps=fps, _crop=crop):
            return decode(film, _crop, fps=fps, pix_fmt=pix_fmt)
        per_side[side] = backend(make_iter)

    out: dict[str, np.ndarray] = {}
    for key, (t1, t2, side) in spans.items():
        seq = per_side.get(side if side in per_side else None)
        if seq is None or len(seq) == 0:
            continue
        a = int(t1 * fps / 1000)
        b = max(a + 1, int(round(t2 * fps / 1000)))
        out[key] = seq[a : min(b, len(seq))].astype(np.float16)
    return out
