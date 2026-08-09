"""Freeze probe: did a shot render as a still instead of a held frame?

The airlock technique for chaining across a setup change asks clip N+1 to
open by holding clip N's exact closing framing for a beat before cutting
to the new setup. That works, but a held framing with nothing happening
in it renders as a literal freeze: several seconds of a motionless actor
that reads as a stalled video player, not a pause. It is easy to miss on
a first watch and obvious once someone points at it.

This walks a clip and reports every stretch where the picture stops
moving.

Usage:
    python freeze_detect.py clip01.mp4 clip02.mp4 ...
    python freeze_detect.py --head-frames 22 --min-ms 400 clip02.mp4

Method: decode, downscale each frame to a small grey thumbnail, and take
the mean absolute difference between consecutive thumbnails. Downscaling
is what makes this work on compressed video, since it averages away the
codec noise that would otherwise register as motion in an otherwise
static frame.

The threshold is relative, not absolute. An absolute one cannot serve
both a dim interior and a bright exterior, so each clip is scored against
its own median frame difference and a stretch is flagged when motion
drops below `--rel` of that median (default 0.15) for at least `--min-ms`
(default 400). A separate absolute floor catches duplicated frames that
are identical to within rounding.

The first `--head-frames` are reported separately whatever they score,
because on a chained clip those are the pinned head: the model's re-render
of the previous clip's tail, which is the region the trim removes and the
region an airlock hold lives in. Motion there that is far below the rest
of the clip means the hold had no business in it.

Reading the result:
    no runs                  nothing held
    a run inside the head    the airlock hold has nothing happening in
                             it. Give it a breath, a weight shift, an
                             eyeline change. The framing holds; the
                             performer should not.
    a run elsewhere          the prompt ran out of things for the shot
                             to do, or the clip is longer than the
                             action written for it

Dependencies: numpy, plus ONE of av / cv2 / imageio for decoding.
ComfyUI's embedded python has numpy and PyAV; run it with that
interpreter if your system python lacks them.
"""

import argparse
import os
import sys

import numpy as np

THUMB_W, THUMB_H = 64, 36


def _to_thumb(rgb):
    """[H, W, 3] uint8 -> [THUMB_H, THUMB_W] float64 grey, box averaged.

    Box averaging rather than nearest sampling, because nearest keeps the
    per-pixel codec noise this is trying to average away.
    """
    a = np.asarray(rgb)
    if a.ndim == 3 and a.shape[2] >= 3:
        grey = a[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    else:
        grey = a.astype(np.float64).squeeze()
    h, w = grey.shape
    ys = np.linspace(0, h, THUMB_H + 1).astype(int)
    xs = np.linspace(0, w, THUMB_W + 1).astype(int)
    out = np.empty((THUMB_H, THUMB_W), dtype=np.float64)
    for i in range(THUMB_H):
        y0, y1 = ys[i], max(ys[i] + 1, ys[i + 1])
        row = grey[y0:y1]
        for j in range(THUMB_W):
            x0, x1 = xs[j], max(xs[j] + 1, xs[j + 1])
            out[i, j] = row[:, x0:x1].mean()
    return out


def iter_thumbs(path):
    """Yield (thumbnail, fps). Tries PyAV, then OpenCV, then imageio."""
    try:
        import av
        with av.open(path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 24.0
            for frame in container.decode(stream):
                yield _to_thumb(frame.to_ndarray(format="rgb24")), fps
        return
    except ImportError:
        pass

    try:
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("cv2 could not open %s" % path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield _to_thumb(frame[..., ::-1]), float(fps)
        cap.release()
        return
    except ImportError:
        pass

    try:
        import imageio.v3 as iio
        meta = iio.immeta(path)
        fps = float(meta.get("fps", 24.0))
        for frame in iio.imiter(path):
            yield _to_thumb(frame), fps
        return
    except ImportError:
        pass

    raise RuntimeError(
        "no video decoder available. Install one of av, opencv-python or "
        "imageio, or run this with ComfyUI's embedded python which ships "
        "PyAV.")


def frame_deltas(path):
    """Mean absolute difference between consecutive frames, 0..255 scale."""
    deltas, prev, fps = [], None, 24.0
    for thumb, f in iter_thumbs(path):
        fps = f
        if prev is not None:
            deltas.append(float(np.abs(thumb - prev).mean()))
        prev = thumb
    return np.array(deltas), fps


def find_runs(deltas, fps, rel=0.15, abs_floor=0.15, min_ms=400.0):
    """Stretches where motion sits below the clip's own normal.

    Returns a list of (start_frame, end_frame_exclusive, mean_delta,
    mean_ratio). `deltas[i]` is the motion between frame i and i+1, so a
    run over delta indices [a, b) means frames a..b are the still ones.
    """
    if len(deltas) < 2:
        return []
    median = float(np.median(deltas))
    if median <= 1e-9:
        # the entire clip is static; report it as one run rather than
        # dividing by a zero normal
        return [(0, len(deltas) + 1, float(deltas.mean()), 0.0)]
    ratio = deltas / median
    still = (ratio < rel) | (deltas < abs_floor)
    min_frames = max(1, int(round(min_ms / 1000.0 * fps)))

    runs, start = [], None
    for i, s in enumerate(list(still) + [False]):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if i - start >= min_frames:
                seg = deltas[start:i]
                runs.append((start, i + 1, float(seg.mean()),
                             float((seg / median).mean())))
            start = None
    return runs


def report(path, head_frames, rel, abs_floor, min_ms):
    deltas, fps = frame_deltas(path)
    name = os.path.basename(path)
    if len(deltas) < 2:
        print("%-28s  too few frames to score" % name)
        return 0
    median = float(np.median(deltas))
    runs = find_runs(deltas, fps, rel, abs_floor, min_ms)
    print("%-28s  %d frames @ %.3f fps, median motion %.3f" %
          (name, len(deltas) + 1, fps, median))

    if head_frames > 0 and len(deltas) > head_frames:
        head = deltas[:head_frames]
        h_ratio = float(head.mean()) / median if median > 0 else 0.0
        flag = "  <- HELD, give it something to do" if h_ratio < rel else ""
        print("    head (first %d frames): motion %.3f, %.2fx the clip "
              "median%s" % (head_frames, float(head.mean()), h_ratio, flag))

    if not runs:
        print("    no freezes")
        return 0
    for a, b, mean_d, mean_r in runs:
        # deltas[i] is the motion between frame i and frame i+1, so a run
        # over frames a..b-1 holds still for b-1-a intervals
        secs = (b - 1 - a) / fps
        where = "head" if a < head_frames else "body"
        print("    FREEZE %6.2fs to %6.2fs  (%5.2fs, %s)  motion %.3f "
              "= %.2fx median" %
              (a / fps, (b - 1) / fps, secs, where, mean_d, mean_r))
    return len(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="video clips to scan")
    ap.add_argument("--head-frames", type=int, default=22,
                    help="frames of pinned head to report separately; match "
                         "your context_length, or 0 to skip")
    ap.add_argument("--rel", type=float, default=0.15,
                    help="fraction of the clip's median frame motion below "
                         "which a frame counts as still")
    ap.add_argument("--abs-floor", type=float, default=0.15,
                    help="absolute motion floor on a 0-255 scale, for "
                         "duplicated frames in an already still shot")
    ap.add_argument("--min-ms", type=float, default=400.0,
                    help="shortest stretch worth reporting")
    args = ap.parse_args()

    total = 0
    for path in args.files:
        total += report(path, args.head_frames, args.rel, args.abs_floor,
                        args.min_ms)
        print()
    print("%d freeze run(s) across %d clip(s)" % (total, len(args.files)))
    if total:
        print("A held framing needs a breath, a weight shift or an eyeline "
              "change written into the prompt. The camera holds; the "
              "performer should not.")


def _self_test():
    """Synthetic checks for the detection maths, no video files needed."""
    rng = np.random.default_rng(0)
    fps = 24.0

    # 120 frames of ordinary motion with a 48 frame (2s) freeze at 30
    deltas = rng.uniform(1.8, 2.2, 120)
    deltas[30:78] = rng.uniform(0.0, 0.02, 48)
    runs = find_runs(deltas, fps, min_ms=400.0)
    assert len(runs) == 1, runs
    a, b, mean_d, mean_r = runs[0]
    assert a == 30 and b == 79, (a, b)          # frames 30..78 identical
    assert abs((b - 1 - a) / fps - 2.0) < 1e-9  # which is exactly 2.0s
    assert mean_r < 0.02, mean_r

    # a short dip must not be reported at the default 400 ms
    short = rng.uniform(1.8, 2.2, 120)
    short[50:55] = 0.01
    assert find_runs(short, fps, min_ms=400.0) == []
    assert len(find_runs(short, fps, min_ms=100.0)) == 1

    # a dim clip with small absolute motion but no freeze must stay clean,
    # which is the case an absolute-only threshold would fail
    dim = rng.uniform(0.30, 0.34, 120)
    assert find_runs(dim, fps, min_ms=400.0) == [], "dim clip false positive"

    # and a freeze inside that same dim clip must still be caught
    dim2 = dim.copy()
    dim2[60:100] = 0.001
    assert len(find_runs(dim2, fps, min_ms=400.0)) == 1

    # a wholly static clip reports one run rather than dividing by zero
    assert len(find_runs(np.zeros(60), fps)) == 1

    # thumbnailing must average codec noise away: two frames of the same
    # picture plus independent noise read as far less motion than two
    # genuinely different pictures
    base = rng.integers(0, 255, (360, 640, 3)).astype(np.uint8)
    noisy = np.clip(base.astype(int) +
                    rng.integers(-6, 7, base.shape), 0, 255).astype(np.uint8)
    other = rng.integers(0, 255, (360, 640, 3)).astype(np.uint8)
    still_d = float(np.abs(_to_thumb(base) - _to_thumb(noisy)).mean())
    moved_d = float(np.abs(_to_thumb(base) - _to_thumb(other)).mean())
    assert still_d < 0.5, still_d
    assert moved_d > 10 * still_d, (still_d, moved_d)

    print("freeze_detect self-test passed "
          "(still %.3f vs moved %.3f after thumbnailing)" %
          (still_d, moved_d))


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
