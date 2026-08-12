"""Seam probe: is clip B's regenerated head audio a drifting copy of clip
A's tail, or unrelated noise?

Usage:
    python seam_probe.py clip_A.flac clip_B_UNTRIMMED.flac [--frames 22] [--fps 24]

clip_B_UNTRIMMED must be B's audio BEFORE the trim node -- branch the audio
VAE decode output to a Save Audio node in parallel with the trim. clip_A is
the previous clip as delivered (tail-matched is fine; match_tail only
removes ~8ms and the probe aligns on content, not file length).

The pinned video covers B's frames 0..N-1, which the model generates as a
reconstruction of A's last N frames. This script slides short windows of
B's head over the corresponding region of A's tail and reports, per window:

    t_B      window position in B (seconds from B's start)
    lag_ms   how far B runs behind A (positive = B is LATE: its copy of
             the bassline arrives after A's original did). Tracked with
             period-ambiguity handling, so a repeating bassline does not
             fake a drift by cycle-hopping.
    corr     normalised cross-correlation peak (1.0 = identical shape,
             ~0 = unrelated)

Reading the result:
    corr high (>0.6) and lag ~0 throughout        clean continuation; the
                                                  blip is something else
    corr high, lag grows/shifts toward the seam   PHASE DRIFT -- crossfade
                                                  at the join is the fix,
                                                  shorter video window
                                                  should shrink it
    corr low from the start                       the ref path is not
                                                  continuing A's audio at
                                                  all; windowing will never
                                                  fix it (idea 4 territory)

One limitation to know before trusting a lag. Cross-correlation cannot
tell a true alignment from one a whole cycle away, so on strongly
periodic content a lag larger than half a period reads as the nearest
alias: 60 ms on material with an 18 ms period comes back as -12.7 ms, at
high correlation. Beat-driven music is exactly this case. Tracking keeps
a drift from cycle-hopping mid-span, and the first window takes the
global peak so a uniform whole-step offset is visible at all, but a large
uniform offset on periodic content is not something the method resolves.
Read a small lag as reliable and a large one as "at least this much,
possibly plus a whole number of cycles".

The node does the same measurement without the file round trip, and
without having to infer where the seam is: H3 Motion Context Seam Probe,
wired between the audio VAE decode and the trim node.

Dependencies: numpy, plus ONE of torchaudio / soundfile / stdlib wave
(16/32-bit PCM wav). ComfyUI's embedded python has numpy and torchaudio.
Run it with that interpreter if your system python lacks them, e.g.:
    python_embeded\\python.exe seam_probe.py A.flac B_untrimmed.flac
"""

import argparse
import sys

import numpy as np


def load_audio(path):
    """Return (mono float64 array, sample_rate)."""
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)
        return wav.mean(dim=0).numpy().astype(np.float64), int(sr)
    except Exception:
        pass
    try:
        import soundfile as sf
        data, sr = sf.read(path, always_2d=True)
        return data.mean(axis=1).astype(np.float64), int(sr)
    except Exception:
        pass
    import wave
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        width = w.getsampwidth()
        ch = w.getnchannels()
        raw = w.readframes(n)
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise RuntimeError("unsupported wav sample width %d" % width)
    data = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if width == 1:
        data -= 128.0
    data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def norm_xcorr(win, ref):
    """Slide `win` across `ref`; return (offsets, ncc) where offsets index
    positions of the window's start within ref."""
    win = win - win.mean()
    ref = ref - ref.mean()
    wn = np.sqrt((win ** 2).sum())
    if wn < 1e-12:
        return None, None
    corr = np.correlate(ref, win, mode="valid")
    csum = np.concatenate(([0.0], np.cumsum(ref ** 2)))
    seg = csum[len(win):] - csum[:-len(win)]
    denom = wn * np.sqrt(np.maximum(seg, 1e-12))
    return np.arange(len(corr)), corr / denom


def tracked_peak(ncc, expect_idx):
    """Pick this window's alignment with period-ambiguity handling.

    Music is periodic -- a 100 Hz bassline repeats every 10 ms -- so the
    global NCC argmax can hop a whole cycle between windows and fake a
    drift. Instead: take every local maximum within 10% of the global
    peak, and follow the one CLOSEST to the alignment predicted by the
    previous window (starting from zero lag). Genuine drift moves far
    less than a period per 25 ms hop, so continuity keeps the true peak
    and discards cycle aliases.
    """
    best = float(ncc.max())
    i = np.arange(1, len(ncc) - 1)
    peaks = i[(ncc[i] >= ncc[i - 1]) & (ncc[i] >= ncc[i + 1]) &
              (ncc[i] >= 0.9 * best)]
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(ncc))])
    pick = int(peaks[np.argmin(np.abs(peaks - expect_idx))])
    return pick, float(ncc[pick])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_a")
    ap.add_argument("clip_b_untrimmed")
    ap.add_argument("--frames", type=int, default=22,
                    help="pinned video frames (the span the model regenerates)")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--win-ms", type=float, default=50.0,
                    help="analysis window length")
    ap.add_argument("--hop-ms", type=float, default=25.0,
                    help="hop between windows (25ms = one audio latent step)")
    ap.add_argument("--search-ms", type=float, default=40.0,
                    help="max lag searched either side")
    args = ap.parse_args()

    a, sr_a = load_audio(args.clip_a)
    b, sr_b = load_audio(args.clip_b_untrimmed)
    if sr_a != sr_b:
        sys.exit("sample rates differ (%d vs %d); resample first" % (sr_a, sr_b))
    sr = sr_a

    span_s = args.frames / args.fps
    span = int(round(span_s * sr))
    if len(b) < span:
        sys.exit("clip B shorter than the pinned span -- is this the "
                 "UNTRIMMED audio?")
    if len(a) < span:
        sys.exit("clip A shorter than the pinned span")

    # B[0:span] should reconstruct A[-span:]
    a_tail_start = len(a) - span
    win = int(round(args.win_ms / 1000.0 * sr))
    hop = int(round(args.hop_ms / 1000.0 * sr))
    search = int(round(args.search_ms / 1000.0 * sr))

    print("clip A: %.3fs   clip B (untrimmed): %.3fs   sr %d" %
          (len(a) / sr, len(b) / sr, sr))
    print("pinned span: %d frames = %.4fs = %d samples" %
          (args.frames, span_s, span))
    print("window %.0fms, hop %.0fms, search +/-%.0fms" %
          (args.win_ms, args.hop_ms, args.search_ms))
    print()
    print("%8s  %9s  %6s   region" % ("t_B (s)", "lag (ms)", "corr"))

    lags, corrs = [], []
    t = 0
    prev_lag = None  # samples; seeded from the first window's global peak
    # continue a little past the seam to see whether alignment against
    # A's (now nonexistent) continuation collapses, as it should
    while t + win <= span + int(0.2 * sr) and t + win <= len(b):
        centre_in_a = a_tail_start + t
        lo = max(0, centre_in_a - search)
        hi = min(len(a), centre_in_a + win + search)
        ref = a[lo:hi]
        if len(ref) <= win:
            break
        offsets, ncc = norm_xcorr(b[t:t + win], ref)
        if ncc is None:
            t += hop
            continue
        # index in the ncc curve where lag == prev_lag; sign convention:
        # positive lag = B is LATE (its content matches an EARLIER part
        # of A than the nominal position), so match offset = centre - lag
        if prev_lag is None:
            # Nothing to track from yet, so take the global peak. Seeding
            # at zero instead assumes the first window is already
            # aligned, and when the whole join is offset by a step that
            # assumption makes the tracker pick the nearest cycle alias:
            # a uniform 25 ms lag on 55 Hz-fundamental content reads as
            # 6.8 ms. That is the failure this probe exists to find, so
            # it must not be the one case it cannot see.
            pick = int(np.argmax(ncc))
            corr = float(ncc[pick])
        else:
            expect_idx = (centre_in_a - prev_lag) - lo
            pick, corr = tracked_peak(ncc, expect_idx)
        lag = centre_in_a - (lo + pick)
        in_span = t + win <= span
        tag = "reconstruction" if in_span else "past seam"
        print("%8.3f  %9.2f  %6.3f   %s" %
              (t / sr, lag / sr * 1000.0, corr, tag))
        if in_span:
            lags.append(lag / sr * 1000.0)
            corrs.append(corr)
            if corr > 0.6:
                prev_lag = lag  # only track when the match is credible
        t += hop

    print()
    if not corrs:
        sys.exit("no windows analysed")
    corrs = np.array(corrs)
    lags = np.array(lags)
    strong = corrs > 0.6
    at_edge = np.abs(lags) >= (args.search_ms - 1.0)
    print("summary over the reconstruction span:")
    print("  mean corr %.3f   windows with corr>0.6: %d/%d   "
          "lags pinned at search edge: %d" %
          (corrs.mean(), int(strong.sum()), len(corrs), int(at_edge.sum())))
    usable = strong & ~at_edge
    if usable.sum() >= 5 and corrs.mean() > 0.55:
        # fit lag = a*t + b over the credible windows; drift is a coherent
        # trend, so demand small residuals before calling it that
        idx = np.nonzero(usable)[0].astype(np.float64)
        sl = lags[usable]
        a_fit, b_fit = np.polyfit(idx, sl, 1)
        resid = sl - (a_fit * idx + b_fit)
        rms = float(np.sqrt((resid ** 2).mean()))
        total = a_fit * (len(corrs) - 1)
        print("  lag trend: %.2fms across the span, residual rms %.2fms" %
              (total, rms))
        if rms < 2.0 and corrs.mean() > 0.75 and abs(total) > 2.0:
            print("  -> DRIFT: B tracks A with a coherent %.1fms slide. A "
                  "~40ms crossfade at the join should mask it; a shorter "
                  "video window should shrink it." % total)
        elif rms < 2.0 and corrs.mean() > 0.75:
            print("  -> B tracks A with stable phase. The blip is not "
                  "accumulated drift; look at the first generated steps "
                  "past the seam instead.")
        else:
            print("  -> PARTIAL RESEMBLANCE: B resembles A's tail but does "
                  "not phase-lock to it (mean corr %.2f, lag rms %.1fms). "
                  "Imitation, not continuation -- idea 4 territory."
                  % (corrs.mean(), rms))
    else:
        print("  -> LOW/INCOHERENT CORRELATION: the regenerated head does "
              "not phase-track A's tail. The ref path is imitating, not "
              "continuing; no window length will fix the seam. Idea 4 "
              "(audio on the keyframe path) is the direction.")


if __name__ == "__main__":
    main()
