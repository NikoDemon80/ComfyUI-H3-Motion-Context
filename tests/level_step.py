"""Level step probe: does the sound level jump at a join?

Companion to seam_probe.py. seam_probe answers "is clip B's audio a real
continuation of clip A's, or an imitation" by cross-correlation. This
answers a different and simpler question: "does the loudness or the room
tone change abruptly across the cut?" A chain can be perfectly phase
locked and still audibly step if one clip invents its own silence.

Usage:
    python level_step.py clip01.flac clip02.flac clip03.flac ...
    python level_step.py --joins 12.5,25.0 episode.wav

Give the DELIVERED clips in chain order (trimmed, the ones you actually
concatenate), or one already concatenated file plus the join times in
seconds. Either way each join is scored twice:

    broadband   RMS of the window either side. Catches gain jumps.
    floor       10th percentile of short-window RMS, which tracks the
                room tone under the content. Catches the failure where
                each clip invents its own silence and the ambience
                restarts at every cut, even though the music matches.

Both are reported as a normalised step in [0, 1]:

    step = |a - b| / (a + b)

so 0.0 is identical level, 1.0 means one side is digitally silent. The
dB difference is printed alongside because it is easier to judge by ear.

Reading the result:
    floor < 0.25       clean, the bed runs through the cut
    floor 0.25 to 0.50 marginal, listen to it
    floor > 0.50       audible step

Broadband is judged more loosely (ok below 0.40) because the content
either side of a real cut is different by definition, so some movement
there is the edit, not a fault. The floor is the one that should hold
still. If broadband is fine and floor is bad,
                   the ambience is resetting: wire the previous clip's
                   audio (context_latent, or context_audio + audio_vae)
                   into Motion Context so the next clip hears it.

One limitation worth knowing: the floor figure reads the quiet moments
between content, so it only means something when the content lets up.
Wall to wall music with no gaps has nothing underneath it to measure and
the floor number collapses onto the broadband one. Dialogue, ambience
and anything with rests are where it earns its keep.

Sample rates are printed per file and any mismatch is called out loudly.
A stream-copy concat cannot change rate mid stream, so mixing 32 kHz H3
output with a 48 kHz assumption decodes the tail to silence while every
duration assert stays green. H3 emits 32 kHz. Read it, never assume it.

Dependencies: numpy, plus ONE of torchaudio / soundfile / stdlib wave.
ComfyUI's embedded python has numpy and torchaudio; run it with that
interpreter if your system python lacks them.
"""

import argparse
import sys

import numpy as np


def load_audio(path):
    """Return (mono float64 array, sample_rate). Same loader as seam_probe."""
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
    scale = {1: 128.0, 2: 32768.0, 4: 2147483648.0}[width]
    data = data.reshape(-1, ch).mean(axis=1) / scale
    return data, sr


def _pair_label(a_name, b_name):
    import os
    short = lambda p: os.path.splitext(os.path.basename(p))[0][-8:]
    return "%s|%s" % (short(a_name), short(b_name))


def rms(x):
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def floor_level(x, sr, win_ms=20.0, pct=10.0):
    """Noise floor estimate: a low percentile of short-window RMS.

    The quietest moments of a window are room tone plus whatever the
    encoder left behind. Content (a note, a word) raises the mean but
    barely moves the 10th percentile, so this isolates the bed that is
    supposed to run continuously underneath the whole chain.
    """
    win = max(1, int(round(win_ms / 1000.0 * sr)))
    n = len(x) // win
    if n < 3:
        return rms(x)
    trimmed = x[:n * win].reshape(n, win)
    levels = np.sqrt(np.mean(np.square(trimmed), axis=1))
    return float(np.percentile(levels, pct))


def step_ratio(a, b):
    """Normalised difference in [0, 1]. 1.0 when one side is silent."""
    denom = a + b
    if denom <= 1e-12:
        return 0.0
    return abs(a - b) / denom


def db_delta(a, b):
    """Level change from a to b in dB, clamped so digital silence on one
    side prints as a readable -99 rather than a wall of digits."""
    if a <= 1e-9 and b <= 1e-9:
        return 0.0
    lo = 1e-9
    d = 20.0 * np.log10(max(b, lo) / max(a, lo))
    return float(np.clip(d, -99.0, 99.0))


# broadband level legitimately moves across a real cut, because the
# content on either side is different by definition. The floor should
# not: the bed under the content is supposed to run continuously
# through the whole chain. So they get different thresholds.
BROADBAND_OK, BROADBAND_BAD = 0.40, 0.65
FLOOR_OK, FLOOR_BAD = 0.25, 0.50


def verdict(step, ok=0.25, bad=0.50):
    if step < ok:
        return "ok"
    if step < bad:
        return "MARGINAL"
    return "STEP"


def score_join(tail, head, sr, label):
    """Score one join given the audio either side of it."""
    b_tail, b_head = rms(tail), rms(head)
    f_tail, f_head = floor_level(tail, sr), floor_level(head, sr)
    b_step = step_ratio(b_tail, b_head)
    f_step = step_ratio(f_tail, f_head)
    print("%-20s  broadband %.3f (%+6.1f dB) %-8s   "
          "floor %.3f (%+6.1f dB) %-8s" %
          (label, b_step, db_delta(b_tail, b_head),
           verdict(b_step, BROADBAND_OK, BROADBAND_BAD),
           f_step, db_delta(f_tail, f_head),
           verdict(f_step, FLOOR_OK, FLOOR_BAD)))
    return b_step, f_step


def from_clips(paths, win_ms, guard_ms):
    rates, joins = [], []
    prev = None
    for p in paths:
        data, sr = load_audio(p)
        rates.append((p, sr, len(data) / sr))
        if prev is not None:
            joins.append((prev[0], data, sr, prev[1], p))
        prev = (data, p)

    print("clips:")
    for p, sr, dur in rates:
        print("  %-40s %6d Hz  %7.3fs" % (p[-40:], sr, dur))
    uniq = sorted(set(sr for _, sr, _ in rates))
    if len(uniq) > 1:
        print()
        print("  !! SAMPLE RATE MISMATCH across the chain: %s" % uniq)
        print("     A stream-copy concat cannot switch rate mid stream. "
              "Resample to one rate before joining, or the tail decodes "
              "to silence with every duration check still passing.")
    print()

    win = None
    results = []
    print("%-20s  %-42s   %s" % ("join", "broadband", "floor"))
    for a_data, b_data, sr, a_name, b_name in joins:
        win = int(round(win_ms / 1000.0 * sr))
        guard = int(round(guard_ms / 1000.0 * sr))
        tail = a_data[max(0, len(a_data) - guard - win):len(a_data) - guard]
        head = b_data[guard:guard + win]
        if len(tail) < win // 2 or len(head) < win // 2:
            print("%-20s  too short to score" % _pair_label(a_name, b_name))
            continue
        results.append(score_join(tail, head, sr,
                                  _pair_label(a_name, b_name)))
    return results


def from_single(path, join_times, win_ms, guard_ms):
    data, sr = load_audio(path)
    print("file: %s   %d Hz   %.3fs" % (path, sr, len(data) / sr))
    print()
    win = int(round(win_ms / 1000.0 * sr))
    guard = int(round(guard_ms / 1000.0 * sr))
    results = []
    print("%-20s  %-42s   %s" % ("join", "broadband", "floor"))
    for t in join_times:
        c = int(round(t * sr))
        tail = data[max(0, c - guard - win):max(0, c - guard)]
        head = data[min(len(data), c + guard):min(len(data), c + guard + win)]
        if len(tail) < win // 2 or len(head) < win // 2:
            print("%-20s  outside the file" % ("%.2fs" % t))
            continue
        results.append(score_join(tail, head, sr, "%.2fs" % t))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+",
                    help="delivered clips in chain order, or one "
                         "concatenated file with --joins")
    ap.add_argument("--joins", default=None,
                    help="comma separated join times in seconds; use with "
                         "a single concatenated file")
    ap.add_argument("--win-ms", type=float, default=400.0,
                    help="measurement window either side of the join. Long "
                         "enough to span typical gaps in the content: a "
                         "window that lands entirely inside a musical rest "
                         "on one side and a downbeat on the other reports a "
                         "step that is really just the material")
    ap.add_argument("--guard-ms", type=float, default=5.0,
                    help="skipped either side, so a single splice sample "
                         "cannot dominate the window")
    args = ap.parse_args()

    if args.joins:
        if len(args.files) != 1:
            sys.exit("--joins takes exactly one concatenated file")
        times = [float(t) for t in args.joins.split(",") if t.strip()]
        results = from_single(args.files[0], times, args.win_ms, args.guard_ms)
    else:
        if len(args.files) < 2:
            sys.exit("give at least two clips, or one file with --joins")
        results = from_clips(args.files, args.win_ms, args.guard_ms)

    print()
    if not results:
        sys.exit("no joins scored")
    b = np.array([r[0] for r in results])
    f = np.array([r[1] for r in results])
    print("summary over %d joins:" % len(results))
    print("  broadband  median %.3f  worst %.3f  passing (<%.2f) %d/%d" %
          (np.median(b), b.max(), BROADBAND_OK,
           int((b < BROADBAND_OK).sum()), len(b)))
    print("  floor      median %.3f  worst %.3f  passing (<%.2f) %d/%d" %
          (np.median(f), f.max(), FLOOR_OK,
           int((f < FLOOR_OK).sum()), len(f)))
    if np.median(f) >= FLOOR_OK and np.median(b) < BROADBAND_OK:
        print()
        print("  -> AMBIENCE RESET: content levels match but the bed under "
              "them does not. Each clip is inventing its own room tone. "
              "Carry the previous clip's audio across the join "
              "(context_latent, or context_audio + audio_vae).")
    elif np.median(b) >= BROADBAND_BAD:
        print()
        print("  -> LEVEL STEP: check for a gain change between clips, or "
              "a normalisation pass applied per clip instead of once at "
              "the end.")
    elif np.median(b) < BROADBAND_OK and np.median(f) < FLOOR_OK:
        print()
        print("  -> joins are level-continuous. If they still sound wrong, "
              "run seam_probe.py: the problem is phase, not level.")


def _self_test():
    """Synthetic checks for the scoring maths, no audio files needed."""
    sr = 32000
    rng = np.random.default_rng(0)
    t = np.arange(sr) / sr
    # gated tone, not a continuous one: the floor estimator reads the
    # quiet moments between content, so content that never stops has no
    # floor underneath it to find (see the note in the module docstring)
    gate = ((t * 4).astype(int) % 2 == 0).astype(np.float64)
    tone = 0.3 * np.sin(2 * np.pi * 220 * t) * gate
    bed = 0.01 * rng.standard_normal(sr)

    assert abs(step_ratio(0.5, 0.5)) < 1e-12
    assert abs(step_ratio(0.5, 0.0) - 1.0) < 1e-12
    assert abs(step_ratio(0.0, 0.0)) < 1e-12
    assert 0.0 < step_ratio(0.5, 0.4) < 0.2

    # a floor estimate must track the bed, not the tone on top of it
    with_bed = floor_level(tone + bed, sr)
    without = floor_level(tone, sr)
    assert with_bed > without * 2, (with_bed, without)
    assert abs(with_bed - rms(bed)) < rms(bed), (with_bed, rms(bed))

    # ambience reset: same content, one side has no bed under it
    a_step = step_ratio(rms(tone + bed), rms(tone))
    f_step = step_ratio(floor_level(tone + bed, sr), floor_level(tone, sr))
    assert a_step < 0.25, a_step        # broadband says fine
    assert f_step > 0.50, f_step        # floor catches it
    assert verdict(a_step, BROADBAND_OK, BROADBAND_BAD) == "ok"
    assert verdict(f_step, FLOOR_OK, FLOOR_BAD) == "STEP"

    # a 6 dB gain jump must register broadband
    assert step_ratio(rms(tone), rms(tone * 2)) > 0.3
    assert abs(db_delta(rms(tone), rms(tone * 2)) - 6.02) < 0.05
    assert db_delta(0.1, 0.0) == -99.0 and db_delta(0.0, 0.1) == 99.0
    assert db_delta(0.0, 0.0) == 0.0

    print("level_step self-test passed")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
