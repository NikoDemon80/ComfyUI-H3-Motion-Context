"""Seam probe node: measure a join without leaving the graph.

seam_probe.py and level_step.py answer the two questions that matter at a
join, but both work on rendered files and both have to infer where the
seam is. That inference is the source of the constant ~8ms lag the CLI
probe reports: anchoring on a file's end rather than on the frame the
picture actually ends at is off by exactly the audio grid overhang, which
is a third of a step.

This node does not have to infer anything. It takes clip A as the same
latent you already pin from, and clip B before the trim, so the seam is
at a known sample and the pinned span is given rather than guessed. It
decodes A's audio and applies match_tail itself, which matters: comparing
against an untrimmed clip A would move the anchor by the overhang and
hand back a very convincing lag that is purely an artifact. Doing it
inside the node means that cannot be got wrong from the canvas.

Wire it inline: the audio VAE decode feeds this node, and this node feeds
the trim. The audio output is the input unchanged, so dropping it into an
existing chain changes nothing about the render.

    clip_a_latent      the same latent wired into Motion Context's
                       context_latent
    audio_vae          the same audio VAE
    clip_b_untrimmed   this clip's audio off the VAE decode, before trim
    trim_frames        from Motion Context, the same value the trim gets

Reported per join:

    lag_ms    how far B's reconstruction of the pinned span runs behind
              A's original. Positive means B is late.
    corr      normalised cross-correlation over the pinned span. 1.0 is
              identical shape, ~0 is unrelated. Low corr means the model
              is imitating rather than continuing.
    broadband RMS step across the cut, normalised to [0, 1].
    floor     the same for the 10th-percentile short-window RMS, which
              tracks the bed under the content. This is the one that
              catches a clip inventing its own silence.

Thresholds in the summary are the ones documented in the two scripts, not
new judgments. Numbers are always reported; the labels are a reading aid.

One limitation to know before trusting a lag figure. Cross-correlation
cannot tell a true alignment from one a whole cycle away, so on strongly
periodic content a lag larger than half a period reads as the nearest
alias instead: a 60 ms offset on material with an 18 ms period comes back
as -12.7 ms, confidently and at high correlation. Beat-driven music is
exactly this case. The tracker follows continuity between windows so a
drift cannot cycle-hop mid-span, and the first window takes the global
peak rather than assuming zero, which is what lets a uniform whole-step
offset be seen at all. But a large uniform offset on periodic content is
not something the method can resolve. Read a small lag as reliable and a
large one as "at least this much, possibly plus a whole number of
cycles".
"""

import logging

import numpy as np

from .nodes import FPS, _pixel_frames, _streams_from_latent

_LOG = logging.getLogger("h3_motion_context")

# from seam_probe.py: a match below this is not credible enough to track
CORR_CREDIBLE = 0.6
# from level_step.py
FLOOR_CLEAN, FLOOR_MARGINAL = 0.25, 0.50
BROADBAND_OK = 0.40


def _mono(waveform):
    """Any [.., C, L] or [.., L, C] audio tensor -> mono float64 numpy.

    The audio VAE's decode output orientation is not something to assume,
    so the channel axis is identified by being the short one rather than
    by position. Audio is thousands of samples long and at most a handful
    of channels wide, so this is unambiguous in practice.
    """
    a = waveform
    if hasattr(a, "detach"):
        a = a.detach()
    if hasattr(a, "cpu"):
        a = a.cpu()
    a = np.asarray(a, dtype=np.float64)
    while a.ndim > 2:
        a = a[0]
    if a.ndim == 2:
        axis = 0 if a.shape[0] <= a.shape[1] else 1
        a = a.mean(axis=axis)
    return a


def _norm_xcorr(win, ref):
    """Slide `win` across `ref`; return normalised correlation curve."""
    win = win - win.mean()
    ref = ref - ref.mean()
    wn = np.sqrt((win ** 2).sum())
    if wn < 1e-12:
        return None
    corr = np.correlate(ref, win, mode="valid")
    csum = np.concatenate(([0.0], np.cumsum(ref ** 2)))
    seg = csum[len(win):] - csum[:-len(win)]
    denom = wn * np.sqrt(np.maximum(seg, 1e-12))
    return corr / denom


def _tracked_peak(ncc, expect_idx):
    """Pick the alignment closest to the previous window's, among local
    maxima within 10% of the best. Music is periodic, so the global argmax
    can hop a whole cycle between windows and read as drift."""
    best = float(ncc.max())
    i = np.arange(1, len(ncc) - 1)
    peaks = i[(ncc[i] >= ncc[i - 1]) & (ncc[i] >= ncc[i + 1]) &
              (ncc[i] >= 0.9 * best)]
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(ncc))])
    pick = int(peaks[np.argmin(np.abs(peaks - expect_idx))])
    return pick, float(ncc[pick])


def _rms(x):
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def _floor_level(x, sr, win_ms=20.0, pct=10.0):
    """A low percentile of short-window RMS: the bed under the content."""
    win = max(1, int(round(win_ms / 1000.0 * sr)))
    n = len(x) // win
    if n < 3:
        return _rms(x)
    trimmed = x[:n * win].reshape(n, win)
    levels = np.sqrt(np.mean(np.square(trimmed), axis=1))
    return float(np.percentile(levels, pct))


def _step_ratio(a, b):
    denom = a + b
    if denom <= 1e-12:
        return 0.0
    return abs(a - b) / denom


def _db(a, b):
    if a <= 1e-12 or b <= 1e-12:
        return float("inf")
    return abs(20.0 * np.log10(a / b))


class MiniMaxH3MotionContextSeamProbe:
    """Measure a chain join: continuation quality and level step."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_b_untrimmed": ("AUDIO", {
                    "tooltip": "This clip's audio straight off the VAE "
                               "decode, BEFORE the trim node. It still "
                               "carries the pinned head, which is what "
                               "gets compared."}),
                "trim_frames": ("INT", {
                    "default": 0, "min": 0, "max": 4096,
                    "tooltip": "Wire this from the Motion Context node's "
                               "trim output, the same value the trim node "
                               "gets. It is the pinned span."}),
            },
            "optional": {
                "clip_a_latent": ("LATENT", {
                    "tooltip": "The PREVIOUS clip's AV latent: the same "
                               "one wired into Motion Context's "
                               "context_latent. The node decodes and "
                               "tail-matches it itself. Without it only "
                               "clip B is described, nothing is "
                               "measured."}),
                "audio_vae": ("VAE", {
                    "tooltip": "The H3 audio VAE, needed to decode clip "
                               "A's audio out of its latent."}),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0,
                    "step": 0.001,
                    "tooltip": "Must match what you feed Create Video."}),
                "window_ms": ("FLOAT", {
                    "default": 50.0, "min": 5.0, "max": 500.0,
                    "step": 1.0,
                    "tooltip": "Correlation analysis window."}),
                "search_ms": ("FLOAT", {
                    "default": 40.0, "min": 5.0, "max": 500.0,
                    "step": 1.0,
                    "tooltip": "Maximum lag searched either side."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report")
    OUTPUT_TOOLTIPS = (
        "clip_b_untrimmed, unchanged. Wire it on to the trim node.",
        "The measurement report, for a Preview Text node.",
    )
    FUNCTION = "probe"
    CATEGORY = "conditioning/minimax"
    OUTPUT_NODE = True
    DESCRIPTION = ("Measure a chain join in the graph: does clip B "
                   "continue clip A's audio, and does the level step at "
                   "the cut.")

    def probe(self, clip_b_untrimmed, trim_frames, clip_a_latent=None,
              audio_vae=None, fps=24.0, window_ms=50.0, search_ms=40.0):
        passthrough = clip_b_untrimmed
        sr = int(clip_b_untrimmed["sample_rate"])
        b = _mono(clip_b_untrimmed["waveform"])
        span_s = max(0, int(trim_frames)) / float(fps)
        span = int(round(span_s * sr))

        lines = ["H3 seam probe",
                 "clip B (untrimmed): %.4fs at %d Hz" % (len(b) / sr, sr),
                 "pinned span: %d frames = %.4fs = %d samples"
                 % (int(trim_frames), span_s, span)]

        if span <= 0:
            lines += ["", "trim_frames is 0, so there is no pinned span "
                          "and nothing to measure. Wire it from the "
                          "Motion Context node."]
            return self._finish(passthrough, lines)
        if len(b) < span:
            lines += ["", "clip B is shorter than the pinned span. This "
                          "is the TRIMMED audio; wire the VAE decode "
                          "output instead."]
            return self._finish(passthrough, lines)
        if clip_a_latent is None or audio_vae is None:
            missing = "clip_a_latent" if clip_a_latent is None else "audio_vae"
            lines += ["", "no %s wired: measuring the join needs the "
                          "previous clip's latent and the audio VAE to "
                          "decode it." % missing]
            return self._finish(passthrough, lines)

        try:
            a, raw, frames = self._clip_a_audio(clip_a_latent, audio_vae,
                                                sr, fps)
        except ValueError as exc:
            lines += ["", str(exc)]
            return self._finish(passthrough, lines)

        pad = len(a) - len(raw)
        lines.append("clip A: %d frames, decoded %.4fs, tail-matched to "
                     "%.4fs (%s %d samples)"
                     % (frames, len(raw) / sr, len(a) / sr,
                        "padded" if pad > 0 else "trimmed" if pad < 0
                        else "exact", abs(pad)))
        lines.append("")
        if len(a) < span:
            lines.append("clip A is shorter than the pinned span, so "
                         "there is nothing to compare against.")
            return self._finish(passthrough, lines)
        lines += self._continuation(a, b, sr, span, window_ms, search_ms)
        if pad > 0 and len(raw) >= span:
            # Padding lengthened the array without moving the content, so
            # the two anchors sit `pad` samples apart and one of them is
            # wrong. Measuring both says which: a lag that vanishes
            # against the raw end is an anchoring artifact, one that
            # survives both is in the render.
            alt = self._continuation(raw, b, sr, span, window_ms,
                                     search_ms)
            lines += ["", "cross-check: same audio anchored on the raw "
                          "decode instead of the padded end"]
            lines += ["  " + ln for ln in alt[1:]]
            lines.append("  the padding is %.2f ms, so these two should "
                         "differ by that much. Whichever reads near zero "
                         "is the true alignment."
                         % (pad / sr * 1000.0,))
        lines.append("")
        lines += self._level(a, b, sr, span)
        return self._finish(passthrough, lines)

    @staticmethod
    def _clip_a_audio(latent, audio_vae, sr, fps):
        """Decode clip A's audio and return (tail-matched, raw, frames).

        The tail is the only part that gets read, and its position is
        what the whole measurement is anchored on, so it has to sit
        exactly where the delivered clip's does. H3 rounds its audio grid
        to the nearest step, so a raw decode is up to a third of a step
        long or short; leaving that in would shift every reported lag by
        the same amount and look exactly like a real finding. The head
        trim is not applied because it does not move the tail.

        The raw decode comes back too, because tail-matching a SHORT clip
        pads it, and padding moves the end of the array without moving
        the content. On those clips the two arrays disagree about where
        the anchor is by exactly the overhang, so the probe measures
        against both and reports the difference rather than picking one
        and hoping.
        """
        parts = _streams_from_latent(latent)
        if len(parts) < 2:
            raise ValueError(
                "clip_a_latent has no audio stream. Wire the sampler "
                "output of an H3 AV graph, not a video-only latent.")
        video, audio = parts[0], parts[1]
        if video.ndim == 4:
            video = video.unsqueeze(0)
        frames = _pixel_frames(int(video.shape[2]))
        if audio.ndim == 3:
            audio = audio.unsqueeze(0)
        decoded = audio_vae.decode(audio[:1])
        a = _mono(decoded)
        vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
        if vae_sr != sr:
            raise ValueError(
                "clip A decodes at %d Hz but clip B is %d Hz. Not "
                "comparable." % (vae_sr, sr))
        want = int(round(frames / float(fps) * sr))
        have = len(a)
        raw = a
        if have > want:
            a = a[:want]
        elif have < want:
            a = np.concatenate([a, np.zeros(want - have)])
        return a, raw, frames

    def _continuation(self, a, b, sr, span, window_ms, search_ms):
        """B's first `span` samples should reconstruct A's last `span`."""
        win = max(1, int(round(window_ms / 1000.0 * sr)))
        hop = max(1, int(round(0.025 * sr)))  # one audio latent step
        search = max(1, int(round(search_ms / 1000.0 * sr)))
        a_tail_start = len(a) - span

        lags, corrs = [], []
        t, prev_lag = 0, None
        while t + win <= span:
            centre = a_tail_start + t
            lo = max(0, centre - search)
            hi = min(len(a), centre + win + search)
            ref = a[lo:hi]
            if len(ref) <= win:
                break
            ncc = _norm_xcorr(b[t:t + win], ref)
            if ncc is None:
                t += hop
                continue
            # positive lag = B is LATE, matching an earlier part of A
            if prev_lag is None:
                # Nothing to track from yet, so take the global peak. Not
                # zero: assuming the first window is aligned makes a join
                # that is uniformly offset by a whole audio step read as
                # the nearest cycle alias instead, which for periodic
                # content can be a fraction of the real number. That is
                # exactly the failure this probe exists to find.
                pick = int(np.argmax(ncc))
                corr = float(ncc[pick])
            else:
                expect = (centre - prev_lag) - lo
                pick, corr = _tracked_peak(ncc, expect)
            lag = centre - (lo + pick)
            lags.append(lag / sr * 1000.0)
            corrs.append(corr)
            if corr > CORR_CREDIBLE:
                prev_lag = lag
            t += hop

        out = ["continuation over the pinned span"]
        if not corrs:
            out.append("  window longer than the span; nothing analysed")
            return out
        corrs = np.array(corrs)
        lags = np.array(lags)
        strong = corrs > CORR_CREDIBLE
        out.append("  mean corr   %.3f   (%d/%d windows above %.1f)"
                   % (corrs.mean(), int(strong.sum()), len(corrs),
                      CORR_CREDIBLE))
        if strong.any():
            sl = lags[strong]
            out.append("  lag         %+.2f ms mean, %+.2f to %+.2f "
                       "(credible windows only)"
                       % (sl.mean(), sl.min(), sl.max()))
            at_edge = np.abs(sl) >= (search / sr * 1000.0 - 1.0)
            if at_edge.any():
                out.append("  %d window(s) pegged at the search limit; the "
                           "real lag is outside it. Raise search_ms."
                           % int(at_edge.sum()))
        else:
            out.append("  lag         not reported: no window correlated "
                       "well enough to trust an alignment")
        if corrs.mean() < 0.3:
            out.append("  reading: the pinned audio is not being continued "
                       "at all. Check that context_latent or "
                       "context_audio is actually wired.")
        elif corrs.mean() < CORR_CREDIBLE:
            out.append("  reading: weak. The model is imitating the tail "
                       "rather than continuing it.")
        elif strong.any() and abs(lags[strong].mean()) > 12.0:
            out.append("  reading: locked but offset by more than half an "
                       "audio step. Placement, not continuation.")
        else:
            out.append("  reading: clean continuation.")
        return out

    def _level(self, a, b, sr, span):
        """The delivered join: A's tail against B's first post-trim audio."""
        b_delivered = b[span:]
        n = min(int(0.5 * sr), len(a), len(b_delivered))
        out = ["level step across the delivered cut"]
        if n < int(0.05 * sr):
            out.append("  too little audio either side of the cut to "
                       "measure")
            return out
        left, right = a[-n:], b_delivered[:n]
        bb = _step_ratio(_rms(left), _rms(right))
        fl = _step_ratio(_floor_level(left, sr), _floor_level(right, sr))
        out.append("  broadband   %.3f  (%.1f dB)   %s"
                   % (bb, _db(_rms(left), _rms(right)),
                      "ok" if bb < BROADBAND_OK else "high"))
        out.append("  floor       %.3f  (%.1f dB)   %s"
                   % (fl, _db(_floor_level(left, sr),
                              _floor_level(right, sr)),
                      "clean" if fl < FLOOR_CLEAN
                      else "marginal" if fl < FLOOR_MARGINAL
                      else "audible step"))
        out.append("  measured over %.0f ms either side" % (n / sr * 1000.0,))
        if fl >= FLOOR_MARGINAL and bb < BROADBAND_OK:
            out.append("  reading: the music matches but the bed underneath "
                       "restarts at the cut. Wire the previous clip's "
                       "audio into Motion Context.")
        out.append("  note: the floor figure reads the quiet moments "
                   "between content, so wall to wall music with no rests "
                   "gives it nothing to measure and it collapses onto the "
                   "broadband number.")
        return out

    @staticmethod
    def _finish(passthrough, lines):
        report = "\n".join(lines)
        for line in lines:
            _LOG.info("h3_motion_context: %s", line)
        return {"ui": {"text": [report]},
                "result": (passthrough, report)}


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3MotionContextSeamProbe": MiniMaxH3MotionContextSeamProbe,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3MotionContextSeamProbe": "H3 Motion Context Seam Probe",
}
