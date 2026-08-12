"""Probe node test: drive the seam probe with joins whose answer is known.

No ComfyUI and no GPU. Builds audio where the continuation, the lag and
the level step are all constructed, then checks the node reports them.
Clip A goes in as a latent, so the fake audio VAE's decode is what the
node reads, exactly as it would in a graph.
"""

import importlib
import os
import sys
import types

import numpy as np

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, _TESTS_DIR)
sys.path.insert(0, os.path.dirname(_PKG_DIR))

from _mock_harness import make_mm, make_torch  # noqa: E402

SR = 32000
FPS = 24.0
TRIM = 22
SPAN = int(round(TRIM / FPS * SR))
# a legal clip length on the ladder: 5g+2 latent steps, 17g+5 frames
LATENT_T, FRAMES = 37, 124


def _install_fakes():
    """nodes.py imports ComfyUI, and the probe imports nodes.py.

    Only enough of each module to get through import: this test never
    runs the other nodes, so the fakes do not have to be faithful the way
    the smoke test's are.
    """
    mm = make_mm()
    for name in ("comfy", "comfy.ldm", "comfy.ldm.minimax"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["comfy.ldm.minimax.model"] = mm
    sys.modules["comfy"].ldm = sys.modules["comfy.ldm"]
    sys.modules["comfy.ldm"].minimax = sys.modules["comfy.ldm.minimax"]
    sys.modules["comfy.ldm.minimax"].model = mm
    sys.modules["torch"] = make_torch()

    # safetensors must be faked even though nodes.py never uses it here:
    # the real package imports torch at module scope, and with the fake
    # torch in sys.modules that dies with an AttributeError the ImportError
    # guard in nodes.py does not catch
    for name, attrs in (("comfy.utils", {}),
                        ("comfy.model_base", {"MiniMaxH3": type(
                            "MiniMaxH3", (), {"extra_conds": lambda s, **k: {}})}),
                        ("folder_paths", {"get_output_directory": lambda: ".",
                                          "get_save_image_path":
                                          lambda *a: (".", "x", 1, "", "")}),
                        ("node_helpers", {"conditioning_set_values":
                                          lambda c, v, **k: c}),
                        ("safetensors", {}),
                        ("safetensors.torch", {"load_file": None,
                                               "save_file": None})):
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
        if name.startswith("comfy."):
            setattr(sys.modules["comfy"], name.split(".", 1)[1], mod)


_install_fakes()
probe_node = importlib.import_module(
    "%s.probe_node" % os.path.basename(_PKG_DIR))
CORR_LIMIT = probe_node.CORR_CREDIBLE


class Tensor:
    """Enough of a tensor for the node's shape checks and the fake VAE."""

    def __init__(self, a):
        self.a = np.asarray(a)

    @property
    def shape(self):
        return self.a.shape

    @property
    def ndim(self):
        return self.a.ndim

    def __getitem__(self, idx):
        return Tensor(self.a[idx])

    def unsqueeze(self, d):
        return Tensor(np.expand_dims(self.a, d))

    def __array__(self, dtype=None, copy=None):
        # real torch tensors implement this, which is how np.asarray()
        # accepts one; without it numpy falls back to iterating the
        # object and the conversion fails
        return self.a if dtype is None else self.a.astype(dtype)


class Nested:
    def __init__(self, parts):
        self.parts = parts

    def unbind(self):
        return list(self.parts)


class FakeAudioVAE:
    """Decodes by handing back the waveform stashed on it.

    The point of the test is the node's anchoring and measurement, not
    the VAE, so the decode is a lookup. It returns [B, C, L], which the
    node must not assume the orientation of.
    """

    audio_sample_rate = SR

    def __init__(self, mono):
        self.mono = mono

    def decode(self, z):
        return Tensor(np.stack([self.mono, self.mono])[None, ...])


def _latent(mono, latent_t=LATENT_T):
    """An AV latent whose video stream says how many frames the clip is."""
    steps = max(1, int(round(len(mono) / SR * 40)))
    return {"samples": Nested([
        Tensor(np.zeros((1, 16, latent_t, 4, 4), dtype=np.float32)),
        Tensor(np.zeros((1, 32, 2, steps), dtype=np.float32)),
    ])}


def _audio(mono):
    return {"waveform": np.stack([mono, mono])[None, ...],
            "sample_rate": SR}


def _bed(n, seed, freq=110.0, amp=1.0, noise=0.02):
    """Periodic content plus a steady floor. The tone is deliberately
    periodic: it is what the peak tracker has to avoid cycle-hopping on."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * freq * t)
            + 0.3 * amp * np.sin(2 * np.pi * freq * 2.5 * t)
            + noise * rng.standard_normal(n))


def _gated(n, seed, floor_amp, freq=110.0, on_ms=120.0, off_ms=80.0):
    """Tone bursts over a continuous noise bed.

    The floor metric reads the quiet moments between content, so a fully
    sustained tone gives it nothing and it collapses onto broadband. That
    limitation is documented in level_step.py, so the fixture for a floor
    step has to have rests in it.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * freq * t)
    period = int(round((on_ms + off_ms) / 1000.0 * SR))
    on = int(round(on_ms / 1000.0 * SR))
    gate = (np.arange(n) % period) < on
    return tone * gate + floor_amp * rng.standard_normal(n)


def _run(a_mono, b_mono, **kw):
    node = probe_node.MiniMaxH3MotionContextSeamProbe()
    out = node.probe(clip_b_untrimmed=_audio(b_mono), trim_frames=TRIM,
                     clip_a_latent=_latent(a_mono),
                     audio_vae=FakeAudioVAE(a_mono), fps=FPS, **kw)
    return out["result"][0], out["result"][1]


def _num(report, label):
    for line in report.splitlines():
        s = line.strip()
        if s.startswith(label):
            for tok in s[len(label):].split():
                try:
                    return float(tok)
                except ValueError:
                    continue
    raise AssertionError("no %r line in:\n%s" % (label, report))


def main():
    # clip A is exactly FRAMES long, which is what the latent declares.
    # The node tail-matches to that, so the fixtures must agree.
    a_len = int(round(FRAMES / FPS * SR))
    source = _bed(a_len + SPAN * 3, seed=1)
    a = source[:a_len]

    # 1. perfect continuation: B's head IS A's tail
    b = np.concatenate([a[-SPAN:], source[a_len:a_len + SPAN * 2]])
    passthru, rep = _run(a, b)
    corr, lag = _num(rep, "mean corr"), _num(rep, "lag")
    assert corr > 0.95, (corr, rep)
    assert abs(lag) < 1.0, (lag, rep)
    assert "clean continuation" in rep, rep
    assert passthru["sample_rate"] == SR
    print("1. exact continuation: corr %.3f, lag %+.2f ms" % (corr, lag))

    # 2. late by one audio step
    shift = int(round(0.025 * SR))
    b_late = np.concatenate([source[a_len - SPAN - shift:a_len - shift],
                             source[a_len:a_len + SPAN * 2]])
    _, rep = _run(a, b_late)
    corr, lag = _num(rep, "mean corr"), _num(rep, "lag")
    assert corr > 0.9, (corr, rep)
    assert abs(lag - 25.0) < 2.0, (lag, rep)
    print("2. one step late: corr %.3f, lag %+.2f ms (built as +25.0)"
          % (corr, lag))

    # 3. imitation, not continuation. A different noise seed alone is not
    #    enough, the deterministic tone would still correlate at 0.99.
    b_fake = np.concatenate([_bed(SPAN, seed=99, freq=173.0),
                             source[a_len:a_len + SPAN * 2]])
    _, rep = _run(a, b_fake)
    corr = _num(rep, "mean corr")
    assert corr < CORR_LIMIT, (corr, rep)
    assert "imitating" in rep or "not being continued" in rep, rep
    print("3. unrelated head: corr %.3f, flagged as imitation" % corr)

    # 4. same music either side, but B's bed is 30 dB quieter. Invisible
    #    to broadband because the content matches.
    gated_a = _gated(a_len, seed=11, floor_amp=0.02)
    loud = np.concatenate([gated_a[-SPAN:],
                           _gated(SPAN * 2, seed=12, floor_amp=0.02)])
    quiet = np.concatenate([gated_a[-SPAN:],
                            _gated(SPAN * 2, seed=12, floor_amp=0.0006)])
    _, rep = _run(gated_a, quiet)
    floor, bb = _num(rep, "floor"), _num(rep, "broadband")
    assert floor > 0.5, (floor, rep)
    assert bb < 0.4, (bb, rep)
    assert "bed underneath" in rep, rep
    print("4. floor drop: floor %.3f (audible), broadband %.3f (ok), "
          "diagnosed as ambience restart" % (floor, bb))

    # 5. no level step when both sides share a bed
    _, rep = _run(gated_a, loud)
    floor = _num(rep, "floor")
    assert floor < 0.25, (floor, rep)
    assert "bed underneath" not in rep, rep
    print("5. matched bed: floor step %.3f, below the clean threshold"
          % floor)

    # 6. a lag outside the search window must not become a confident
    #    number. On non-periodic content nothing correlates, so no lag.
    rng = np.random.default_rng(21)
    na = rng.standard_normal(a_len)
    big = int(round(0.060 * SR))
    nb = np.concatenate([na[-SPAN - big:-big],
                         rng.standard_normal(SPAN * 2)])
    _, rep = _run(na, nb, search_ms=20.0)
    assert "not reported" in rep, rep
    assert _num(rep, "mean corr") < CORR_LIMIT, rep
    print("6. lag outside the search window: no lag reported rather than "
          "a confident wrong one")

    # 7. the aliasing limitation, asserted rather than left implicit.
    #    Beat-driven music is exactly this case.
    period_ms = 1000.0 / 55.0  # 110 and 275 Hz share a 55 Hz fundamental
    b_alias = np.concatenate([source[a_len - SPAN - big:a_len - big],
                              source[a_len:a_len + SPAN * 2]])
    _, rep = _run(a, b_alias, search_ms=20.0)
    lag = _num(rep, "lag")
    assert _num(rep, "mean corr") > 0.9, rep
    assert abs(lag - 60.0) > 10.0, (lag, rep)
    assert abs(((60.0 - lag) / period_ms)
               - round((60.0 - lag) / period_ms)) < 0.05, (lag, rep)
    print("7. periodic aliasing: 60.0 ms offset read as %+.2f ms, a whole "
          "number of cycles out (known limitation)" % lag)

    # 8. the node must tail-match clip A itself. Hand it a decode that is
    #    a third of an audio step long, the real grid overhang, and the
    #    lag must stay at zero. Without the internal match this reads as
    #    a confident 8.3 ms, which is exactly the artifact the node
    #    exists to remove.
    over = int(round(SR / 40.0 / 3.0))
    a_long = np.concatenate([a, source[a_len:a_len + over]])
    node = probe_node.MiniMaxH3MotionContextSeamProbe()
    out = node.probe(clip_b_untrimmed=_audio(b), trim_frames=TRIM,
                     clip_a_latent=_latent(a),  # declares FRAMES frames
                     audio_vae=FakeAudioVAE(a_long), fps=FPS)
    lag = _num(out["result"][1], "lag")
    assert abs(lag) < 1.0, (lag, out["result"][1])
    print("8. overlong decode (%d samples, %.2f ms): trimmed internally, "
          "lag stays %+.2f ms" % (over, over / SR * 1000.0, lag))

    # 9. wrong wiring refused with a reason, audio still passed through
    node = probe_node.MiniMaxH3MotionContextSeamProbe()
    cases = [
        ("TRIMMED", dict(clip_b_untrimmed=_audio(b[SPAN:][:SPAN // 2]),
                         trim_frames=TRIM, clip_a_latent=_latent(a),
                         audio_vae=FakeAudioVAE(a))),
        ("trim_frames is 0", dict(clip_b_untrimmed=_audio(b),
                                  trim_frames=0, clip_a_latent=_latent(a),
                                  audio_vae=FakeAudioVAE(a))),
        ("no clip_a_latent", dict(clip_b_untrimmed=_audio(b),
                                  trim_frames=TRIM)),
        ("no audio_vae", dict(clip_b_untrimmed=_audio(b),
                              trim_frames=TRIM,
                              clip_a_latent=_latent(a))),
    ]
    for needle, kwargs in cases:
        res = node.probe(fps=FPS, **kwargs)
        assert needle in res["result"][1], (needle, res["result"][1])
        assert res["result"][0]["sample_rate"] == SR
    vid_only = {"samples": Nested([
        Tensor(np.zeros((1, 16, LATENT_T, 4, 4), dtype=np.float32))])}
    res = node.probe(clip_b_untrimmed=_audio(b), trim_frames=TRIM,
                     clip_a_latent=vid_only, audio_vae=FakeAudioVAE(a),
                     fps=FPS)
    assert "no audio stream" in res["result"][1], res["result"][1]
    print("9. bad wiring: trimmed input, zero span, missing latent, "
          "missing VAE and video-only latent each refused with a reason, "
          "audio still passed through")

    # 10. a SHORT clip A gets padded, and padding moves the end of the
    #     array without moving the content. So the padded anchor and the
    #     raw one disagree by exactly the pad, and the node must report
    #     both rather than pick one. Built here so the raw anchor is the
    #     correct one: B's head is A's real tail.
    short = int(round(SR / 40.0 / 3.0))
    a_short = a[:a_len - short]
    b_short = np.concatenate([a_short[-SPAN:],
                              source[a_len:a_len + SPAN * 2]])
    node = probe_node.MiniMaxH3MotionContextSeamProbe()
    out = node.probe(clip_b_untrimmed=_audio(b_short), trim_frames=TRIM,
                     clip_a_latent=_latent(a),  # declares FRAMES frames
                     audio_vae=FakeAudioVAE(a_short), fps=FPS)
    rep = out["result"][1]
    assert "cross-check" in rep, rep
    assert "padded %d samples" % short in rep, rep
    lags = [float(l.strip()[len("lag"):].split()[0])
            for l in rep.splitlines() if l.strip().startswith("lag")]
    assert len(lags) == 2, (lags, rep)
    padded_lag, raw_lag = lags
    assert abs(raw_lag) < 1.0, (raw_lag, rep)
    assert abs(abs(padded_lag - raw_lag) - short / SR * 1000.0) < 1.0, \
        (padded_lag, raw_lag, rep)
    print("10. short clip A: both anchors reported, %+.2f ms padded vs "
          "%+.2f ms raw, differing by the %.2f ms pad"
          % (padded_lag, raw_lag, short / SR * 1000.0))

    print("probe node test passed")


if __name__ == "__main__":
    main()
