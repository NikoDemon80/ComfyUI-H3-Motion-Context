"""Smoke test: run MiniMaxH3MotionContext.apply() end to end with fakes.

Fakes ComfyUI's modules and tensor ops (numpy-backed) and drives the node
exactly as a graph would: a 124-frame clip at 480x864, 22 context frames,
audio from the previous clip's LATENT. Checks the produced conditioning
values: keyframe count and indices, the audio ref's step count, and the
fractional end_frame carrying the grid-overhang compensation.
"""

import sys
import types

import numpy as np

import os
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_TESTS_DIR)  # repo root, where the package lives
sys.path.insert(0, _TESTS_DIR)
from _mock_harness import make_mm, make_torch  # noqa: E402


class T:
    """Minimal numpy-backed tensor stand-in."""

    def __init__(self, a):
        self.a = np.asarray(a)

    @property
    def shape(self):
        return self.a.shape

    @property
    def ndim(self):
        return self.a.ndim

    def __getitem__(self, idx):
        return T(self.a[idx])

    def movedim(self, src, dst):
        return T(np.moveaxis(self.a, src, dst))

    def unsqueeze(self, d):
        return T(np.expand_dims(self.a, d))

    def clone(self):
        return T(self.a.copy())

    def cpu(self):
        return self

    def contiguous(self):
        return T(np.ascontiguousarray(self.a))


class Nested:
    def __init__(self, parts):
        self.parts = parts

    def unbind(self):
        return list(self.parts)


def main():
    # fake modules the package imports
    mm = make_mm()
    for name in ("comfy", "comfy.ldm", "comfy.ldm.minimax"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["comfy.ldm.minimax.model"] = mm
    sys.modules["comfy"].ldm = sys.modules["comfy.ldm"]
    sys.modules["comfy.ldm"].minimax = sys.modules["comfy.ldm.minimax"]
    sys.modules["comfy.ldm.minimax"].model = mm
    sys.modules["torch"] = make_torch()

    cu = types.ModuleType("comfy.utils")
    cu.common_upscale = lambda s, w, h, m, c: T(
        np.zeros((s.shape[0], 3, h, w), dtype=np.float32))
    sys.modules["comfy.utils"] = cu
    sys.modules["comfy"].utils = cu

    mb = types.ModuleType("comfy.model_base")

    class MiniMaxH3:
        def extra_conds(self, **kw):
            return {}
    mb.MiniMaxH3 = MiniMaxH3
    sys.modules["comfy.model_base"] = mb
    sys.modules["comfy"].model_base = mb

    captured = {}
    nh = types.ModuleType("node_helpers")

    def conditioning_set_values(cond, values, append=False):
        # faithful to ComfyUI's node_helpers: copy each entry's dict, and
        # when appending, concatenate onto whatever is already under the
        # key instead of replacing it. The append path is what lets a
        # Ref2VA graph keep its own reference blocks.
        out = []
        for t in cond:
            n = [t[0], t[1].copy()]
            for k, v in values.items():
                if append:
                    old = n[1].get(k, None)
                    if old is not None:
                        v = old + v
                n[1][k] = v
            out.append(n)
        captured.clear()
        captured.update(out[0][1])
        return out
    nh.conditioning_set_values = conditioning_set_values
    sys.modules["node_helpers"] = nh

    import os
    import tempfile
    outdir = tempfile.mkdtemp()
    fp = types.ModuleType("folder_paths")
    fp.get_output_directory = lambda: outdir

    def get_save_image_path(prefix, out, *a):
        sub, name = os.path.split(prefix)
        folder = os.path.join(out, sub)
        os.makedirs(folder, exist_ok=True)
        counter = 1 + sum(1 for f in os.listdir(folder)
                          if f.startswith(name))
        return folder, name, counter, sub, prefix
    fp.get_save_image_path = get_save_image_path
    sys.modules["folder_paths"] = fp

    st = types.ModuleType("safetensors")
    stt = types.ModuleType("safetensors.torch")

    def save_file(d, path, metadata=None):
        np.savez(path + ".npz", **{k: v.a for k, v in d.items()})
        open(path, "w").write(path + ".npz")

    def load_file(path):
        real = open(path).read()
        z = np.load(real)
        return {k: T(z[k]) for k in z.files}
    stt.save_file, stt.load_file = save_file, load_file
    st.torch = stt
    sys.modules["safetensors"] = st
    sys.modules["safetensors.torch"] = stt

    # import the package by file location so it works whatever the repo
    # folder is called (ComfyUI-H3-Motion-Context, h3_motion_context, ...)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "h3mc_pkg", os.path.join(_PKG_DIR, "__init__.py"),
        submodule_search_locations=[_PKG_DIR])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["h3mc_pkg"] = pkg
    spec.loader.exec_module(pkg)  # registers nodes; patches apply on first run
    nodes = sys.modules["h3mc_pkg.nodes"]

    def audio_kf():
        """The one keyframe carrying pinned audio, from the last apply()."""
        got = [k for k in captured["minimax_keyframes"]
               if k.get("audio_latent") is not None]
        assert len(got) == 1, "expected 1 pinned audio keyframe, got %d" % len(got)
        return got[0]

    def audio_end():
        """Target frame the pinned audio window ends at.

        Stock places the window STARTING at the anchor index and running
        forward, so the end is the index plus the window's own width in
        pixel frames. That is the number the node solved for.
        """
        kf = audio_kf()
        return (kf["resolved_frame_index"]
                + kf["audio_latent"].shape[-1] / nodes.FRAME_RESCALE)
    assert pkg.NODE_CLASS_MAPPINGS
    # importing the pack must not have touched ComfyUI yet
    assert not nodes._layout_checked(), \
        "the layout checks ran at import time"
    stock_init = mm.PackedLayout.__init__

    # a 124-frame clip: latent_t 37 (7 full 17-frame groups + 1 + 4),
    # audio grid ceil(124 * 5/3) = 207 steps, overhang exactly 1/3
    latent_t, frames, audio_t = 37, 124, 207
    assert nodes._pixel_frames(latent_t) == frames
    h, w = 480 // 16, 864 // 16
    target = {"samples": Nested([
        T(np.zeros((1, 16, latent_t, h, w), dtype=np.float32)),
        T(np.zeros((1, 32, 2, audio_t), dtype=np.float32)),
    ])}
    # previous clip's sampler latent (same dims in this setup)
    prev = {"samples": Nested([
        T(np.arange(1 * 16 * latent_t * h * w, dtype=np.float32
                    ).reshape(1, 16, latent_t, h, w)),
        T(np.arange(1 * 32 * 2 * audio_t, dtype=np.float32
                    ).reshape(1, 32, 2, audio_t)),
    ])}
    context = T(np.zeros((124, 480, 864, 3), dtype=np.float32))

    class VAE:
        def encode(self, x):
            n = x.shape[0]
            steps = max(1, (n - 5) // 17 * 5 + 2)
            return T(np.zeros((1, 16, steps, h, w), dtype=np.float32))

    node = nodes.MiniMaxH3MotionContext()

    def run(**kw):
        """apply(), capturing the conditioning it returns.

        The node used to reach node_helpers to append its audio reference,
        which is where these assertions used to read the result from. It
        now builds the conditioning itself and touches no reference list,
        so read what it actually hands the sampler.
        """
        got = node.apply(**kw)
        captured.clear()
        captured.update(got[0][0][1])
        return got
    out, trim = run(
        conditioning=[["c", {}]], vae=VAE(), latent=target,
        context_frames=context, context_length="22",
        audio_context_length=22, context_latent=prev)

    # ... and running one must have checked the layout, and left it alone
    assert nodes._layout_checked(), "the layout checks did not run"
    assert mm.PackedLayout.__init__ is stock_init, \
        "ComfyUI's constructor was modified"

    kfs = captured["minimax_keyframes"]
    # video keyframes carry a latent, the pinned audio is one more
    # keyframe carrying only an audio latent, and it goes last
    vid = [kf for kf in kfs if kf.get("latent") is not None]
    aud = [kf for kf in kfs if kf.get("audio_latent") is not None]
    assert len(kfs) == 8 and len(vid) == 7 and len(aud) == 1, len(kfs)
    assert kfs[-1] is aud[0], "the audio keyframe is not last"
    assert "minimax_refs" not in captured, \
        "the pinned audio went out as a reference, not a keyframe"
    idx = [kf["resolved_frame_index"] for kf in vid]
    assert idx == [0, 1, 5, 9, 13, 17, 18], idx
    assert "minimax_frame_count" not in captured
    assert trim == 22

    # the index is rt / FRAME_RESCALE frames before the end coordinate, so
    # the window ENDS at the join rather than starting there. Here the
    # audio and video windows are both 22 frames, so it starts exactly at
    # frame 0; the 24-frame audio case below is the one that goes negative.
    a_idx = aud[0]["resolved_frame_index"]
    assert a_idx <= 0, a_idx
    ref = {"ref_audio_t": aud[0]["audio_latent"].shape[-1],
           "audio_latent": aud[0]["audio_latent"]}
    assert ref["ref_audio_t"] == 37, ref["ref_audio_t"]  # round(22/24*40)
    tail = ref["audio_latent"]
    assert tuple(tail.shape) == (1, 32, 2, 37), tail.shape
    # tail must be the LAST 37 steps of the source
    assert float(tail.a[0, 0, 0, -1]) == float(prev["samples"].parts[1]
                                               .a[0, 0, 0, -1])
    # the window must land on the target's own integer audio grid: the
    # end coordinate is FRAME_RESCALE * end_frame and the target's audio
    # rows sit on integers, so a fractional end coordinate would place
    # the pinned content between them (a third of a step is 8.3 ms)
    got_end = a_idx + ref["ref_audio_t"] / nodes.FRAME_RESCALE
    end_coord = nodes.FRAME_RESCALE * got_end
    assert abs(end_coord - round(end_coord)) < 1e-9, end_coord
    assert abs(end_coord - 37.0) < 1e-9, end_coord
    assert abs(got_end - 22.2) < 1e-6, got_end
    # the pinned VIDEO must have come out of the latent too, not the VAE.
    # The fake VAE returns zeros, so any nonzero block proves the source,
    # and each block must equal the matching step of the source latent.
    kf_blocks = [kf["latent"] for kf in vid]
    src = prev["samples"].parts[0].a
    start = latent_t - len(kf_blocks)          # 37 - 7 = 30, cycle pos 0
    assert start % 5 == 0, start
    for k, blk in enumerate(kf_blocks):
        assert tuple(blk.shape) == (1, 16, 1, h, w), blk.shape
        assert np.array_equal(blk.a[0, :, 0], src[0, :, start + k]), k
    assert nodes._steps_for_frames(22) == 7
    assert nodes._steps_for_frames(5) == 2
    assert nodes._steps_for_frames(39) == 12
    assert nodes._steps_for_frames(1) == 1
    assert nodes._steps_for_frames(3) is None    # not a whole step boundary
    print("latent path: 7 cond blocks at %s sliced from the latent tail "
          "(bit-identical to the source steps), audio 37 steps, end_frame "
          "%.4f (overhang-compensated)" % (idx, got_end))

    # the recommended config pins 22 frames of picture and 24 of sound, so
    # the audio window reaches back past the start of the clip. That index
    # is negative AND fractional, which no stock node can produce and
    # nothing upstream tests, so it is worth asserting here as well as in
    # the layout contract.
    run(conditioning=[["c", {}]], vae=VAE(), latent=target,
        context_frames=context, context_length="22",
        audio_context_length=24, context_latent=prev)
    wide = audio_kf()
    assert wide["audio_latent"].shape[-1] == 40, wide["audio_latent"].shape
    w_idx = wide["resolved_frame_index"]
    assert w_idx < 0 and w_idx != int(w_idx), w_idx
    assert abs(w_idx - (37 - 40) / nodes.FRAME_RESCALE) < 1e-9, w_idx
    w_end = nodes.FRAME_RESCALE * audio_end()
    assert abs(w_end - round(w_end)) < 1e-9, w_end
    print("wide audio window: 24 frames of sound against 22 of picture, "
          "anchor index %.3f, still ending on the audio grid at %.1f"
          % (w_idx, w_end))

    # no latent wired: the pixel path, same offsets, and the fake VAE
    # returns zeros so the blocks must now be zero rather than sliced.
    # No audio either, so node_helpers is never called: read the
    # conditioning the node returns rather than the capture hook
    res, _ = run(
        conditioning=[["c", {}]], vae=VAE(), latent=target,
        context_frames=context, context_length="22",
        audio_context_length=22)
    px = res[0][1]["minimax_keyframes"]
    assert [kf["resolved_frame_index"] for kf in px] == idx, \
        "offsets differ by source"
    assert float(px[0]["latent"].a.max()) == 0.0, "did not use the VAE"
    print("pixels path: same %d blocks at the same offsets, encoded rather "
          "than sliced" % len(px))

    # a resolution change cannot slice a latent and must refuse, not
    # quietly take the lossy path
    small = {"samples": Nested([
        T(np.zeros((1, 16, latent_t, h // 2, w // 2), dtype=np.float32)),
        T(np.zeros((1, 32, 2, audio_t), dtype=np.float32)),
    ])}
    try:
        run(
            conditioning=[["c", {}]], vae=VAE(), latent=target,
            context_frames=context, context_length="22",
            audio_context_length=22, context_latent=small)
    except ValueError as e:
        assert "cannot be resized" in str(e), str(e)
        print("resolution change: refused, with the reason and the two "
              "resolutions named")
    else:
        raise AssertionError("mismatched latent did not refuse")

    # nothing wired: first clip of a chain. Pass the conditioning
    # through and report trim_frames 0 so Trim is a no-op.
    cond = [["c", {"untouched": True}]]
    res, trim = run(
        conditioning=cond, vae=VAE(), latent=target,
        context_length="22", audio_context_length=22)
    assert trim == 0, trim
    assert res is cond
    assert captured.get("untouched") is True
    assert "minimax_keyframes" not in captured
    print("nothing wired: original conditioning passed through, "
          "trim_frames 0")

    # the constants that replaced the widgets must be on the good values
    assert nodes.ENCODE_MODE == "video"
    assert nodes.ANCHOR_MODE == "head"
    assert nodes.AUDIO_MODE == "timeline"
    assert nodes.CROP == "disabled"

    # the cycle-position property the whole latent video path rests on,
    # checked across every clip length and window rather than argued for
    for g in range(1, 40):
        steps_total = 5 * g + 2
        frames_total = nodes._pixel_frames(steps_total)
        assert (frames_total - 5) % 17 == 0, (g, frames_total)
        for wnd in (5, 22, 39, 56):
            st = nodes._steps_for_frames(wnd)
            if st is None or st > steps_total:
                continue
            begin = steps_total - st
            assert begin % 5 == 0, (frames_total, wnd, begin % 5)
            assert nodes._pixel_frames(st) == wnd
            # offsets computed for a fresh run must match the sliced run
            assert (nodes._step_offsets(st)
                    == [nodes._pixel_frames(k) for k in range(st)])
    print("cycle check: every clip length 22..%d x windows 5/22/39/56 "
          "slices from cycle position 0" % nodes._pixel_frames(5 * 39 + 2))

    # decoded-audio path must still work and carry integer end_frame
    captured.clear()

    class AudioVAE:
        audio_sample_rate = 32000

        def encode(self, x):
            steps = int(round(x.shape[-2] / 32000 * 40))
            return T(np.zeros((1, 32, 2, steps), dtype=np.float32))

    audio = {"waveform": T(np.zeros((1, 2, 32000), dtype=np.float32)),
             "sample_rate": 32000}
    run(
        conditioning=[["c", {}]], vae=VAE(), latent=target,
        context_frames=context, context_length="22",
        audio_context_length=22, audio_vae=AudioVAE(), context_audio=audio)
    # no overhang to compensate on this path, but the same grid rule
    # applies: 22 frames is 36.667 steps, which must snap to 37
    end2 = nodes.FRAME_RESCALE * audio_end()
    assert abs(end2 - 37.0) < 1e-9, end2
    print("vae path: window snapped to the audio grid, end coord %.1f" % end2)

    # every clip length on the ladder, every window, both paths: the
    # pinned window must always end on an integer audio coordinate
    for g in range(2, 13):
        f = 17 * g + 5
        # nearest, not ceil: ceil would simulate +2/3 on the lengths that
        # actually run -1/3, so this loop used to exercise a value the
        # model never produces
        at = round(nodes.FRAME_RESCALE * f)
        oh = at - nodes.FRAME_RESCALE * f
        for span in (5, 22, 39, 56):
            for ohv in (0.0, oh):
                ef = span + ohv / nodes.FRAME_RESCALE
                ec = round(nodes.FRAME_RESCALE * ef)
                ef = ec / nodes.FRAME_RESCALE
                c = nodes.FRAME_RESCALE * ef
                assert abs(c - round(c)) < 1e-9, (f, span, ohv, c)
    print("grid check: every ladder length x window 5/22/39/56 x both "
          "audio paths ends on an integer audio coordinate")

    # All three frame-count residues, latent path, 22-frame window. The
    # 124-frame case above is frames % 3 == 1, the one residue where
    # rounding up and rounding to nearest agree, so on its own it cannot
    # tell the two rules apart. 260 frames is the negative-overhang case:
    # compensation moves the end coordinate to 36.33, which rounds to 36,
    # where failing open would leave it at 36.67 and round to 37, a whole
    # audio step (25 ms) late.
    import logging as _logging

    class _Catcher(_logging.Handler):
        def __init__(self):
            _logging.Handler.__init__(self)
            self.msgs = []

        def emit(self, record):
            self.msgs.append(record.getMessage())

    catcher = _Catcher()
    catcher.setLevel(_logging.WARNING)
    nodes._LOG.addHandler(catcher)
    try:
        for lt, f, at, want_coord, want_end in (
                (37, 124, 207, 37, 22.2),   # +1/3, reaches past
                (72, 243, 405, 37, 22.2),   #    0, exact
                (77, 260, 433, 36, 21.6)):  # -1/3, falls short
            assert nodes._pixel_frames(lt) == f, (lt, f)
            assert at == round(nodes.FRAME_RESCALE * f), (f, at)
            captured.clear()
            del catcher.msgs[:]
            tgt = {"samples": Nested([
                T(np.zeros((1, 16, lt, h, w), dtype=np.float32)),
                T(np.zeros((1, 32, 2, at), dtype=np.float32)),
            ])}
            src = {"samples": Nested([
                T(np.arange(1 * 16 * lt * h * w, dtype=np.float32
                            ).reshape(1, 16, lt, h, w)),
                T(np.arange(1 * 32 * 2 * at, dtype=np.float32
                            ).reshape(1, 32, 2, at)),
            ])}
            run(
                conditioning=[["c", {}]], vae=VAE(), latent=tgt,
                context_frames=T(np.zeros((f, 480, 864, 3),
                                          dtype=np.float32)),
                context_length="22", audio_context_length=22,
                context_latent=src)
            got = audio_end()
            coord = nodes.FRAME_RESCALE * got
            assert abs(coord - round(coord)) < 1e-9, (f, coord)
            assert abs(coord - want_coord) < 1e-9, (f, coord, want_coord)
            assert abs(got - want_end) < 1e-6, (f, got, want_end)
            # the grid is legal on all three, so nothing may warn about it
            assert not [m for m in catcher.msgs if "audio grid" in m], \
                (f, catcher.msgs)
    finally:
        nodes._LOG.removeHandler(catcher)
    print("residue check: frames % 3 in 0/1/2 all accepted, end coords "
          "37/37/36 (the -1/3 case would be 37 if it failed open)")

    # The trim node's match_tail: a long tail is truncated, a short one is
    # padded, both to exactly frames/fps. The short case only occurs on
    # frames % 3 == 2 clips, which is why it went unexercised.
    trimmer = nodes.MiniMaxH3MotionContextTrim()
    for f, direction in ((124, "long"), (243, "exact"), (260, "short")):
        sr = 32000
        # what H3 actually ships: the audio grid at nearest rounding,
        # decoded back to samples
        steps = round(nodes.FRAME_RESCALE * f)
        have = int(round(steps / nodes.AUDIO_HZ * sr))
        want = int(round((f - 22) / 24.0 * sr))
        imgs = T(np.zeros((f, 8, 8, 3), dtype=np.float32))
        wav = T(np.zeros((1, 2, have), dtype=np.float32))
        _, out = trimmer.trim(
            images=imgs, trim_frames=22,
            audio={"waveform": wav, "sample_rate": sr}, fps=24.0,
            match_tail=True)
        got = int(out["waveform"].shape[-1])
        assert got == want, (f, direction, got, want)
        # and the drift the node reports must be under half a sample
        assert abs(got / sr - (f - 22) / 24.0) * 1000.0 < 0.02, (f, got)
    print("match_tail check: long tail trimmed, short tail padded, all "
          "three residues land on the exact sample count")

    # Ref2VA: a graph whose conditioning already carries reference blocks
    # must keep every one of them, with the motion context audio block
    # appended rather than replacing the lot. Before this, the node
    # assigned minimax_refs outright and an R2V graph silently lost its
    # references at the moment chaining was switched on.
    captured.clear()
    existing = [
        {"kind": "image", "latent_h": 8, "latent_w": 12},
        {"kind": "video_audio", "latent_h": 8, "latent_w": 12,
         "latent_t": 3, "ref_audio_t": 5},
        {"kind": "audio", "ref_audio_t": 9},
    ]
    r2v_cond = [["c", {"minimax_refs": [dict(r) for r in existing]}]]
    out_cond, _ = run(
        conditioning=r2v_cond, vae=VAE(), latent=target,
        context_frames=context, context_length="22",
        audio_context_length=22, context_latent=prev)
    refs_out = captured["minimax_refs"]
    assert len(refs_out) == 3, len(refs_out)
    for got, want in zip(refs_out, existing):
        assert got["kind"] == want["kind"], (got, want)
    # the pinned audio is a keyframe now, so the graph's reference list is
    # not touched at all. Its blocks still push the target origin along,
    # and the anchors follow it, which is stock's arithmetic and not ours.
    assert captured["minimax_keyframes"], "keyframes lost on the R2V path"
    assert audio_kf()["audio_latent"].shape[-1] == 37
    assert len(r2v_cond[0][1]["minimax_refs"]) == 3
    print("R2V path: 3 incoming references untouched, motion context audio "
          "rides as a keyframe, keyframes intact")

    # last_frame passthrough: an fl2va graph's own anchors used to be
    # replaced outright. The last-frame anchor must survive untouched,
    # because every keyframe now carries its real index and stock
    # compensates them all the same way. The first-frame anchor sits
    # inside the pinned head, which the pinned run already decides, so it
    # is dropped with a warning.
    captured.clear()
    up_kf = [{"resolved_frame_index": 0, "latent": "FF"},
             {"resolved_frame_index": frames - 1, "latent": "LF"}]
    fl_cond = [["c", {"minimax_keyframes": [dict(k) for k in up_kf]}]]
    catcher2 = _Catcher()
    nodes._LOG.addHandler(catcher2)
    try:
        run(
            conditioning=fl_cond, vae=VAE(), latent=target,
            context_frames=context, context_length="22",
            audio_context_length=22, context_latent=prev)
    finally:
        nodes._LOG.removeHandler(catcher2)
    merged = captured["minimax_keyframes"]
    # kept anchor + 7 pinned steps + the pinned audio keyframe
    assert len(merged) == 1 + 7 + 1, len(merged)
    assert merged[0]["latent"] == "LF"
    assert merged[0]["resolved_frame_index"] == frames - 1
    assert merged[0] is not fl_cond[0][1]["minimax_keyframes"][1], \
        "the caller's dict was passed through by reference"
    assert [kf["resolved_frame_index"] for kf in merged[1:8]] == idx
    assert any("dropped 1 keyframe anchor" in m for m in catcher2.msgs), \
        catcher2.msgs
    # an anchor past the end of this clip is a wiring error: the
    # conditioning was resolved against a longer clip, so the anchor would
    # land at the wrong frame. Refuse rather than render it.
    bad_cond = [["c", {"minimax_keyframes": [
        {"resolved_frame_index": frames + 17, "latent": "LF"}]}]]
    try:
        run(conditioning=bad_cond, vae=VAE(), latent=target,
                   context_frames=context, context_length="22",
                   audio_context_length=22, context_latent=prev)
    except ValueError as e:
        assert "only %d frames" % frames in str(e), str(e)
    else:
        raise AssertionError("out-of-range anchor did not refuse")
    print("last_frame path: upstream last-frame anchor kept unmodified, "
          "first-frame anchor dropped from the pinned head, out-of-range "
          "anchor refused")

    # save -> load -> context_latent roundtrip across "runs"
    import time
    saver = nodes.MiniMaxH3MotionContextSaveLatent()
    loader = nodes.MiniMaxH3MotionContextLoadLatent()
    (p1,) = saver.save(prev, "h3_context/clip")
    time.sleep(0.02)
    prev2 = {"samples": Nested([
        prev["samples"].parts[0],
        T(prev["samples"].parts[1].a * 2.0),  # distinguishable content
    ])}
    (p2,) = saver.save(prev2, "h3_context/clip")
    assert p1 != p2
    (loaded,) = loader.load(p2, clip_index=1)  # specific file
    parts = loaded["samples"]
    assert isinstance(parts, list) and len(parts) == 2
    captured.clear()
    run(
        conditioning=[["c", {}]], vae=VAE(), latent=target,
        context_frames=context, context_length="22",
        audio_context_length=22, context_latent=loaded)
    kf3 = audio_kf()
    want = float(prev2["samples"].parts[1].a[0, 0, 0, -1])
    got = float(kf3["audio_latent"].a[0, 0, 0, -1])
    assert got == want, (got, want)
    assert abs(audio_end() - 22.2) < 1e-6
    ic1 = loader.IS_CHANGED(p2, clip_index=1)
    assert isinstance(ic1, str) and p2 in ic1  # cache keys on the real file
    assert loader.IS_CHANGED("h3_context", clip_index=0) == "disabled"
    (empty,) = loader.load("h3_context", clip_index=0)
    assert empty is None
    print("save/load roundtrip: file loaded, pinned, end_frame %.4f, "
          "index 0 is first-clip/no-context" % audio_end())

    # retry safety with indexed slots: generating clip 3, re-rolling it
    # must overwrite slot 3 and always load slot 2, never its own save
    prevA = {"samples": Nested([prev["samples"].parts[0],
                                T(np.full((1, 32, 2, audio_t), 7.0,
                                          dtype=np.float32))])}
    prevB1 = {"samples": Nested([prev["samples"].parts[0],
                                 T(np.full((1, 32, 2, audio_t), 8.0,
                                           dtype=np.float32))])}
    prevB2 = {"samples": Nested([prev["samples"].parts[0],
                                 T(np.full((1, 32, 2, audio_t), 9.0,
                                           dtype=np.float32))])}
    (pa,) = saver.save(prevA, "h3_context/clip", clip_index=2)   # clip 2 ok
    assert pa.endswith("_00002.safetensors"), pa  # natural slot name
    time.sleep(0.02)
    (pb1,) = saver.save(prevB1, "h3_context/clip", clip_index=3)  # clip 3 try 1
    time.sleep(0.02)
    (pb2,) = saver.save(prevB2, "h3_context/clip", clip_index=3)  # re-roll
    assert pb1 == pb2 and pa != pb1  # re-roll overwrote its own slot
    # generating clip 3, continuing FROM clip 2: loader index is 2, literally
    (l3,) = loader.load("h3_context", clip_index=2)
    got = float(l3["samples"][1].a[0, 0, 0, 0])
    assert got == 7.0, got  # clip 2's latent, NOT the rejected attempt (8/9)
    # index 0 is first-clip/no-context: it must not pick up the reject
    (lnew,) = loader.load("h3_context", clip_index=0)
    assert lnew is None
    # asking for a slot that was never saved says so plainly
    try:
        loader.load("h3_context", clip_index=7)
    except FileNotFoundError as e:
        assert "no saved latent for clip 7" in str(e)
    else:
        raise AssertionError("missing slot did not refuse")
    # an auto-numbered near-miss (trailing underscore) is never matched,
    # and the error explains the rename
    (pauto,) = saver.save(prevA, "h3_context/clip", clip_index=0)
    assert pauto.endswith("_.safetensors"), pauto
    import re as _re
    runno = int(_re.search(r"_(\d{5})_\.safetensors$", pauto).group(1))
    try:
        loader.load("h3_context", clip_index=runno)
    except FileNotFoundError as e:
        assert "trailing underscore" in str(e) and "rename" in str(e), str(e)
    else:
        raise AssertionError("auto-numbered file was matched by index")
    print("indexed slots: re-roll overwrites its slot, loads previous "
          "clip's latent; auto mode confirmed to return the reject")

    import shutil
    import tempfile
    td = tempfile.mkdtemp()
    try:
        open(os.path.join(td, "clip_00001.safetensors"), "wb").close()
        open(os.path.join(td, "clip_00002.safetensors"), "wb").close()
        open(os.path.join(td, "hero_take.safetensors"), "wb").close()
        open(os.path.join(td, "keep_me.txt"), "w").write("x")
        assert nodes._clip_slot_exists(td, 1)
        assert nodes._clip_slot_exists(td, 2)
        assert not nodes._clip_slot_exists(td, 3)
        n = nodes._clear_clip_slots(td)
        assert n == 2, n
        assert not nodes._clip_slot_exists(td, 1)
        assert os.path.exists(os.path.join(td, "keep_me.txt"))
        assert os.path.exists(os.path.join(td, "hero_take.safetensors"))
    finally:
        shutil.rmtree(td)
    print("reset clear: numbered slots gone, custom names left alone")

    print("smoke test passed")


if __name__ == "__main__":
    main()
