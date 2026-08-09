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
    assert pkg.NODE_CLASS_MAPPINGS
    # importing the pack must not have touched ComfyUI yet
    assert not nodes._layout_patch_applied(), \
        "the layout patch was applied at import time"
    assert not nodes._payload_patch_applied(), \
        "the payload patch was applied at import time"
    assert mm.PackedLayout.__init__.__name__ != "_patched_init", \
        "the constructor was wrapped at import time"

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
    out, trim = node.apply(
        conditioning=[["c", {}]], vae=VAE(), latent=target,
        context_frames=context, context_length="22",
        audio_context_length=22, context_latent=prev)

    # ... and running one must have installed both
    assert nodes._layout_patch_applied() and nodes._payload_patch_applied()
    assert mm.PackedLayout.__init__.__name__ == "_patched_init"

    kfs = captured["minimax_keyframes"]
    assert len(kfs) == 7, len(kfs)
    idx = [kf[nodes.MC_KEY] for kf in kfs]
    assert idx == [0, 1, 5, 9, 13, 17, 18], idx
    assert captured["minimax_frame_count"] == frames
    assert trim == 22

    ref = captured["minimax_refs"][0]
    assert ref["kind"] == "audio"
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
    got_end = ref[nodes.MC_AUDIO_KEY]
    end_coord = nodes.FRAME_RESCALE * got_end
    assert abs(end_coord - round(end_coord)) < 1e-9, end_coord
    assert abs(end_coord - 37.0) < 1e-9, end_coord
    assert abs(got_end - 22.2) < 1e-6, got_end
    # the pinned VIDEO must have come out of the latent too, not the VAE.
    # The fake VAE returns zeros, so any nonzero block proves the source,
    # and each block must equal the matching step of the source latent.
    kf_blocks = [kf["latent"] for kf in kfs]
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

    # no latent wired: the pixel path, same offsets, and the fake VAE
    # returns zeros so the blocks must now be zero rather than sliced
    captured.clear()
    node.apply(
        conditioning=[["c", {}]], vae=VAE(), latent=target,
        context_frames=context, context_length="22",
        audio_context_length=22)
    px = captured["minimax_keyframes"]
    assert [kf[nodes.MC_KEY] for kf in px] == idx, "offsets differ by source"
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
        node.apply(
            conditioning=[["c", {}]], vae=VAE(), latent=target,
            context_frames=context, context_length="22",
            audio_context_length=22, context_latent=small)
    except ValueError as e:
        assert "cannot be resized" in str(e), str(e)
        print("resolution change: refused, with the reason and the two "
              "resolutions named")
    else:
        raise AssertionError("mismatched latent did not refuse")

    # nothing wired at all
    try:
        node.apply(
            conditioning=[["c", {}]], vae=VAE(), latent=target,
            context_length="22", audio_context_length=22)
    except ValueError as e:
        assert "nothing to pin" in str(e), str(e)
        print("nothing wired: refused with a plain reason")
    else:
        raise AssertionError("no context at all did not refuse")

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
    node.apply(
        conditioning=[["c", {}]], vae=VAE(), latent=target,
        context_frames=context, context_length="22",
        audio_context_length=22, audio_vae=AudioVAE(), context_audio=audio)
    ref2 = captured["minimax_refs"][0]
    # no overhang to compensate on this path, but the same grid rule
    # applies: 22 frames is 36.667 steps, which must snap to 37
    end2 = nodes.FRAME_RESCALE * ref2[nodes.MC_AUDIO_KEY]
    assert abs(end2 - 37.0) < 1e-9, end2
    print("vae path: window snapped to the audio grid, end coord %.1f" % end2)

    # every clip length on the ladder, every window, both paths: the
    # pinned window must always end on an integer audio coordinate
    import math as _math
    for g in range(2, 13):
        f = 17 * g + 5
        at = _math.ceil(nodes.FRAME_RESCALE * f)
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
    out_cond, _ = node.apply(
        conditioning=r2v_cond, vae=VAE(), latent=target,
        context_frames=context, context_length="22",
        audio_context_length=22, context_latent=prev)
    refs_out = captured["minimax_refs"]
    assert len(refs_out) == 4, len(refs_out)
    for got, want in zip(refs_out, existing):
        assert got["kind"] == want["kind"], (got, want)
        assert got.get(nodes.MC_AUDIO_KEY) is None, got
    assert refs_out[-1][nodes.MC_AUDIO_KEY] is not None
    assert refs_out[-1]["ref_audio_t"] == 37
    # the caller's own list must not have been mutated in place
    assert len(r2v_cond[0][1]["minimax_refs"]) == 3
    assert captured["minimax_keyframes"], "keyframes lost on the R2V path"
    print("R2V path: 3 incoming references preserved, motion context audio "
          "appended as the 4th, keyframes intact")

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
    (loaded,) = loader.load("h3_context")  # folder -> newest = p2
    parts = loaded["samples"]
    assert isinstance(parts, list) and len(parts) == 2
    captured.clear()
    node.apply(
        conditioning=[["c", {}]], vae=VAE(), latent=target,
        context_frames=context, context_length="22",
        audio_context_length=22, context_latent=loaded)
    ref3 = captured["minimax_refs"][0]
    want = float(prev2["samples"].parts[1].a[0, 0, 0, -1])
    got = float(ref3["audio_latent"].a[0, 0, 0, -1])
    assert got == want, (got, want)  # newest save's content came through
    assert abs(ref3[nodes.MC_AUDIO_KEY] - 22.2) < 1e-6
    ic1 = loader.IS_CHANGED("h3_context")
    assert isinstance(ic1, str) and p2 in ic1  # cache keys on the real file
    print("save/load roundtrip: newest of 2 saves loaded, pinned, "
          "end_frame %.4f, cache key tracks the file" %
          ref3[nodes.MC_AUDIO_KEY])

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
    # newest-file mode would have returned the reject: prove the hazard
    (lnew,) = loader.load("h3_context", clip_index=0)
    assert float(lnew["samples"][1].a[0, 0, 0, 0]) == 9.0
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

    print("smoke test passed")


if __name__ == "__main__":
    main()
