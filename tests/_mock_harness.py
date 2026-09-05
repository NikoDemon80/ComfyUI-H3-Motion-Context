"""Standalone harness for layout_contract, no ComfyUI and no GPU needed.

Fakes `comfy.ldm.minimax.model` with a PackedLayout that reproduces the
structure of the real one in comfy/ldm/minimax/model.py as of ComfyUI
0.33:

  segment order   text, then keyframe rows, then reference blocks, then
                  target audio, then target video. The target rows are
                  always the last two segments.
  segment kinds   text / cond / cond_audio / ref_img / ref_audio /
                  audio / video
  position_ids    [S, 3] float64, columns (t, h, w)
  origin          the target timeline starts past the span of every
                  reference block, and keyframe anchors are measured from
                  there, so stock compensates anchors for references
  keyframe rows   cond_t = origin + FRAME_RESCALE * resolved_frame_index.
                  A keyframe emits a cond segment only if it carries a
                  video latent, sized by that latent's step count, and a
                  cond_audio segment only if it carries an audio latent,
                  running FORWARD from cond_t for as many steps as the
                  latent has.
  references      laid out from a cursor starting at text_len:
                    image        one ref_img segment, cursor += 1
                    audio        one ref_audio segment of rt*2 rows,
                                 channel-major stereo; cursor += rt, and
                                 the segment is skipped when rt is 0
                    video,       ref_audio then ref_img, both sharing the
                    video_audio  cursor origin; cursor advances by
                                 max(rt, sum of the video spans)

The pack no longer patches any of this. What it needs is that the
keyframe line above stays literal for a fractional, negative index, which
is how the pinned audio window is made to END at the join. No stock node
can produce such an index, so nothing upstream tests it. The mutation
knobs on make_mm are the ways that could plausibly stop being true, and
every one of them must be caught.

Checks:
  1. the contract passes against a faithful 0.33 layout
  2. an integer cast on the anchor index is caught
  3. an audio window anchored at its end rather than its start is caught
  4. a layout that stops compensating anchors for references is caught
  5. a changed audio row count is caught
  6. the older constructor is refused by signature, before anything is
     built, with a message that describes the layout rather than naming a
     ComfyUI version, and points at the pack version that runs on both
  7. the failure is remembered, so a second render does not re-run the
     checks and does not fail silently either
"""

import os
import sys
import types

import numpy as np

# the package dir has to be on sys.path so layout_contract imports by name
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

FRAME_RESCALE = 5.0 / 3.0
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def _video_t_spans(latent_t):
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(latent_t)]


def _frame_grid(h, w):
    """[n, 2] (h, w) coordinates for one latent frame's 2x2-patch rows.

    The values only have to be deterministic and shaped right; nothing
    under test reads these columns.
    """
    n_h, n_w = h // 2, w // 2
    hh, ww = np.meshgrid(np.arange(n_h, dtype=np.float64),
                         np.arange(n_w, dtype=np.float64), indexing="ij")
    return np.stack([hh.reshape(-1), ww.reshape(-1)], axis=-1)


def _audio_grid(cursor, t, rows_per_step=2):
    g = np.zeros((t * rows_per_step, 3), dtype=np.float64)
    g[:, 0] = np.tile(cursor + np.arange(t, dtype=np.float64), rows_per_step)
    if rows_per_step == 2:
        g[:t, 2] = -1.0
        g[t:, 2] = 1.0
    return g


def _video_t_grid(n, origin):
    spans = np.array(_video_t_spans(n), dtype=np.float64)
    head = np.cumsum(spans[:-1]) if n > 1 else np.zeros(0, dtype=np.float64)
    return float(origin) + np.concatenate(([0.0], head))


def _video_grid(vt, frame, cursor):
    g = np.empty((vt, frame.shape[0], 3), dtype=np.float64)
    g[:, :, 0] = _video_t_grid(vt, cursor)[:, None]
    g[:, :, 1:] = frame[None]
    return g.reshape(-1, 3)


def make_mm(audio_rows_per_step=2, int_index=False, audio_anchor="start",
            compensate_refs=True, legacy=False):
    """Build a fake comfy.ldm.minimax.model.

    Defaults reproduce ComfyUI 0.33. Each other argument breaks one
    property the pack depends on:

      int_index          upstream adds a cast, and the fractional audio
                         index silently rounds
      audio_anchor="end" the audio window is measured from its end rather
                         than its start, so it runs the wrong way
      compensate_refs    False puts the 0.32 behaviour back, where anchors
                         are computed from text_len and references slide
                         them relative to the target
      audio_rows_per_step  a change to how many rows one audio step takes
      legacy             the 0.32 constructor, frame_count and all
    """
    mm = types.ModuleType("comfy.ldm.minimax.model")
    mm.FRAME_RESCALE = FRAME_RESCALE
    mm.FRAME_PER_TOKEN = FRAME_PER_TOKEN
    mm._video_t_spans = _video_t_spans

    def _ref_t_span(blk):
        kind = blk["kind"]
        if kind == "image":
            return 1.0
        if kind == "audio":
            return float(blk["ref_audio_t"])
        if kind in ("video", "video_audio"):
            return max(float(blk["ref_audio_t"]),
                       sum(_video_t_spans(blk["latent_t"])))
        raise ValueError("mock: unsupported ref kind %r" % kind)

    def _build(self, text_len, latent_t, latent_h, latent_w, audio_t,
               keyframes, refs, frame_count):
        frame = _frame_grid(latent_h, latent_w)
        frame_rows = frame.shape[0]
        segs, blocks = [], []

        def emit(kind, g):
            segs.append((kind, g.shape[0]))
            blocks.append(g)

        g = np.zeros((text_len, 3), dtype=np.float64)
        g[:, 0] = np.arange(text_len, dtype=np.float64)
        emit("text", g)

        origin = float(text_len)
        if compensate_refs:
            for blk in (refs or []):
                origin += _ref_t_span(blk)

        for kf in (keyframes or []):
            p = kf["resolved_frame_index"]
            if legacy:
                if p == 0:
                    cond_t = float(text_len)
                elif frame_count is not None and p == frame_count - 1:
                    cond_t = (float(text_len) + sum(_video_t_spans(latent_t))
                              - FRAME_RESCALE)
                else:
                    raise ValueError(
                        "only first/last keyframe anchors are supported")
                g = np.empty((frame_rows, 3), dtype=np.float64)
                g[:, 0] = cond_t
                g[:, 1:] = frame
                emit("cond", g)
                continue
            if int_index:
                p = int(p)
            cond_t = origin + FRAME_RESCALE * p
            video_latent = kf.get("latent")
            if video_latent is not None:
                emit("cond", _video_grid(video_latent.shape[2], frame, cond_t))
            audio_latent = kf.get("audio_latent")
            if audio_latent is not None:
                rt = audio_latent.shape[-1]
                start = cond_t if audio_anchor == "start" else cond_t - rt
                emit("cond_audio", _audio_grid(start, rt, audio_rows_per_step))

        cursor = float(text_len)
        for blk in (refs or []):
            kind = blk["kind"]
            if kind == "image":
                r_frame = _frame_grid(blk["latent_h"], blk["latent_w"])
                g = np.empty((r_frame.shape[0], 3), dtype=np.float64)
                g[:, 0] = cursor
                g[:, 1:] = r_frame
                emit("ref_img", g)
                cursor += 1.0
            elif kind == "audio":
                rt = int(blk["ref_audio_t"])
                if rt > 0:
                    emit("ref_audio",
                         _audio_grid(cursor, rt, audio_rows_per_step))
                cursor += float(rt)
            elif kind in ("video", "video_audio"):
                rt = int(blk["ref_audio_t"])
                vt = int(blk["latent_t"])
                r_frame = _frame_grid(blk["latent_h"], blk["latent_w"])
                if rt > 0:
                    emit("ref_audio",
                         _audio_grid(cursor, rt, audio_rows_per_step))
                emit("ref_img", _video_grid(vt, r_frame, cursor))
                cursor += max(float(rt), sum(_video_t_spans(vt)))
            else:
                raise ValueError("mock: unsupported ref kind %r" % kind)

        # target audio then target video, always the last two segments
        emit("audio", _audio_grid(cursor, audio_t))
        emit("video", _video_grid(latent_t, frame, cursor))

        seg_abs, off = [], 0
        for kind, n in segs:
            seg_abs.append((off, off + n, kind))
            off += n
        self.segments = seg_abs
        self.seq_len = off
        self.position_ids = np.concatenate(blocks)

    # Two real signatures, not one with **kwargs: the pack decides whether
    # this ComfyUI is too old by inspecting the signature, and 0.33 has to
    # raise a real TypeError on frame_count.
    if legacy:
        class PackedLayout:
            def __init__(self, text_len, latent_t, latent_h, latent_w,
                         audio_t, keyframes=None, refs=None,
                         frame_count=None):
                _build(self, text_len, latent_t, latent_h, latent_w, audio_t,
                       keyframes, refs, frame_count)
    else:
        class PackedLayout:
            def __init__(self, text_len, latent_t, latent_h, latent_w,
                         audio_t, keyframes=None, refs=None):
                _build(self, text_len, latent_t, latent_h, latent_w, audio_t,
                       keyframes, refs, None)

    mm.PackedLayout = PackedLayout
    return mm


def make_torch():
    t = types.ModuleType("torch")
    t.equal = lambda a, b: a.shape == b.shape and bool(np.array_equal(a, b))

    # nodes.py pads a short audio tail with torch.nn.functional.pad. Only
    # the trailing-last-dim form is used, which is what this covers.
    def _pad(x, pad, mode="constant", value=0.0):
        before, after = int(pad[0]), int(pad[1])
        widths = [(0, 0)] * (x.a.ndim - 1) + [(before, after)]
        return type(x)(np.pad(x.a, widths, mode="constant",
                              constant_values=value))

    functional = types.ModuleType("torch.nn.functional")
    functional.pad = _pad
    nn = types.ModuleType("torch.nn")
    nn.functional = functional
    t.nn = nn
    return t


def load_contract(mm):
    for name in ("comfy", "comfy.ldm", "comfy.ldm.minimax"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["comfy.ldm.minimax.model"] = mm
    sys.modules["comfy"].ldm = sys.modules["comfy.ldm"]
    sys.modules["comfy.ldm"].minimax = sys.modules["comfy.ldm.minimax"]
    sys.modules["comfy.ldm.minimax"].model = mm
    sys.modules["torch"] = make_torch()
    sys.modules.pop("layout_contract", None)
    import layout_contract
    return layout_contract


def _refused(mm, label, expect):
    """The contract must refuse this layout, for the stated reason."""
    lc = load_contract(mm)
    try:
        lc.ensure()
    except RuntimeError as e:
        assert expect in str(e), "%s: wrong reason: %s" % (label, e)
        assert not lc.is_checked()
        return str(e)
    raise AssertionError("%s: the contract accepted it" % label)


def main():
    # 1. faithful 0.33: the checks must pass and stay passed
    mm = make_mm()
    lc = load_contract(mm)
    lc.ensure()
    assert lc.is_checked()
    lc.ensure()  # second call is a no-op, not a second layout build
    print("1. contract passes against a faithful ComfyUI 0.33 layout")

    # 2. an int cast on the anchor index. The video anchors are whole
    # numbers and survive it, so only the audio window moves, which is
    # exactly the silent breakage this check exists for.
    _refused(make_mm(int_index=True), "int cast",
             "fractional or negative resolved_frame_index")
    print("2. an integer cast on the anchor index is caught")

    # 3. the window measured from its end instead of its start: it would
    # run forwards from the join instead of backwards into it
    _refused(make_mm(audio_anchor="end"), "audio anchored at end",
             "pinned audio window")
    print("3. an audio window anchored at its end is caught")

    # 4. references stop compensating the anchors, as they did on 0.32
    _refused(make_mm(compensate_refs=False), "no ref compensation",
             "moved the anchors relative to the target")
    print("4. a layout that stops compensating anchors for references is "
          "caught")

    # 5. a changed audio row count: the window is no longer the shape the
    # placement assumes
    _refused(make_mm(audio_rows_per_step=3), "audio rows",
             "rows for")
    print("5. a changed audio row count is caught")

    # 6. the older constructor: refused by signature, before anything is
    # built. The message must describe the LAYOUT, not guess a ComfyUI
    # version. A wrapper's signature is not ComfyUI's, and the new layout
    # sat on master unreleased for a while, so a version number inferred
    # here would have been wrong in both directions.
    msg = _refused(make_mm(legacy=True), "legacy", "older H3 layout")
    assert "0.3.1" in msg, msg
    assert "0.32" not in msg, "the refusal is guessing a ComfyUI version: %s" % msg
    print("6. the older constructor refused by signature, pointing at pack "
          "0.3.1 without guessing a ComfyUI version")

    # 7. a failure is remembered. The node calls ensure() on every render,
    # and the second call must raise the same way rather than quietly
    # passing because some module state got left behind.
    lc_bad = load_contract(make_mm(int_index=True))
    for attempt in (1, 2):
        try:
            lc_bad.ensure()
        except RuntimeError:
            pass
        else:
            raise AssertionError("attempt %d did not raise" % attempt)
    assert not lc_bad.is_checked()
    print("7. a failed check stays failed on the next render")

    # 8. the case that actually bit: another pack vendoring this project's
    # old layout patch wraps the constructor, and that wrapper's signature
    # carries frame_count on EVERY ComfyUI. Read naively it looks exactly
    # like the older layout, and the user gets told to update a ComfyUI
    # that is already current. Behaviour has to decide, not the signature.
    mm = make_mm()
    stock_init = mm.PackedLayout.__init__

    def _patched_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                      keyframes=None, refs=None, frame_count=None):
        stock_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                   keyframes=keyframes, refs=refs)

    mm.PackedLayout.__init__ = _patched_init
    lc = load_contract(mm)
    lc.ensure()
    assert lc.is_checked(), "a wrapped constructor was mistaken for the old layout"
    print("8. a vendored copy of the old patch wrapping the constructor does "
          "not get mistaken for an older ComfyUI")

    print("all checks passed")


if __name__ == "__main__":
    main()
