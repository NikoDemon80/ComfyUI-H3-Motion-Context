"""Standalone harness for patch_layout, no ComfyUI and no GPU needed.

Fakes `comfy.ldm.minimax.model` with a PackedLayout that reproduces the
structure of the real one in comfy/ldm/minimax/model.py:

  segment order   text, then keyframe cond rows, then reference blocks,
                  then target audio, then target video. Reference blocks
                  come AFTER the keyframe rows, and the target rows are
                  always the last two segments.
  segment kinds   text / cond / ref_img / ref_audio / audio / video
  position_ids    [S, 3] float64, columns (t, h, w)
  cond rows       one segment per keyframe, at STOCK coordinates computed
                  from text_len and NOT from the reference cursor, which
                  is the uncompensated behaviour the patch exists to fix,
                  and rejecting interior anchors
  references      laid out from a cursor starting at text_len:
                    image        one ref_img segment of (h//2)*(w//2)
                                 rows, all on the cursor; cursor += 1
                    audio        one ref_audio segment of rt*2 rows,
                                 channel-major stereo, coordinates
                                 cursor..cursor+rt-1 once per channel;
                                 cursor += rt. Skipped entirely when rt
                                 is 0, which still advances.
                    video,       ref_audio then ref_img, both sharing the
                    video_audio  cursor origin; cursor advances by
                                 max(rt, sum of the video spans)
  target          audio rows from the final cursor, then video rows whose
                  first row sits exactly on it

An earlier version of this file put references before the cond rows, used
four position columns and labelled every reference block `ref`. All three
were wrong, and stopped mattering only because nothing read the segment
table. Anything that does read it needs this to be faithful.

Checks:
  1. apply_patch succeeds against the faithful layout
  2. interior anchors build, and reference compensation lands
  3. the marked audio block moves onto the target timeline while a
     Ref2VA graph's own references stay put
  4. the mixed stock/MC keyframe guard trips
  5. a layout emitting the wrong number of rows for an audio reference is
     caught rather than silently half-moved
  6. a layout whose reference cursor advances differently still produces
     correctly aligned output, because nothing here recomputes it
  7. a second copy of the patch, as another pack vendoring the same file
     would load, stands down instead of wrapping the first copy
  8. a DIFFERENT copy (a fork, or a version predating the marker) is also
     recognised and stood down for
  9. a wrapper from an unrelated pack solving the same problem its own
     way is recognised too, and refused rather than stacked on. Several
     H3 packs lift the same restriction independently and they cannot
     both own the constructor.
"""

import importlib
import sys
import types

import numpy as np

FRAME_RESCALE = 5.0 / 3.0
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def _video_t_spans(latent_t):
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(latent_t)]


def _frame_grid(h, w):
    """[n, 2] (h, w) coordinates for one latent frame's 2x2-patch rows.

    The values only have to be deterministic and shaped right; the patch
    never writes these columns, it only has to leave them alone.
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


def make_mm(ref_advance_factor=1.0, audio_rows_per_step=2):
    """Build a fake comfy.ldm.minimax.model.

    ref_advance_factor != 1 simulates an upstream change to how an audio
    reference advances the cursor. audio_rows_per_step != 2 simulates a
    change to how many rows one audio latent step occupies.
    """
    mm = types.ModuleType("comfy.ldm.minimax.model")
    mm.FRAME_RESCALE = FRAME_RESCALE
    mm.FRAME_PER_TOKEN = FRAME_PER_TOKEN
    mm._video_t_spans = _video_t_spans

    class PackedLayout:
        def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t,
                     keyframes=None, refs=None, frame_count=None):
            frame = _frame_grid(latent_h, latent_w)
            frame_rows = frame.shape[0]
            segs, blocks = [], []

            def emit(kind, g):
                segs.append((kind, g.shape[0]))
                blocks.append(g)

            g = np.zeros((text_len, 3), dtype=np.float64)
            g[:, 0] = np.arange(text_len, dtype=np.float64)
            emit("text", g)

            spans = _video_t_spans(latent_t)
            for kf in (keyframes or []):
                p = kf["resolved_frame_index"]
                # faithful stock: computed from text_len, NOT the cursor
                if p == 0:
                    t = float(text_len)
                elif frame_count is not None and p == frame_count - 1:
                    t = float(text_len) + sum(spans) - FRAME_RESCALE
                else:
                    raise ValueError(
                        "only first/last keyframe anchors are supported")
                g = np.empty((frame_rows, 3), dtype=np.float64)
                g[:, 0] = t
                g[:, 1:] = frame
                emit("cond", g)

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
                    cursor += float(rt) * ref_advance_factor
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

    mm.PackedLayout = PackedLayout
    return mm


def make_torch():
    t = types.ModuleType("torch")
    t.equal = lambda a, b: a.shape == b.shape and bool(np.array_equal(a, b))
    return t


def load_patch(mm):
    for name in ("comfy", "comfy.ldm", "comfy.ldm.minimax"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["comfy.ldm.minimax.model"] = mm
    sys.modules["comfy"].ldm = sys.modules["comfy.ldm"]
    sys.modules["comfy.ldm"].minimax = sys.modules["comfy.ldm.minimax"]
    sys.modules["comfy.ldm.minimax"].model = mm
    sys.modules["torch"] = make_torch()
    sys.modules.pop("patch_layout", None)
    return importlib.import_module("patch_layout")


TEXT_LEN, LATENT_T, LH, LW, AUDIO_T = 7, 7, 22, 38, 16
FC = sum(FRAME_PER_TOKEN[k % 5] for k in range(LATENT_T))


def _cond_ts(layout):
    return [float(layout.position_ids[a, 0])
            for a, _, k in layout.segments if k == "cond"]


def main():
    # 1. faithful stock: patch must apply
    mm = make_mm()
    pl = load_patch(mm)
    assert pl.apply_patch(), "self-test failed against faithful stock"
    assert pl.is_applied()
    print("1. self-test passes against the faithful stock layout")

    # 2. interior anchors build, reference compensation lands
    run = [{"resolved_frame_index": 0, pl.MC_KEY: i} for i in range(4)]
    lay = mm.PackedLayout(TEXT_LEN, LATENT_T, LH, LW, AUDIO_T,
                          keyframes=run, frame_count=FC)
    ts = _cond_ts(lay)
    exp = [TEXT_LEN + FRAME_RESCALE * i for i in range(4)]
    assert np.allclose(ts, exp), (ts, exp)
    ref = [{"kind": "audio", "ref_audio_t": 8}]
    lay2 = mm.PackedLayout(TEXT_LEN, LATENT_T, LH, LW, AUDIO_T,
                           keyframes=run, refs=ref, frame_count=FC)
    ts2 = _cond_ts(lay2)
    assert np.allclose(ts2, [t + 8.0 for t in ts]), (ts, ts2)
    print("2. interior run at", [round(t, 4) for t in ts],
          "-> with an 8-step reference", [round(t, 4) for t in ts2])

    # 3. Ref2VA: the graph's own references survive the audio move intact.
    # The marked block sits third of four, not last, because locating it
    # by segment does not depend on where in the list it is.
    rt, end_frame = 8, 22
    head = [
        {"kind": "image", "latent_h": 8, "latent_w": 12},
        {"kind": "video_audio", "latent_h": 8, "latent_w": 12,
         "latent_t": 3, "ref_audio_t": 5},
    ]
    tail = [{"kind": "audio", "ref_audio_t": 3}]
    marked = {"kind": "audio", "ref_audio_t": rt, pl.MC_AUDIO_KEY: end_frame}
    plain = {"kind": "audio", "ref_audio_t": rt}
    refs_plain = head + [plain] + tail
    refs_marked = head + [marked] + tail

    base = mm.PackedLayout(TEXT_LEN, LATENT_T, LH, LW, AUDIO_T,
                           keyframes=run, refs=refs_plain, frame_count=FC)
    after = mm.PackedLayout(TEXT_LEN, LATENT_T, LH, LW, AUDIO_T,
                            keyframes=run, refs=refs_marked, frame_count=FC)
    a, b = pl._ref_segment_map(base, refs_plain)[2]["ref_audio"]
    diff = set(i for i in range(base.position_ids.shape[0])
               if float(base.position_ids[i, 0])
               != float(after.position_ids[i, 0]))
    assert diff == set(range(a, b)), (len(diff), b - a)
    origin = pl._target_origin(after)
    got_end = float(after.position_ids[a, 0]) + rt
    assert abs(got_end - (origin + FRAME_RESCALE * end_frame)) < 1e-9
    ts3 = _cond_ts(after)
    assert np.allclose(
        ts3, [origin + FRAME_RESCALE * i for i in range(4)]), ts3
    print("3. marked block (3rd of 4 references) ends at target frame %d; "
          "the image, video and audio references either side of it did not "
          "move" % end_frame)

    # 4. mixed stock/MC keyframes under a reference are rejected
    mixed = [{"resolved_frame_index": 0},
             {"resolved_frame_index": 0, pl.MC_KEY: 2}]
    try:
        mm.PackedLayout(TEXT_LEN, LATENT_T, LH, LW, AUDIO_T,
                        keyframes=mixed, refs=ref, frame_count=FC)
    except RuntimeError as e:
        assert "mixed" in str(e)
        print("4. mixed stock/MC keyframes under a reference rejected loudly")
    else:
        raise AssertionError("mixed keyframes under a reference not rejected")

    # 5. an audio block with an unexpected row count must be caught, not
    # half-moved. The count is asserted exactly, so 3 rows per step trips.
    bad_rows = make_mm(audio_rows_per_step=3)
    pl_bad = load_patch(bad_rows)
    assert not pl_bad.apply_patch(), \
        "self-test passed against a changed audio row count"
    assert bad_rows.PackedLayout.__init__.__name__ != "_patched_init"
    print("5. changed audio row count caught, patch refused")

    # 6. a changed reference cursor advance must NOT break anything. The
    # target origin is read back from the layout rather than recomputed,
    # so the anchors and the audio window follow it wherever it goes. The
    # previous implementation kept its own copy of that arithmetic and
    # had no choice but to refuse.
    drift = make_mm(ref_advance_factor=2.0)
    pl_drift = load_patch(drift)
    assert pl_drift.apply_patch(), \
        "a changed reference advance should no longer defeat the patch"
    only = [dict(marked)]
    lay_d = drift.PackedLayout(TEXT_LEN, LATENT_T, LH, LW, AUDIO_T,
                               keyframes=run, refs=only, frame_count=FC)
    origin_d = pl_drift._target_origin(lay_d)
    ts_d = _cond_ts(lay_d)
    assert np.allclose(
        ts_d, [origin_d + FRAME_RESCALE * i for i in range(4)]), ts_d
    a_d, _ = pl_drift._ref_segment_map(lay_d, only)[0]["ref_audio"]
    end_d = float(lay_d.position_ids[a_d, 0]) + rt
    assert abs(end_d - (origin_d + FRAME_RESCALE * end_frame)) < 1e-9
    print("6. doubled reference advance: anchors and audio window still "
          "land correctly, because the origin is read and not recomputed")

    # 7. two packs vendoring this file: the second must stand down, and
    # the first must still own the constructor and still work.
    mm7 = make_mm()
    first = load_patch(mm7)
    assert first.apply_patch()
    installed = mm7.PackedLayout.__init__

    sys.modules.pop("patch_layout", None)
    second = importlib.import_module("patch_layout")
    assert second is not first, "second load returned the same module object"
    assert second.apply_patch(), \
        "standing down should report success, the patch IS active"
    assert second.is_applied(), \
        "is_applied must be true when another pack owns the patch, or the "\
        "second pack's nodes would refuse to run"
    assert mm7.PackedLayout.__init__ is installed, \
        "the second copy wrapped the first instead of standing down"
    assert second._orig_init is None, "the second copy captured an original"

    lay7 = mm7.PackedLayout(TEXT_LEN, LATENT_T, LH, LW, AUDIO_T,
                            keyframes=run, refs=[dict(marked)], frame_count=FC)
    origin7 = first._target_origin(lay7)
    ts7 = _cond_ts(lay7)
    assert np.allclose(
        ts7, [origin7 + FRAME_RESCALE * i for i in range(4)]), ts7
    a7, _ = first._ref_segment_map(lay7, [marked])[0]["ref_audio"]
    end7 = float(lay7.position_ids[a7, 0]) + rt
    assert abs(end7 - (origin7 + FRAME_RESCALE * end_frame)) < 1e-9
    print("7. second copy of the patch stands down, first still owns the "
          "constructor and still places anchors and audio correctly")

    # 8. a wrapper from a different copy carries no marker, only the
    # name. It must still be recognised, or the second copy self-tests
    # through the first one's behaviour and refuses over a limitation
    # that may not exist in its own code.
    mm8 = make_mm()
    other = load_patch(mm8)
    assert other.apply_patch()
    wrapper = mm8.PackedLayout.__init__
    try:
        delattr(wrapper, other.PATCH_MARKER)     # simulate an older copy
    except AttributeError:
        raise AssertionError("the wrapper was never marked")
    assert not getattr(wrapper, other.PATCH_MARKER, False)
    assert wrapper.__name__ == "_patched_init"

    sys.modules.pop("patch_layout", None)
    mine = importlib.import_module("patch_layout")
    assert mine._already_patched() == "other", mine._already_patched()
    assert mine.apply_patch(), "an unmarked copy must still be stood down for"
    assert mine.is_applied()
    assert mm8.PackedLayout.__init__ is wrapper, "wrapped a foreign copy"
    assert mine._orig_init is None
    print("8. an unmarked copy of the patch (older version or fork) is "
          "recognised by name and stood down for, not wrapped")

    # 9. an unrelated pack's wrapper: no marker, a different name, and
    # defined somewhere other than the module PackedLayout lives in.
    mm9 = make_mm()
    stock_init = mm9.PackedLayout.__init__
    assert stock_init.__module__ == mm9.PackedLayout.__module__

    def _some_other_pack_init(self, *args, **kwargs):
        stock_init(self, *args, **kwargs)
    _some_other_pack_init.__module__ = "some_other_h3_pack.patches"
    mm9.PackedLayout.__init__ = _some_other_pack_init

    ours = load_patch(mm9)
    assert ours._already_patched() == "foreign", ours._already_patched()
    assert not ours.apply_patch(), \
        "stacked on an unrelated pack's wrapper instead of refusing"
    assert not ours.is_applied()
    assert mm9.PackedLayout.__init__ is _some_other_pack_init, \
        "replaced another pack's wrapper"

    # functools.wraps copies __module__ and __name__ across, so the
    # module check alone would miss it; __wrapped__ is the giveaway
    import functools
    mm10 = make_mm()
    orig10 = mm10.PackedLayout.__init__

    @functools.wraps(orig10)
    def _wrapped_init(self, *args, **kwargs):
        orig10(self, *args, **kwargs)
    mm10.PackedLayout.__init__ = _wrapped_init
    assert _wrapped_init.__module__ == mm10.PackedLayout.__module__
    ours2 = load_patch(mm10)
    assert ours2._already_patched() == "foreign", ours2._already_patched()
    assert not ours2.apply_patch()
    print("9. an unrelated pack's wrapper is detected and refused, whether "
          "it hides behind functools.wraps or not")

    print("all checks passed")


if __name__ == "__main__":
    main()
