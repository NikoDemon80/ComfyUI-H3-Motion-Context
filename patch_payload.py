"""Let keyframes and refs coexist.

`MiniMaxH3.extra_conds` in comfy/model_base.py fills the payload from two
independent `if` blocks. The keyframe block sets `cond_video_latents`, then
the refs block **overwrites** it:

    if keyframes is not None:
        payload["cond_video_latents"] = [kf["latent"] for kf in keyframes]
    if refs is not None:
        payload["cond_video_latents"] = [r["latent"] for r in refs if "latent" in r]
        payload["cond_audio_latents"] = [r["audio_latent"] for r in refs ...]

So attaching an audio-only ref alongside keyframes wipes the keyframe video
content: an audio-only block has no "latent" key, the list comes back empty,
and the cond rows the layout built have nothing to fill them.

The layout itself handles the combination fine. Keyframe cond rows are
emitted first, ref rows second, target rows last, which is exactly the order
the forward pass expects when it writes rows into the never-denoised slots.
Only this payload assignment is in the way.

This wrapper re-runs the same logic and concatenates instead, keeping
keyframe latents first to match the row order.

It only does that for graphs that asked for it. The wrapper is installed
process-wide, but it returns stock output unless a keyframe or a ref
carries one of this pack's markers. Graphs that combine stock keyframes
and refs without going anywhere near Motion Context therefore behave
exactly as they did before the pack was installed. That matters because
such graphs exist: a Ref2VA workflow with a last_frame anchor is a
legitimate combination of both mechanisms and has nothing to do with
chaining. Whether stock's overwrite is a bug there is not this pack's
call to make.

Graphs using only one mechanism are unaffected either way: with no refs
the ref list is empty, with no keyframes the keyframe list is.

The order is what makes this safe under Ref2VA, where the ref list holds
the graph's own image, video and audio blocks as well as ours. Stock
builds the layout as text, then cond rows, then reference blocks in list
order, so keyframe latents first and reference latents in list order is
exactly the order the rows are waiting in. Audio is filled from the same
list in the same order, and an audio-only block contributes nothing to
the video list, so the two stay in step however the graph is wired.
"""

import logging

import comfy.model_base as model_base

_LOG = logging.getLogger("h3_motion_context")

# Duplicated from patch_layout rather than imported, so this module stays
# independently testable against a fake model_base. The marker strings are
# a shared ABI across the two patches, the node, and any pack vendoring
# them: rename in lockstep or not at all.
MC_KEY = "motion_context_index"
MC_AUDIO_KEY = "motion_context_audio_end_frame"

# Marker set on our wrapper so a second copy of this file, vendored into
# another pack, can recognise it and stand down instead of wrapping it.
# Shared ABI across every pack that vendors this patch.
PATCH_MARKER = "_h3_motion_context_payload_patch"

_orig_extra_conds = None
_applied = False


def _patched_extra_conds(self, **kwargs):
    out = _orig_extra_conds(self, **kwargs)

    keyframes = kwargs.get("minimax_keyframes", None)
    refs = kwargs.get("minimax_refs", None)
    if not keyframes or not refs:
        return out  # only one mechanism in play, stock behaviour is correct
    if not (any(MC_KEY in kf for kf in keyframes)
            or any(MC_AUDIO_KEY in r for r in refs)):
        # nothing here came from this pack. The layout patch is gated the
        # same way, so leaving the payload alone keeps the two consistent
        # and leaves unrelated graphs bit-identical to stock.
        return out

    cond = out.get("minimax_payload", None)
    payload = getattr(cond, "cond", None) if cond is not None else None
    if not isinstance(payload, dict):
        _LOG.warning("h3_motion_context: could not reach the H3 payload, "
                     "keyframe latents may have been overwritten by refs")
        return out

    kf_video = [kf["latent"] for kf in keyframes if "latent" in kf]
    ref_video = [r["latent"] for r in refs if "latent" in r]
    payload["cond_video_latents"] = kf_video + ref_video
    payload["cond_audio_latents"] = [r["audio_latent"] for r in refs
                                     if r.get("audio_latent") is not None]
    # only write frame_count when we actually have one. This wrapper fires
    # for ANY graph combining keyframes and refs, not just ours; a graph
    # that reaches here without minimax_frame_count may have a valid value
    # already set by the original, and overwriting it with None would break
    # the last-frame anchor branch downstream.
    fc = kwargs.get("minimax_frame_count", None)
    if fc is not None:
        payload["frame_count"] = fc
    return out


setattr(_patched_extra_conds, PATCH_MARKER, True)


def _already_patched(cls):
    """Has another copy of this file already wrapped extra_conds?

    Returns None, "same", "other" or "foreign". The marker only
    recognises copies new enough to set it, so a wrapper merely NAMED
    like ours counts as another copy: an older version, or a fork, and
    the second one in stands down. Anything else wrapping extra_conds is
    a different pack solving the same problem its own way, and this one
    refuses rather than stacking on it. See patch_layout's version for
    how the detection works and what it cannot see.
    """
    fn = getattr(cls, "extra_conds", None)
    if fn is None:
        return None
    if getattr(fn, PATCH_MARKER, False):
        return "same"
    if getattr(fn, "__name__", "") == "_patched_extra_conds":
        return "other"
    if hasattr(fn, "__wrapped__"):
        return "foreign"
    home = getattr(cls, "__module__", None)
    where = getattr(fn, "__module__", None)
    if home and where and where != home:
        return "foreign"
    return None


def apply_patch():
    global _orig_extra_conds, _applied
    if _applied:
        return True
    cls = getattr(model_base, "MiniMaxH3", None)
    if cls is None or not hasattr(cls, "extra_conds"):
        _LOG.warning("h3_motion_context: MiniMaxH3.extra_conds not found, "
                     "keyframes and refs cannot be combined")
        return False
    who = _already_patched(cls)
    if who == "foreign":
        _LOG.warning(
            "h3_motion_context: another pack has already patched "
            "MiniMaxH3.extra_conds (it now comes from %r). Both packs are "
            "solving the same keyframe/ref collision and they cannot both "
            "own it, so this one is refusing. Disable one of them and "
            "restart.",
            getattr(getattr(cls, "extra_conds", None), "__module__", "?"))
        return False
    if who:
        # report success: the patch IS active, just not ours, and the
        # calling pack's nodes check is_applied() before they will run
        _applied = True
        if who == "same":
            _LOG.info("h3_motion_context: keyframe/ref coexistence already "
                      "enabled by another pack, standing down")
        else:
            _LOG.warning(
                "h3_motion_context: the H3 payload patch is already "
                "installed by a DIFFERENT copy of this code (another "
                "version, or a fork). Standing down. If you have more than "
                "one H3 Motion Context folder in custom_nodes, keep one and "
                "remove the rest.")
        return True
    _orig_extra_conds = cls.extra_conds
    cls.extra_conds = _patched_extra_conds
    _applied = True
    _LOG.info("h3_motion_context: keyframe/ref coexistence enabled")
    return True


def is_applied():
    return _applied
