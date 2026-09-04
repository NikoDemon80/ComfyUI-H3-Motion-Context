# Changelog

Newest first. Dates are release dates.

ComfyUI's H3 layout changed shape once so far: `PackedLayout.__init__`
dropped its `frame_count` parameter along with the restriction that
rejected any keyframe anchor other than the first or last frame. That
landed in ComfyUI 0.34.0. Every release through 0.33.4 has the older
layout. Each entry below says which of the two it works with.

## 0.6.0 - 2026-09-03

Requires ComfyUI 0.34.0 or newer. Use 0.3.1 on anything older.

- Chain is Approve on a loop: it advances Load/Save, then queues, then
  keeps going. At Load 0 / Save 1 with no clip 1 on disk it generates
  that first clip instead of walking into a missing file. `segments` is
  how many clips that loop runs (0 = until Stop). Reset sets 0/1 and
  does not delete files. Clear latents deletes numbered chain slots;
  custom filenames are left alone.

## 0.5.1 - 2026-09-02

Requires ComfyUI 0.34.0 or newer. Use 0.3.1 on anything older.

- Example workflow updated.

## 0.5.0 - 2026-09-02

Requires ComfyUI 0.34.0 or newer. Use 0.3.1 on anything older.

#25 by feigo313 is why Load 0 is first-clip/no-context, so Motion Context
stays enabled from the first clip, and why the Chain node exists: walking
those indices with run-on-change queues twice and skips slots.

- Load Latent `clip_index` 0 is first-clip/no-context: nothing is read,
  Motion Context passes the original conditioning through, and
  `trim_frames` is 0. Leave the node enabled. The chain is Load 0 / Save 1,
  then Load 1 / Save 2. Positive indices still load that exact slot.
  The old load-newest meaning of index 0 is gone.
- H3 Motion Context Chain node: Approve advances Load/Save then queues
  once, Run/Re-roll repeats the current slot (use it instead of
  ComfyUI's Run), Chain auto-approves from the current indices, Reset
  sets 0/1 without queuing. Load, Save, and Chain must share one canvas
  group or the buttons do nothing. Button clicks do not open the node
  menu. Do not use queue "run on change" to walk indices.
- Save Latent can overwrite a slot that Load still has memory-mapped on
  Windows (os error 1224). Load copies tensors off the file first, and
  Save writes through a temp file.

## 0.4.0 - 2026-08-26

Requires ComfyUI 0.34.0 or newer. Use 0.3.1 on anything older.

ComfyUI now places keyframe anchors at any frame itself, so the two
runtime patches this pack carried are gone. Nothing in ComfyUI is
modified any more.

- `patch_layout.py` and `patch_payload.py` removed. The node builds plain
  keyframe dicts and hands them to stock code.
- `layout_contract.py` added. It proves, once before the first render,
  that anchors sit where the arithmetic says and that the pinned audio
  window is placed literally from a fractional, negative anchor index.
  No stock node can produce such an index, so nothing upstream tests it,
  and an integer cast added later would move the pinned sound silently.
  If a check fails the node refuses and names what moved.
- The pinned audio is a keyframe rather than a reference block. A Ref2VA
  graph's reference list is left untouched.
- Add Guide for MiniMax H3 anchors survive alongside a chained head,
  including ones carrying their own audio. Guides landing inside the
  pinned head are dropped with a warning, as `first_frame` anchors are.
- Other packs may patch the layout without this one standing down, since
  it no longer competes for that code. If it finds the constructor
  wrapped it says who by and checks the behaviour anyway.
- Anchor and audio window coordinates are identical to 0.3.1, verified
  against the real upstream layout on both shapes.

## 0.3.1 - 2026-08-14

Works with both H3 layouts.

Fixed the crash on the newer layout, reported in #12 by javawock7618 and
#8 by azra1l. The patch had passed `frame_count` unconditionally, so its
self-test raised `TypeError` and the node refused to run.

- The patch reads the layout constructor's signature and adapts, rather
  than assuming either shape. Verified against the real upstream file on
  both: identical anchor and audio window coordinates to the last bit.
- Keyframe audio latents are no longer dropped from the payload. The
  newer layout lets a keyframe carry audio of its own, and rebuilding the
  list from references alone filled every audio conditioning row with the
  wrong content.
- Pinned anchors pair with the correct rows when a stock anchor that
  carries audio but no picture shares the graph.
- The mixed keyframe guard is retired on the newer layout, where stock
  compensates untagged keyframes for reference blocks itself. A Ref2VA
  graph carrying a stock anchor is now supported rather than refused.
- The test harness fakes both layout shapes.

## 0.3.0 - 2026-08-11

Works with the older H3 layout only. Refuses to run on the newer one.

- **Seam Probe node.** Measures join quality inline in the graph: lag,
  correlation, RMS step and floor level step across the join. Passes
  clip B through unchanged so it can sit inline without rewiring.
- **last_frame passthrough.** An fl2va graph's own anchors used to be
  replaced outright. A last-frame anchor now survives a chained head,
  tagged so it gets the same reference compensation as the pinned run.
  Anchors falling inside the pinned head are dropped with a warning,
  since the pinned run already decides those frames.
- Conditioning carrying keyframes resolved against a different clip
  length is refused rather than rendered at the wrong frame.

## 0.2.0 - 2026-08-09

Works with the older H3 layout only. First release published to the
ComfyUI registry.

- **Reference mode support.** `patch_layout.py` was rewritten to locate
  reference blocks by segment table rather than by recomputing cursor
  arithmetic, which made multiple reference blocks work cleanly. Ref2VA
  graphs can carry their own image, video and audio references alongside
  a chained head.
- **Latent picture path.** The pinned run is sliced straight out of the
  previous clip's latent, skipping the decode and re-encode round trip
  that dulls the picture a little more at every link.
- **Patches install on first use, not at import.** Having the pack in
  `custom_nodes` now changes nothing until a Motion Context node
  actually runs.
- **Coexistence detection.** A second copy of the patch, vendored into
  another pack, recognises the first and stands down instead of
  wrapping it.
- The payload patch is gated on this pack's own markers, so unrelated H3
  graphs stay bit-identical to stock.
- Two settings exposed; everything with only one correct answer was
  removed from the UI.
- `freeze_detect.py` and `level_step.py` added to the measurement
  scripts.
- The example workflow was replaced with an fl2va / ref2va one.

## Initial release - 2026-08-06

Works with the older H3 layout only. Not published to the registry; no
version number, as `pyproject.toml` came later.

Clip chaining for MiniMax H3 with audio continuation rather than
imitation. Four nodes: Motion Context, Trim, Save Latent, Load Latent.
The pinned audio window is placed so it ends at the join and reaches
backwards, which is what makes the model continue a soundtrack instead
of starting something that merely resembles it.
