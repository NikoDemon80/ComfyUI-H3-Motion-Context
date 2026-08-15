# Changelog

Newest first. Dates are release dates.

ComfyUI's H3 layout changed shape once so far: `PackedLayout.__init__`
dropped its `frame_count` parameter along with the restriction that
rejected any keyframe anchor other than the first or last frame. That
change is on ComfyUI's master branch and has not been tagged. Every
release up to v0.33.1 has the older layout. Each entry below says which
of the two it works with.

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
