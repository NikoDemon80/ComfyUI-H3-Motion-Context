"""H3 Motion Context.

Clip chaining for MiniMax H3: pin the tail of the previous clip, picture
and sound, so the next clip genuinely continues it.

This registers the nodes and nothing else. The two ComfyUI patches

  patch_layout   lifts the first/last-only keyframe anchor restriction,
                 moves pinned audio onto the clip's own timeline, and
                 keeps everything aligned when references shift the layout
  patch_payload  stops the refs branch clobbering keyframe cond latents,
                 so pinned video and pinned audio can be used together

install themselves the first time a Motion Context node runs, not at
import. ComfyUI imports every folder in custom_nodes at startup, and
patching there would put this pack's wrappers in the path of every H3
graph on the machine. Installing on first use means having the pack
installed changes nothing until you actually chain a clip.

Both patches are also gated on this pack's own markers, so even once
installed they leave unrelated H3 graphs bit-identical to stock.

Each patch self-tests against the live ComfyUI code before it commits.
If a test fails the node refuses to run and says why, so an upstream
change produces a clear error rather than a silently wrong render.
"""

import logging

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

logging.getLogger("h3_motion_context").info(
    "h3_motion_context: nodes registered. ComfyUI patches install on the "
    "first run of a Motion Context node.")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
