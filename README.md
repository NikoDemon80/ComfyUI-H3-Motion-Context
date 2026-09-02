# H3 Motion Context

Chain MiniMax H3 clips so motion and sound keep going across the cut.

Generate clip A. Feed its last frames and audio into this node. Generate
clip B. B picks up where A left off: same motion, same speed, same
direction, and the same audio continued rather than a new take that
sounds similar. Repeat as long as you like.

Nothing on disk is edited and nothing in ComfyUI is modified. The nodes
check their arithmetic against the live ComfyUI code before the first
render, and if an update breaks an assumption they refuse to run and say
what moved. A loud failure beats a bad render you don't notice.

Requires ComfyUI 0.34.0 or newer, which is where H3 gained arbitrary
keyframe anchors. Version 0.3.1 runs on that and on everything older;
see [CHANGELOG.md](CHANGELOG.md).

## Why this exists

ComfyUI can anchor a still or a short clip at any frame of an H3
generation, with the Add Guide for MiniMax H3 node. If that is all you
need, use it. You do not need this pack for it.

Chaining is a different problem, and two parts of it are still unsolved
by anything else.

**The picture goes out to pixels and back at every link.** Add Guide takes
images and encodes them. Chain through it and every join costs a decode, a
resize and a re-encode. That round trip is where the colour drift and the
softening down a long chain come from. This pack slices the previous
clip's tail straight out of its latent, so the pinned frames are the same
numbers they were, bit for bit.

**The sound restarts instead of continuing.** Add Guide anchors audio
starting at a frame and running forward. To actually continue a
soundtrack, the pinned window has to END at the join and reach backwards
into the sound that already played. That is the difference between the
model continuing your track and the model writing something that sounds
like it, and on anything with a beat you hear it immediately.

Everything else here follows from those two: the Trim node that removes
the pinned head from the delivery, the Save/Load Latent pair that carries
a clip across without touching pixels, and the Seam Probe that measures
whether a join is a real continuation or a convincing imitation.

Audio was the harder half, and it's the more useful half, since H3
generates picture and sound together. See "Why the audio needed work"
below if you care how.

## Install

Drop the folder in `ComfyUI/custom_nodes/` and restart. At startup you'll
just see:

```
h3_motion_context: nodes registered. ComfyUI is not modified; the layout
checks run on the first use of a Motion Context node.
```

Having the pack installed changes nothing about your other H3 workflows,
and nothing runs until you actually chain a clip. The first time you do,
you'll see:

```
h3_motion_context: ComfyUI H3 layout checks passed, anchors and pinned audio will land where intended
```

Anything else and the node refuses to run. The reason is logged.

On ComfyUI 0.33.4 or older the node says so and points you at version
0.3.1, which runs on both. That is read from the layout code itself
rather than a version number, so a nightly or a fork gets the right
answer.

## Wiring

```
MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo (or the t2v path)
  -> H3 Motion Context      <- previous clip's latent (picture + sound)
  -> guider / sampler
  ...
  decoded IMAGE + AUDIO
  -> H3 Motion Context Trim         <- wire trim_frames across
  -> Create Video / save
```

Wire `trim_frames` into the Trim node. The pinned frames come back at the
start of the new clip and have to come off before you concatenate, picture
and sound together.

### Carrying the previous clip across

The previous clip reaches the node as a latent, but you can't wire the sampler
straight into `context_latent`. ComfyUI will call it a circular
connection, and it's right: the latent you want is from the previous run,
not this one. Two helper nodes move it across runs the same way your
frames and audio already move across, through a file:

```
this run:   SamplerCustomAdvanced -> H3 Motion Context Save Latent
next run:   H3 Motion Context Load Latent -> context_latent
```

Both have a `clip_index` and the numbers mean what they say. On Load, the
clip you're continuing FROM. On Save, the clip this one IS. First clip:
Load 0, Save 1. Making clip 2 from clip 1: Load 1, Save 2.

Load 0 does not read a file. Motion Context then passes the incoming
conditioning through and reports `trim_frames` 0, so you can leave the
node enabled for the first clip. Do not mute it.

Do not use queue **run on change** to walk the chain. Each index change is
a separate graph edit, so two widgets incrementing will queue two runs and
skip slots. Use **H3 Motion Context Chain** instead of ComfyUI's Run
button.

**The Chain node only sees Load and Save if all three sit in the same
canvas group** (one of those coloured boxes you draw around nodes). If
they are not grouped together, every button does nothing. Select Load,
Save, and Chain, then Group Selected (or drop Chain into the existing
Motion Context group).

- **Run/Re-roll** is the Run button for this graph. It queues at the
  current Load/Save indices and does not advance them. First clip: leave
  Load 0 / Save 1 and click it. Hate the result? Click it again; the
  retry overwrites that slot.
- **Approve** advances both indices, then queues the next clip once.
- **Chain** queues at the current Load/Save indices and auto-approves
  after each successful run. It does not reset the pair first, so you
  can walk a few clips by hand and then let it take over. Click **Stop**
  (the same button) to halt the next queue.
- **Reset** sets Load 0 / Save 1. It does not queue.

Files are named the obvious way, `clip_00002.safetensors` is clip 2.

Auto-saved files (Save at 0) get a trailing underscore
(`clip_00002_.safetensors`) because they're numbered by run, not by clip,
and indexed loading skips them on purpose.

You can also point the loader at a specific file when `clip_index` is
greater than 0; the file is loaded and the index is ignored. Index 0
never reads a file.
Its output is only for `context_latent`. Don't wire it into a decode node.
Stock Save/Load Latent won't work here, it can't handle H3's paired
video/audio latent.

The latent carries both streams, so with it wired you don't need to load
the previous clip's video at all. The pinned frames are sliced straight
out of it rather than decoded to pixels and encoded again, so they're
exactly what the model made instead of a reconstruction. Nothing shifts
the colour or the contrast, so there's no seam to see, and it's faster.

Nothing to set for any of that. Leave `context_frames` unwired. Resolution
has to match between clips, since a latent can't be resized; if it doesn't
the node refuses and names both resolutions rather than quietly dropping
to the lossy path.

There's an older path for graphs with no latent: the previous clip's
frames into `context_frames`, its decoded audio into `context_audio`, and
the H3 audio VAE into `audio_vae`. It works, but it costs a lossy round
trip per link on both streams and it's where the visible seams came from.
Use the latent unless you have a reason not to.

### Reference mode

Put this node after `MiniMaxH3ReferenceToVideo` and wire it as above. Your
reference blocks (image, video, video with audio, audio) are left alone
entirely: the pinned sound rides as a keyframe, not a reference, so the
two mechanisms no longer share a list.

Nothing to configure. Worth mentioning only because versions before 0.2.0
overwrote that list, so turning chaining on quietly threw your references
away, and because up to 0.3.1 the pinned audio was added to it.

References still push the target timeline along, and the pinned frames
follow it. That is ComfyUI's own arithmetic now rather than this pack's.

### Alongside Add Guide

Anchor whatever you like with Add Guide for MiniMax H3 and put this node
after it. Both survive: the pinned head decides how the clip starts, the
guide decides what happens where you put it. Guides that carry audio keep
their own placement, which is forward from the anchor frame, while the
pinned sound keeps its backwards-reaching window.

Guides anchored inside the pinned head are dropped with a warning, for
the same reason a `first_frame` anchor is.

### Keeping a last-frame target

Wire `last_frame` on the stock conditioning node as usual and put this
node after it. The anchor is kept and pinned alongside the head: the
pinned run decides how the clip starts, your image decides where it
ends. It keeps the coordinates ComfyUI gave it, including the shift
references apply, so it stays put whatever else is in the graph.

A `first_frame` anchor is dropped, with a warning. The pinned head owns
those frames, and a second block pinned at the same instant with
different content would fight it. Older versions silently replaced both
anchors, so if a chain graph carried a last-frame target before this
version, it never reached the model.

## Settings

Two, because everything else had exactly one right answer.

**context_length** - frames of the previous clip's picture to carry over.
5, 22, 39 or 56. Those are the lengths that are a whole number of latent
steps, which is why the others aren't offered. 5 is just barely fluid, 22
is nearly seamless. Longer windows pin more motion but they come off the
front of the delivered clip, so 56 spends 2.3 seconds of every render on
frames you throw away. **Use 22.**

**audio_context_length** - frames of tail audio to pin, independent of the
picture window. It ends at the same instant as the pinned video, so this
only controls how far back the sound reaches. 0 follows context_length.
**Use 24**: that is exactly one second of sound, and any multiple of 3
lands exactly on the model's 40 Hz audio grid (a frame is 5/3 of an
audio step), so the window is pinned at precisely the width you asked
for. Off-grid values are widened to the nearest whole step. Increments
of 24 keep whole seconds: 48 pins the last two.

Everything else is fixed: the pinned run is encoded in one VAE call, it
sits at the head of the clip where the Trim node removes it, and the
pinned audio goes on this clip's own timeline. The alternatives all
existed only to reproduce their own failures, so they're constants at the
top of `nodes.py` now. Change one there if you ever need to see what they
did.

`match_tail` on the Trim node stays a setting because that node has no
idea what the other one did. Leave it on. H3 rounds its audio grid up, so
every clip carries about 8ms more sound than picture, and that error
stacks at every join.

## Writing prompts for a chain

The settings get motion and sound across the join. What happens in the
next clip is on you, and there are a few traps.

**The model renders contradictions as unions.** Clip N ends on a close-up
of A, clip N+1's prompt opens with "a two-shot of B and C," and the model
doesn't pick. You get all three. The pinned frames aren't a suggestion, so
a prompt describing a different arrangement of people reads as an addition
to them, not a replacement.

**The airlock.** Don't ask for the change and the continuation at the same
moment. Open clip N+1 holding clip N's exact closing framing, no dialogue,
about two seconds, then cut to the new setup. Joins done this way measure
tighter than an ordinary frame-to-frame cut. Joins that skip it measure
like two different rooms spliced together.

**Give the hold something to do.** A held framing with nothing happening
renders as a literal freeze, and two seconds of a motionless actor looks
like the video stalled. Write in a breath, a weight shift, an eyeline
change. The camera holds still, the performer doesn't.
`tests/freeze_detect.py` finds these.

**Budget the pinned head in your timecodes.** In `head` mode the clip
comes out `context_length` frames shorter than it was sampled. At 22 that's
0.92 seconds. Your prompt timings land against the sampled version, which
starts 0.92s earlier, so a beat you wrote for 4.0s shows up at 3.08s in the
file.

## Why the audio needed work

The first version ran pinned audio through H3's reference mechanism, which
is where audio conditioning normally goes. Every join had a small tick,
like the audio briefly sped up and went offbeat. Looking at the waveform
showed nothing wrong. Both sides of every join were smooth on their own.

Cross-correlating each clip's opening against the previous clip's ending
(that's `tests/seam_probe.py`) showed what was actually happening: the new
clip's audio resembled the old one. Same instruments, same groove, never
the same recording. A cover band. The model was reading the reference as
"a separate clip that sounds like this," which is what references are for
and exactly wrong for continuation.

The fix is the same one that already worked for video. The rows the model
sees are identical either way. What differs is their time coordinates, and
the coordinates are what say "separate clip" versus "this clip, earlier."
So the pinned audio still rides the reference machinery, but its
coordinates get rewritten onto the new clip's timeline, ending exactly
where the pinned video ends. Correlation at the joins went from about 0.45
with incoherent timing to 0.95+ with a flat offset, and the tick was gone.
Measured across a chain, the offset doesn't grow from join to join.

## Limitations

**Quality degrades down a chain.** The big one, and it's mostly audio.
Each clip is generated from the previous clip's output, which came from
the one before it. Losses compound like photocopying a photocopy, and in
audio the top end goes first. Timing and tempo stay locked, but after
several clips the sound gets duller and more muffled. Picture holds up
much better.

Two things stack per link: the model's own smoothing, and a round trip
out to pixels and back. `context_latent` removes the second one for both
streams. On picture that's the difference between a visible join and no
join at all. On sound it helps, but the model's own smoothing is still
there. Long chains are worth listening to critically, and restarts land
best at a natural musical transition.

**A small audio offset, now believed fixed.** Chained clips used to come
out about 8ms late. The cause was arithmetic: a frame is 5/3 of an audio
step, so unless the pinned window's length works out to a whole number of
steps it landed between the ones the model was filling. The window is now
snapped to that grid. The offset was constant, well under where lip-sync
errors get noticeable, and it didn't grow down a chain, so this is a
tidy-up rather than a rescue. `tests/seam_probe.py` reports the lag in
milliseconds if you want to check it on your own material.

**H3 emits 32 kHz audio, not 48.** Read the rate off the clip in any script
that remuxes or concatenates. A stream-copy concat can't change rate
partway, so a hardcoded 48000 silently turns the tail of a long episode
into nothing while every duration check still passes.
`tests/level_step.py` prints every clip's rate and flags a mismatch.

**Turbo LoRAs and Spectrum both cost you audio.** A turbo LoRA hits a
result in very few steps, and fine detail is what those last steps were
for. It thickens the sound and softens the picture. Step-skipping
optimizers like ComfyUI-Spectrum-MiniMax-H3 do the same thing to the
audio, less so to the picture, and they also mispredict the pinned rows,
which never change. Run both together and it stacks. If a chain sounds
duller or closer than you expected, try turning these off before blaming
the chaining. **Keep Spectrum off for these graphs.**

**Resolution can't change mid-chain** while using `context_latent`. A
latent can't be resized, so the node refuses. Regenerate the previous clip
at the new resolution, or start a fresh chain there.

**Prompting a chain takes work**, especially with reference mode. Every
reference conditions the whole clip; there's no way to say "this one
starts two seconds in." Timing has to come from the shot structure and
the description. See "Writing prompts for a chain" above.

**Tested narrowly.** Joins have been verified on dense beat-driven
electronic music, where timing errors are most audible, and on spoken word
through the latent path, where nothing hides a seam. One Windows machine,
one resolution, one sampler. The math self-tests every startup. The
perceptual results are one person's renders.

**ComfyUI's H3 support is young.** This pack no longer patches any of it,
but it still depends on how the layout places things, and one of those
dependencies has no upstream test behind it: the pinned audio window is
positioned with a fractional, negative anchor index. That is legal
arithmetic in the layout and no stock node can produce one, so nothing
upstream exercises it. `layout_contract.py` checks it before the first
render and refuses if it ever stops holding.

The failure mode after a ComfyUI update is therefore "the node won't
run," not bad output. There is a worked example: when the layout
constructor changed, 0.3.0 of this pack refused until 0.3.1 caught up.
Annoying, but the alternative was joins landing at the wrong instant with
nothing to tell you.

**License.** The H3 community license reportedly doesn't currently cover
the EU, UK, Korea or the US. Check for yourself before shipping anything
on it.

## Recommended starting point

`context_length 22`, `audio_context_length 24`, `context_latent` wired
through the Save/Load Latent pair, Trim node wired for picture and sound
with `match_tail` on, Spectrum off. Every "it works" in this README means
that config.

## Testing

Six scripts, all runnable without ComfyUI or a GPU.

```
python tests/_mock_harness.py        # the layout checks against a fake stock model
python tests/_node_smoke_test.py     # the node end to end, refs + save/load
python tests/_probe_node_test.py     # the seam probe node, joins with known answers
python tests/seam_probe.py A.flac B_untrimmed.flac    # is the join real continuation?
python tests/level_step.py clip*.flac                 # does the level or room tone jump?
python tests/freeze_detect.py clip*.mp4               # did a held shot render as a still?
```

The first three print their checks and end with a pass line. The other
three measure real output; `level_step` and `freeze_detect` take
`--self-test` to check their own math on made-up data first, and
`seam_probe`'s math is what `_probe_node_test.py` exercises.

The three measurement scripts ask different questions and a join can fail
any one on its own. `seam_probe` is timing: is the audio the same waveform
continued, or a sound-alike. `level_step` is volume: does the loudness, or
the room tone underneath it, jump at the cut. `freeze_detect` is picture:
is anything happening in that shot.

For `seam_probe` and `level_step`, give them the UNTRIMMED audio of the new
clip. Branch the audio decode to a second Save Audio node alongside the
Trim.

### Measuring a join in-graph

The H3 Motion Context Seam Probe node runs the same measurements inside
the graph, where the seam position is known exactly rather than inferred
from file ends (that inference is where the CLI probe's phantom ~8 ms lag
came from). Wire it inline between the audio VAE decode and the Trim
node:

```
clip_a_latent     the same latent wired into context_latent
audio_vae         the same audio VAE
clip_b_untrimmed  this clip's audio off the VAE decode, before the trim
trim_frames       from the Motion Context node, same value the Trim gets
```

The audio output is the input unchanged, so it drops into an existing
chain without changing the render. The report goes to a Preview Text
node: lag, correlation, broadband and floor steps, judged against the
same thresholds the CLI scripts document.

## Upgrading

[CHANGELOG.md](CHANGELOG.md) lists what changed in each release and which
ComfyUI H3 layout it works with. Worth a look before opening an issue or
starting a fork: several fixes people have rebuilt from scratch were
already in an earlier release.

The node's widgets changed. ComfyUI stores widget values by position, so a
workflow saved against an older version will load its numbers into the
wrong slots. Delete the Motion Context node, add it again and rewire it.
Takes a minute and beats rendering with scrambled settings.

**Only install this once.** A fork of this repo, a manual clone next to a
Manager install, or a renamed backup still sitting in `custom_nodes` will
all load. Renaming a folder does not stop ComfyUI loading it.

Older versions of this pack, and other H3 packs, patch ComfyUI's layout
code at runtime. This one does not, so it no longer competes for that
code and will run alongside them. If it finds the layout wrapped it says
who by, then checks whether anchors still land correctly and carries on
if they do. Keeping one H3 chaining pack installed is still the quiet
life.

## Credits

The Ref2VA multi-reference support was worked out by **seitanism** in the
Banodoco MiniMax H3 seamless-extension thread and first implemented in
**ethanfel**'s fork of this repo. The code here is written independently,
but the idea is theirs and they got there first.

The prompting section comes out of a 16-clip, 4:34 multicam sitcom episode
chained with this pack at 736x576 on a 5070 Ti, about two hours of
generation. The airlock, the freeze warning, the 32 kHz trap and the
timecode budget are all that build's findings. Same run measured a
video-only chain at a median seam level step of 0.905, dropping to 0.16
with the audio carried across.

**javawock7618** and **azra1l** reported the layout change that broke
0.3.0 and narrowed it to specific commits, which turned a hunt into a
diff. **ChimeraWerks** found and fixed the audio grid overhang.

If you build something long with this, numbers are more useful than praise.
Open an issue.

## Files

| File | Role |
|---|---|
| `layout_contract.py` | Proves ComfyUI still places anchors and pinned audio where this pack needs them, once, before the first render. Modifies nothing. |
| `nodes.py` | The four core nodes: Motion Context, Trim, and the latent Save/Load pair. |
| `probe_node.py` | The Seam Probe node: measures a join in-graph, with the seam at a known sample instead of inferred from file ends. |
| `tests/seam_probe.py` | Is a join's audio a real continuation, a sound-alike, or drifting. |
| `tests/level_step.py` | Level and room-tone continuity at each join. Also catches sample-rate mismatches. |
| `tests/freeze_detect.py` | Stretches where the picture stops moving. |
| `tests/_mock_harness.py`, `tests/_node_smoke_test.py`, `tests/_probe_node_test.py` | Layout, node and probe tests, numpy only. |
| `CHANGELOG.md` | What changed in each release, and which ComfyUI H3 layout it works with. |
