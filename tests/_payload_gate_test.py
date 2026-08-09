"""The payload patch must only touch graphs that came from this pack.

`MiniMaxH3.extra_conds` is wrapped process-wide, so every H3 graph in the
process runs through it. A Ref2VA workflow with a last_frame anchor is a
legitimate combination of keyframes and refs that has nothing to do with
chaining, and it must come out of the wrapper bit-identical to stock.

Fakes comfy.model_base with a MiniMaxH3 whose extra_conds reproduces the
stock overwrite, then checks four cases:

  1. stock keyframes + stock refs      -> untouched (refs overwrite, as stock)
  2. marked keyframes + marked ref     -> merged (the pack's own graph)
  3. marked keyframes, unmarked refs   -> merged (chaining under Ref2VA)
  4. one mechanism only                -> untouched either way
"""

import sys
import types

_PKG = __file__.rsplit("/", 2)[0]
sys.path.insert(0, _PKG)

MC_KEY = "motion_context_index"
MC_AUDIO_KEY = "motion_context_audio_end_frame"


class Cond:
    def __init__(self, cond):
        self.cond = cond


def make_model_base():
    mb = types.ModuleType("comfy.model_base")

    class MiniMaxH3:
        def extra_conds(self, **kwargs):
            payload = {}
            kfs = kwargs.get("minimax_keyframes")
            refs = kwargs.get("minimax_refs")
            if kfs is not None:
                payload["cond_video_latents"] = [kf["latent"] for kf in kfs]
            if refs is not None:
                # the stock overwrite this patch exists to work around
                payload["cond_video_latents"] = [r["latent"] for r in refs
                                                 if "latent" in r]
                payload["cond_audio_latents"] = [
                    r["audio_latent"] for r in refs
                    if r.get("audio_latent") is not None]
            fc = kwargs.get("minimax_frame_count")
            if fc is not None:
                payload["frame_count"] = fc
            return {"minimax_payload": Cond(payload)}

    mb.MiniMaxH3 = MiniMaxH3
    sys.modules.setdefault("comfy", types.ModuleType("comfy"))
    sys.modules["comfy.model_base"] = mb
    sys.modules["comfy"].model_base = mb
    return mb


def main():
    mb = make_model_base()
    sys.modules.pop("patch_payload", None)
    import importlib
    pp = importlib.import_module("patch_payload")
    assert pp.apply_patch(), "payload patch did not apply"

    model = mb.MiniMaxH3()

    def run(kfs, refs):
        return model.extra_conds(minimax_keyframes=kfs, minimax_refs=refs,
                                 minimax_frame_count=124)["minimax_payload"].cond

    # 1. an unrelated graph: stock keyframes plus stock refs. Stock drops
    # the keyframe latent here. Right or wrong, that is not this pack's
    # call, and the wrapper must leave it alone.
    plain_kf = [{"resolved_frame_index": 0, "latent": "KF"}]
    plain_ref = [{"kind": "image", "latent": "REF"}]
    got = run(plain_kf, plain_ref)
    assert got["cond_video_latents"] == ["REF"], got
    print("1. stock keyframes + stock refs: untouched, stock overwrite intact")

    # 2. this pack's own graph: marked keyframes and a marked audio ref
    mc_kf = [{"resolved_frame_index": 0, MC_KEY: 0, "latent": "KF"}]
    mc_ref = [{"kind": "audio", MC_AUDIO_KEY: 22.2, "audio_latent": "A"}]
    got = run(mc_kf, mc_ref)
    assert got["cond_video_latents"] == ["KF"], got
    assert got["cond_audio_latents"] == ["A"], got
    print("2. marked keyframes + marked ref: keyframe latent preserved")

    # 3. chaining under Ref2VA: the graph's own references are unmarked,
    # ours is marked, and every video latent must survive in list order
    mixed_refs = [{"kind": "image", "latent": "R1"},
                  {"kind": "video_audio", "latent": "R2", "audio_latent": "A2"},
                  {"kind": "audio", MC_AUDIO_KEY: 22.2, "audio_latent": "A3"}]
    got = run(mc_kf, mixed_refs)
    assert got["cond_video_latents"] == ["KF", "R1", "R2"], got
    assert got["cond_audio_latents"] == ["A2", "A3"], got
    print("3. marked keyframes + Ref2VA refs: merged in layout row order")

    # 4. one mechanism only, marked or not, is stock either way
    assert run(mc_kf, None)["cond_video_latents"] == ["KF"]
    assert run(None, mixed_refs)["cond_video_latents"] == ["R1", "R2"]
    print("4. one mechanism only: unchanged")

    print("payload gate test passed")


if __name__ == "__main__":
    main()
