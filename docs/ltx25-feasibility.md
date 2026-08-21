<!-- Track 3 research, 2026-08-21. Sources cited inline; unverified items listed at the end. -->

# LTX-2.5 as an audio-conditioned, mask-local lip re-articulation editor for Dubline

Sources checked directly: `Lightricks/LTX-2` repo at commit `400fd31` (2026-08-16) — pipelines, `ltx-core` conditioning classes, `ltx-trainer` configs/docs; `Lightricks/ComfyUI-LTXVideo` example workflows; HF model cards; the JUST-DUB-IT paper; HF discussion threads. Claims I could only get from secondary summaries are flagged.

## 1. What LTX-2.5 is

- **Architecture.** 22B asymmetric dual-stream DiT (video stream + audio stream, coupled by bidirectional cross-attention, shared timestep/AdaLN), flow-matching. The LTX-2 tech report ([arXiv 2601.03233](https://arxiv.org/abs/2601.03233)) describes the family design (14B video + 5B audio in LTX-2; 2.5 is 22B total). Text encoder is a Lightricks-finetuned Gemma-4-12B with bundled projection (stock Gemma is rejected). Video VAE latent is **8× temporal, 32×32 spatial** (`SpatioTemporalScaleFactors.default() = (8,32,32)` in `packages/ltx-core/src/ltx_core/types.py`); a "DiffVAE" diffusion decoder and a lighter conv decoder are both shipped. Audio VAE: 8 channels × 16 mel bins latent, with a vocoder ([HF card](https://huggingface.co/Lightricks/LTX-2.5)).
- **Checkpoints** ([HF repo](https://huggingface.co/Lightricks/LTX-2.5)): `ltx-2.5-22b-dev` (full, guided 2-stage), `ltx-2.5-22b-distilled` (8+4 sigma steps; what `ICLoraPipeline`/`DubItPipeline` expect), distilled LoRA-450, 2× spatial and 2× temporal latent upscalers, duration head, and a separate [2.5 Pixel-Spatial-Upscaler IC-LoRA](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler). Also NVFP4 and ComfyUI int8 variants. **[LTX-2.5-Pre-Trained](https://huggingface.co/Lightricks/LTX-2.5-Pre-Trained)**: a 43 GiB single-file pre-SFT bundle (DiT + both VAEs + vocoder) explicitly "for researchers building specialized models" with the LTX-2 Trainer.
- **License.** "LTX-2.x Community License": free commercial use under $10M annual revenue; paid agreement above. The Pre-Trained card warns that *transferring fine-tuned models may require paid licensing* — relevant if Dubline ships an adapter.
- **Limits.** W/H divisible by 32, `num_frames % 8 == 1`, 24/25 fps typical; example inference up to 121–161 frames; 4K HDR via DFR pipeline. VRAM: README recommends `--quantization fp8-cast --offload cpu|disk` under memory pressure; on a 32 GB 5090, fp8/NVFP4 inference of the 22B is the expected path (bf16 transformer alone is ~44 GB). Not verified: exact peak VRAM for 1080p×121f on 32 GB.

## 2. Conditioning pathways that exist today (official code)

From `packages/ltx-core/src/ltx_core/conditioning/types/`:

| Class | What it does |
|---|---|
| `VideoConditionByLatentIndex` / `VideoConditionByKeyframeIndex` | first-frame / keyframe latent replacement (I2V, keyframe interpolation) |
| `VideoConditionByReferenceLatent` | **IC-LoRA**: clean reference latents appended as extra tokens sharing RoPE positions with the target; supports downscaled references |
| `AudioConditionByReferenceLatent` | reference audio tokens appended at **negative RoPE positions** (speaker identity, used by Dub-It) |
| `VideoConditionByMask` | **spatial-temporal inpainting mask in latent space** `[B,F,H,W]`: mask=1 → clean latent, no denoising; mask=0 → generated |
| `TemporalRegionMask` | regenerate only a time window (`RetakePipeline`) |

Pipelines (`packages/ltx-pipelines/docs/pipelines.md`):
- **A2V (`a2vid_two_stage.py`)**: you *supply* an audio file; it is VAE-encoded and passed as a **frozen** modality (`frozen=True`, `noise_scale=0`) while video is denoised. So yes, the base model can follow a given final audio track. No source-video input in this pipeline.
- **IC-LoRA V2V (`ic_lora.py`)**: source video as reference tokens + LoRA (depth/pose/canny/detailer/etc.).
- **Dub-It (`dubit.py`)**: source video → IC-LoRA reference tokens; source *audio* → reference tokens for voice identity; **new audio and video are jointly generated from the prompt text**. Audio is *not* user-supplied (`audio: ModalitySpec(..., conditionings=audio_conditionings)` is generated in stage 1, frozen only in stage 2). Official docs confirm: text-driven, "mask-free", single speaker, "LTX-2.5 support is in development" ([docs](https://docs.ltx.io/open-source-model/advanced-workflows/lip-dub-beta); [model card](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-DubIt)).
- **Inpainting (ComfyUI)**: `LTX-2.5_ICLoRA_Inpaint_Two_Stage_Distilled.json` takes **original video + B&W mask video + optional prompt + the original audio frozen** (`VAEEncodeAudio → SetAudioRefTokens`), uses the [In-Outpainting IC-LoRA](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting) (2.3-trained, listed as usable on 2.5), composites green under the mask before diffusion, and **Laplacian-pyramid blends** the decode back onto the original at both stages. Workflow notes warn about "green leftovers at mask edges" and that dilation is "crucial".
- **Retake**: temporal-window regeneration of an existing video, optionally regenerating audio too.

## 3. IC-LoRA mechanism and the closest existing adapters

IC-LoRA = LoRA on attention (`to_q/k/v/out`) trained with paired (reference, target) latents concatenated in the token sequence; reference tokens get sigma=0 and are excluded from loss (`docs/training-modes.md`). Lightricks' 2.3 zoo includes Depth/Pose/Canny/Union control, Detailer/Upscaler, Deblur, Colorization, Relight, Clean-Plate, **In-Outpainting**, **Instant-Shave**, **Dub-It** (HF API listing). 2.3 LoRAs "mostly" run on 2.5 per Lightricks' HF discussion; only the Pixel-Spatial-Upscaler has a native 2.5 build so far.

Closest adapters to "masked region regeneration conditioned on audio":
1. **In-Outpainting IC-LoRA + frozen audio** — already exactly "original frames + mask + given audio → filled frames"; it just wasn't trained with mouth masks or for phoneme-level sync.
2. **Dub-It IC-LoRA / JUST-DUB-IT** (Tel Aviv U + Lightricks, SIGGRAPH 2026, [arXiv 2601.22143](https://arxiv.org/abs/2601.22143), [project](https://justdubit.github.io/), [code](https://github.com/justdubit/just-dub-it)) — full-frame V2V with audio generated jointly. Key technical lesson for Dubline: their **"latent-aware masking"** — because the VAE receptive field is 32×32×8, lip information leaks into neighbouring jaw/cheek tokens; they compute the effective latent mask as `|F_mask − F_empty| > 0.1` and regenerate that whole region. Training recipe: rank-128 LoRA on attn+FFN, 2,000 steps, batch 1, LR 2e-4 video / 1e-5 audio, 960×544×121 @25 fps, synthetic multilingual pairs generated by LTX-2 itself. Reported on their "challenging" benchmark: CSIM 0.646 vs LatentSync 0.707 / MuseTalk 0.577; FVD 354 vs 759/902; ASync 2.44 vs 1.34/5.65 (i.e., LatentSync still scores better on raw sync and identity; LTX wins on temporal quality and robustness). Limitation stated: voice identity not always preserved (irrelevant for Dubline, which supplies audio).
3. Community: [fbjr Audio-Only-Context IC-LoRA](https://huggingface.co/fbjr/LTX-2.3-22b-IC-LoRA-Audio-Only-Context) — trained on a single 24 GB 4090, 1,000 steps, rank 32; proves small-GPU IC-LoRA training is practical, though it generates rather than edits.

## 4. Fine-tuning support

The trainer's **"flexible" strategy** (`packages/ltx-trainer/docs/training-modes.md`) composes per-modality conditions in YAML. The exact Dubline configuration is expressible with no code changes:

```yaml
video:
  is_generated: true
  conditions:
    - type: mask        # per-sample mouth-mask video; 1=keep (clean, sigma=0, no loss), 0=generate
      mask_dir: "video_masks"
    - type: reference   # optional: the untouched source clip as IC tokens
      latents_dir: "reference_latents"
audio:
  is_generated: false   # frozen dubbed/original audio = pure conditioning
```

- **Masked loss is native**: loss is computed only on mask=0 tokens (`flexible.py` `loss_mask`; docs: "tokens with mask > 0.5 receive clean latents and timestep=0 and are excluded from loss"). Masks are given as a `video_mask` column and downsampled to latent grid by `process_dataset.py`.
- **LoRA vs full FT**: LoRA on dev/distilled/pre-trained. Full fine-tune "requires 4–8× H100 80GB with FSDP".
- **VRAM**: 80 GB recommended; an official **`t2v_lora_low_vram.yaml` for 32 GB (RTX 5090)** uses int8-quanto transformer, 8-bit Gemma, adamw8bit, rank 16, grad checkpointing, 576×576×49 buckets. Adding a reference stream doubles the video token count, so on 32 GB expect ~512–576 px crops and ≤49–73 frames per sample; 1080p full-frame training is not feasible locally.

## 5. Evidence of lip-sync/dubbing with LTX-2.x

- Official Dub-It IC-LoRA (2.3) + pipelines + ComfyUI workflow; Lightricks says 2.5 port "very soon" ([HF discussion #44](https://huggingface.co/Lightricks/LTX-2.5/discussions/44)).
- JUST-DUB-IT paper (above) with quantitative comparison to LatentSync/MuseTalk/X-Dub/HeyGen.
- Supplied-audio lipsync on 2.5 without a LoRA: community workaround in discussion #44 (freeze audio latent, Euler, disable modality guidance); reported as "highly material-dependent and seed-sensitive", sometimes "refusing to sync". Lightricks' own X article distinguishes "Lipsync" (supplied audio, I/A2V) from "LipDub" (text-driven) — I could not fetch it (paywalled 402).
- No repo or paper found that does **mask-local, supplied-audio** lip editing on LTX-2.x. That specific combination is unbuilt.

## 6. Feasibility for Dubline

**(a) Zero-shot.** Partially. Two zero-shot routes exist, both lossy for Dubline's goal:
- *Dub-It*: wrong interface (it synthesizes the audio from text; Dubline already has the dubbed track), 2.3-only, full-frame regeneration through the VAE (everything outside the mouth is re-rendered, though visually preserved), 4.8 s training length.
- *Inpainting IC-LoRA + mouth mask + frozen dubbed audio*: correct interface, but the adapter was never trained to articulate to audio — base-model A2V prior may yield sync sometimes (cf. discussion #44's seed sensitivity). Worth a 1-day test, not a product path.

**(b) Adapter.** Yes, and the tooling is unusually close. The `mask + frozen audio (+ reference)` flexible config gives masked-loss, audio-frozen, region-local training out of the box. Data: HDTF/VoxCeleb2/TalkingHead-1KH-style clips with their **original** audio as the frozen condition and a dilated mouth/jaw mask (derive the *latent-effective* mask à la JUST-DUB-IT: dilate ≥1 latent cell = 32 px, and temporally ≥8 frames). Include bearded speakers deliberately. Scale: JUST-DUB-IT and the community LoRA converged in 1–2k steps at batch 1; budget 2–5k steps, rank 32–64, on 512–768 px face crops × 49–121 frames. On a 5090 with the low-VRAM config that is on the order of 1–3 days; an 80 GB rental makes 960×544×121 possible in hours.

**(c) Risks.**
- *Resolution*: 1080p faces must be processed as crops (the docs themselves suggest cropping for Dub-It) and pasted back — so some crop-boundary handling remains, but at the crop edge in untouched pixels, not at the mouth.
- *Latent granularity/seams*: the mask is 32×32×8; outside-mask pixels are still VAE round-tripped unless you Laplacian-blend the original back in (the official inpaint workflow does exactly this). Beard texture inside the dilated region will be regenerated; the 22B prior with the source as in-context reference is far stronger at hallucinating consistent beard than MuseTalk's 256-px UNet, but it is still regeneration.
- *Length*: 10 s utterances = 241 frames @24 fps; trained lengths are ≤121–161 frames. Needs chunking with prefix/suffix (video-extension) conditioning or retake-style windows; untested.
- *Identity drift*: JUST-DUB-IT's CSIM trails LatentSync, so measure it.
- *Licensing*: community license OK for a small project; adapter redistribution clause unclear.
- *VRAM*: fp8/NVFP4 inference fits; training needs the int8 low-VRAM config.

**Versus crop inpainters (MuseTalk/LatentSync)**: LTX sees the whole head in context and models motion jointly, so mouth-patch boundary and head-motion integration are structurally better (FVD 354 vs 759–902 in the paper), and beard/jaw are regenerated with a full-face prior rather than a 256-px mask patch. It costs ~10–50× compute per clip and currently scores lower on raw SyncNet-style offset than LatentSync.

## VERDICT: adapter needed (not zero-shot; clearly a fit for the architecture)

**Next experiment (1–2 weeks, single 5090 + optional 80 GB rental):**
1. *Day 1 baseline*: run ComfyUI `LTX-2.5_ICLoRA_Inpaint_Two_Stage_Distilled.json` with a dilated mouth mask and the dubbed track as frozen audio on 5 Dubline clips; score with SyncNet (LSE-C/D), ArcFace CSIM, LPIPS outside the mask (should be ~0 after pyramid blend).
2. *Data*: 2–5k clips, 512–768 px face-tracked crops, 49–121 frames, original audio; masks = mouth+jaw landmarks, dilated to latent-effective extent; `video_mask` column; `process_dataset.py` with `--reference-downscale-factor 1`.
3. *Train*: flexible strategy `mask + reference` on video, audio `is_generated:false`, rank 32–64 on `to_q/k/v/out` (+FFN if VRAM allows), LR 1e-4 video, 2–4k steps, on `ltx-2.5-22b-dev` (or distilled for 8-step inference); low-VRAM int8 config on the 5090.
4. *Evaluate*: LSE-C/D vs MuseTalk/LatentSync on held-out speakers, CSIM, LPIPS/PSNR outside mask, and a bearded-speaker subset judged visually; also test a 241-frame utterance via two overlapping windows with prefix conditioning.

**Could not verify**: exact VRAM of 2.5 fp8 inference at 1080p×121f; the X "Lipsync vs LipDub" article; whether the 2.3 In-Outpainting and Dub-It LoRAs degrade on 2.5; JUST-DUB-IT training GPU count; any Lightricks statement on per-token audio→video sync quality with frozen audio and no LoRA.
