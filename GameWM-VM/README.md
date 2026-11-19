# GameWM-VM

The video-model half of GameWM. Given what the
[GameWM-engine](../GameWM-engine) resolved for a take (a coarse proxy video
from the player's camera, a first frame, an optional character sheet and a
caption with the skill being used), GameWM-VM generates what is expensive to
author by hand: the character's limbs and contact, the skill's projectile and
impact, the target's reaction, and the appearance of the whole scene.
Positions, timing and outcomes stay the engine's.

GameWM-VM is not a renderer of the proxy. A renderer is trained on pairs where
each proxy has exactly one correct output, so it can only restyle the motion
the engine authored. GameWM-VM is trained on interaction-centric pairs where
one proxy sequence has several targets, and it realizes one engine event as
the skill of whichever character the game supplies: the same `ranged_attack`
becomes a fireball, an ice shard or a lightning call, with the attacker's
motion following the skill, on the engine's hit frame.

GameWM-VM is built on [SolarWM](https://github.com/Junchao-cs/SolarWM)
(Apache-2.0); the Python package and CLI keep the name `solarwm`. On top of
SolarWM's camera-controlled world models it adds **proxy-conditioned Ref2VA**
training and inference for three backbones.

## Model family

| Model | Backbone | Proxy entry | Adaptation | Clip | Tier |
|---|---|---|---|---|---|
| GameWM-H3-33B | MiniMax-H3 (33B) | Native Ref2VA reference rows | LoRA r128 (attention Q/K/V/O) | 1280×704, 124 frames, 24 fps | offline quality, teacher |
| GameWM-Wan2.2-14B | Wan2.2-I2V-A14B, high-noise expert only | Sequence prefix, half resolution | LoRA r128 | 832×480, 81 frames, 16 fps | low-cost offline |
| GameWM-Wan2.2-5B | Wan2.2-TI2V-5B | Sequence prefix, half resolution | Full fine-tuning | 1280×704, 121 frames, 24 fps | real-time player tier |

All three share one interface:

| Slot | Input |
|---|---|
| `<Picture 1>` | first frame: sets the world's look |
| `<Picture 2>` | character sheet (front / side / back), optional: sets the protagonist's identity |
| `<Video 1>` | proxy video: sets layout, camera, motion and event timing |
| text | `Reference modality: {style\|depth\|semantic\|duv} video.`, the identity line, then the caption |

One model accepts four proxy modalities: low-poly / capsule / voxel re-renders
(`style`), depth, semantics, and packed depth + semantics (DUV). The full
request contract is in
[`GameWM-engine/protocol/PROTOCOL.md`](../GameWM-engine/protocol/PROTOCOL.md).

## Training stages

The SolarWM three-stage recipe, applied to the proxy interface:

1. **BD (Stage0.5)**: bidirectional flow matching on whole clips. The
   `-Sheet` variants add the character sheet as a second picture, dropped for
   a fifth of the samples so the first frame alone still works.
2. **AR (Stage1)**: teacher-forced AnyFlow causal initialization.
3. **DMD (Stage2)**: distribution matching with self-gradient forcing; a
   four-step causal student. Chunk *k* reads only proxy chunks *k−5 … k*.

Long takes are generated in overlapping windows: the first receives one latent
of context, later ones ten (124-frame segments, stride 85).

## Code map

| Path | What |
|---|---|
| `src/solarwm/backends/minimax_h3/` | H3 backend. Proxy Ref2VA: `stage0p5_ref2va.py`, `stage1_ref2va.py`, `stage2_ref2va.py`, `ref2va_layout.py`, `proxy_inference.py`, `proxy_rollout.py` (long takes), `proxy_serve.py` (resident workers), `proxy_loop.py` (experiment CLI) |
| `src/solarwm/backends/wan22/ref2va/` | Wan2.2 5B / 14B Ref2VA: sequence layout, LoRA, BD / AR / DMD cores, runtime |
| `configs/examples/minimax_h3/*ref2va*` | H3 proxy omni configs (train and infer, all stages) |
| `configs/examples/wan22_ti2v_5b/ref2va-*`, `configs/examples/wan22_i2v_a14b/ref2va-*` | Wan Ref2VA configs |
| `docs/backends/minimax-h3.md` | H3 guide, including the Ref2VA proxy and omni sections |
| `docs/runbooks/h3-proxy-*.md` | Experiment loop and trainability checklist |
| `docs/experiments/gta0930-omni-runs.json` | Registry of the gta0930 omni runs |
| `data-engine/` | SolarWM's data engine (git submodule) |

Proxy caches (`.pt` samples with `vae_latent`, `proxy_latents`, `anchor_latent`,
optional `oc_latent`, text embeddings and `info`) are produced by FastVideo-oc
(`scripts/h3_proxy/prepare_data/encode_proxy_samples.py`,
`encode_wan_proxy_samples.py`).

## Install

Each backbone has its own runtime environment
([`environments/`](environments/README.md)). In the selected one:

```bash
git submodule update --init GameWM-VM/data-engine   # from the GameWM root
cd GameWM-VM
python -m pip install -e .
solarwm environment probe
```

## Run

Train (one `torchrun` per node; H3 BD-Sheet shown):

```bash
torchrun --nnodes=$NNODES --node-rank=$NODE_RANK --nproc-per-node=8 \
  --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m solarwm train \
  --config configs/examples/minimax_h3/stage0p5-124f-ref2va-omni-mixed-low-704p-sp1.yaml \
  --set model.checkpoint_path=/path/to/MiniMax-H3 \
  --set data.data_path=/path/to/proxy-cache-manifest.txt \
  --set runtime.output_dir=/path/to/runs/my-run
```

Inference on a cached manifest (single node):

```bash
python -m solarwm.runtime.local_file_launcher --nproc-per-node 8 --store-path /tmp/store -- \
  -m solarwm infer --config configs/examples/minimax_h3/infer-stage0p5-124f-ref2va-omni-mixed-low-704p-sp8.yaml \
  --set data.data_path=/path/to/heldout-manifest.txt \
  --set checkpoint.resume_from=/path/to/checkpoint_model_000600 \
  --set checkpoint.weight_source=live \
  --set runtime.output_dir=/path/to/evals/my-eval
```

Resident conversion workers, one per GPU, fed by the engine adapter or the
conversion UI through a spool directory:

```bash
CUDA_VISIBLE_DEVICES=0 python -m solarwm.backends.minimax_h3.proxy_loop serve \
  --run gta0930-r5b-oc-r4a900-lr1e-4-s600 --step 600 --weight-source live \
  --spool /workspace/serve/spool --worker gpu0 --sp 1 \
  --data /path/to/any-small-manifest-of-the-run.txt \
  --infer-config configs/examples/minimax_h3/infer-stage0p5-124f-ref2va-omni-mixed-low-704p-sp8.yaml
```

## Checkpoints

Released weights are on Hugging Face at
[binghe-sjtu/GameWM](https://huggingface.co/binghe-sjtu/GameWM), as
`<model>/<report name>/` (for example `GameWM-H3-33B/BD-Sheet/`). They hold only the live
weights: pass an H3 folder as `checkpoint.resume_from` with `checkpoint.weight_source=live`, and a
Wan folder as `inference.checkpoint.path`. [`tools/release/export_weights.py`](tools/release/export_weights.py)
builds that tree from the training runs below.

H3 runs (`h3_proxy/solarwm-runs/<run>/checkpoint_model_<step>/`, LoRA in
`adapter.pt`, EMA in `ema.pt`):

| Report name | Run | Step |
|---|---|---|
| BD | `gta0930-r4a-lr2e-4-s1200` | 900 |
| AR | `gta0930-s1b-r4a900-lr1e-4-s600` | 600 |
| DMD | `gta0930-s2b-s1b600-r4a900-b8-s1000` | 1000 |
| BD-Sheet | `gta0930-r5b-oc-r4a900-lr1e-4-s600` | 600 |
| AR-Sheet | `gta0930-s3a-oc-r5b600-b16-lr1e-4-s900` | 900 |
| DMD-Sheet | `gta0930-s4a-oc-s3a900-r5b600-b8-s2000` | 1000 |

Wan runs (`wan_proxy/runs/<run>/`):

| Model | BD | BD-Sheet | AR-Sheet | DMD-Sheet |
|---|---|---|---|---|
| 5B | `wan5b-ref2va-stage0p5-121f` | `wan5b-ref2va-stage0p5-oc-121f` | `wan5b-ref2va-stage1-oc-121f` | `wan5b-ref2va-stage2-oc-121f` |
| 14B | `wan14b-ref2va-stage0p5-81f-v2` | `wan14b-ref2va-stage0p5-oc-81f` | `wan14b-ref2va-stage1-oc-81f-r2` | `wan14b-ref2va-stage2-oc-81f` |

## License and attribution

The code is Apache-2.0, inherited from SolarWM; see [LICENSE](LICENSE) and
[NOTICE](NOTICE). This directory is a modified copy of SolarWM (branch
`wan-ref2va` at `acda90b`). Backbone weights carry their own licenses: the
MiniMax H3 Community License (with territory restrictions) and the Wan2.2
license. Please also cite SolarWM:

```bibtex
@misc{huang2026solarwmopendatascalable,
  title={SolarWM: Open Data and Scalable Training for Long-Horizon Video World Models},
  author={Junchao Huang and Guian Fang and Shengju Qian and Xianghao Kong and Zhuoran Zhao and Wei Huang and Yihua Du and Zixin Zhang and Justin Cui and Yuchao Gu and Yukang Chen and Xinting Hu and Tianyu He and Shaoshuai Shi and Zhuotao Tian and Xin Wang and Mike Zheng Shou and Li Jiang},
  year={2026},
  eprint={2609.02886},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.02886},
}
```
