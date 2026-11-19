"""Export the released GameWM checkpoints into the tree of the Hugging Face model repository.

    python tools/release/export_weights.py --h3-runs /data/.../h3_proxy/solarwm-runs \\
        --wan-runs /data/.../wan_proxy/runs --out /workspace/hf-release/GameWM [--only GameWM-Wan2.2-14B]

Only the weights inference reads are kept:

- GameWM-H3-33B (LoRA r128): ``adapter.pt`` (live weights). The checkpoint manifest and
  ``COMPLETE.json`` are rewritten to list just that file, so ``checkpoint.resume_from=<dir>`` with
  ``checkpoint.weight_source=live`` loads it through ``verify_checkpoint`` as before.
- GameWM-Wan2.2-5B (full fine-tuning) and GameWM-Wan2.2-14B (LoRA on the high-noise expert):
  ``student.safetensors`` (live weights), loaded with ``inference.checkpoint.path=<dir>``.

Every variant directory also gets ``release.json``: source run, step, stage, base model, bytes and
SHA-256 of each file. Optimizer, critic, EMA and resume state are not exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from solarwm.checkpoint import verify_checkpoint
from solarwm.runtime.serialization import canonical_json_bytes

BASES = {
    "GameWM-H3-33B": "MiniMax-H3 (transformer_ref)",
    "GameWM-Wan2.2-14B": "Wan-AI/Wan2.2-I2V-A14B (high_noise_model)",
    "GameWM-Wan2.2-5B": "Wan-AI/Wan2.2-TI2V-5B",
}
# model -> variant -> (run, step, stage); the checkpoints of the technical report
RELEASE = {
    "GameWM-H3-33B": {
        "BD": ("gta0930-r4a-lr2e-4-s1200", 900, "stage0p5"),
        "AR": ("gta0930-s1b-r4a900-lr1e-4-s600", 600, "stage1"),
        "DMD": ("gta0930-s2b-s1b600-r4a900-b8-s1000", 1000, "stage2"),
        "BD-Sheet": ("gta0930-r5b-oc-r4a900-lr1e-4-s600", 600, "stage0p5"),
        "AR-Sheet": ("gta0930-s3a-oc-r5b600-b16-lr1e-4-s900", 900, "stage1"),
        "DMD-Sheet": ("gta0930-s4a-oc-s3a900-r5b600-b8-s2000", 1000, "stage2"),
    },
    "GameWM-Wan2.2-5B": {
        "BD": ("wan5b-ref2va-stage0p5-121f", 1200, "stage0p5"),
        "BD-Sheet": ("wan5b-ref2va-stage0p5-oc-121f", 600, "stage0p5"),
        "AR-Sheet": ("wan5b-ref2va-stage1-oc-121f", 900, "stage1"),
        "DMD-Sheet": ("wan5b-ref2va-stage2-oc-121f", 2000, "stage2"),
    },
    "GameWM-Wan2.2-14B": {
        "BD": ("wan14b-ref2va-stage0p5-81f-v2", 1200, "stage0p5"),
        "BD-Sheet": ("wan14b-ref2va-stage0p5-oc-81f", 600, "stage0p5"),
        "AR-Sheet": ("wan14b-ref2va-stage1-oc-81f-r2", 900, "stage1"),
        "DMD-Sheet": ("wan14b-ref2va-stage2-oc-81f", 2000, "stage2"),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(source: Path, target: Path) -> None:
    if target.is_file() and target.stat().st_size == source.stat().st_size:
        return
    partial = target.with_name(target.name + ".partial")
    shutil.copyfile(source, partial)
    partial.replace(target)


def export_h3(source: Path, target: Path) -> list[Path]:
    """``adapter.pt`` plus a manifest and ``COMPLETE.json`` that list only it."""
    verified = verify_checkpoint(source)
    manifest = json.loads((source / "checkpoint-manifest.json").read_bytes())
    adapter = next(f for f in manifest["files"] if f["path"] == "adapter.pt")
    if manifest["metadata"].get("roles", {}).get("adapter.pt") != "trainable_model":
        raise SystemExit(f"{source}: adapter.pt is not the trainable model")
    _copy(source / "adapter.pt", target / "adapter.pt")
    manifest["files"] = [adapter]
    manifest["required_components"] = ["adapter.pt"]
    manifest["metadata"] = {**manifest["metadata"], "roles": {"adapter.pt": "trainable_model"},
                            "release": "GameWM: live LoRA weights only; optimizer, EMA and resume state removed"}
    body = canonical_json_bytes(manifest)
    (target / "checkpoint-manifest.json").write_bytes(body)
    (target / "COMPLETE.json").write_bytes(canonical_json_bytes({
        "schema": "solarwm.checkpoint-complete.v2", "step": verified.step,
        "manifest_digest": hashlib.blake2s(body).hexdigest()}))
    if verify_checkpoint(target).step != verified.step:
        raise SystemExit(f"{target}: exported checkpoint does not verify")
    return [target / "adapter.pt"]


def export_wan(source: Path, target: Path) -> list[Path]:
    if not (source / "COMPLETE.json").is_file():
        raise SystemExit(f"{source}: checkpoint is not complete")
    _copy(source / "student.safetensors", target / "student.safetensors")
    return [target / "student.safetensors"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h3-runs", required=True)
    ap.add_argument("--wan-runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    out = Path(args.out)
    for model, variants in RELEASE.items():
        if args.only and model not in args.only:
            continue
        for variant, (run, step, stage) in variants.items():
            target = out / model / variant
            target.mkdir(parents=True, exist_ok=True)
            if model == "GameWM-H3-33B":
                files = export_h3(Path(args.h3_runs) / run / f"checkpoint_model_{step:06d}", target)
            else:
                files = export_wan(Path(args.wan_runs) / run / f"checkpoint-{step:06d}", target)
            record = {"model": model, "variant": variant, "stage": stage, "step": step, "source_run": run,
                      "base_model": BASES[model], "weight_source": "live",
                      "files": {f.name: {"bytes": f.stat().st_size, "sha256": _sha256(f)} for f in files}}
            (target / "release.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(f"{model}/{variant}: {', '.join(f'{f.name} {f.stat().st_size / 1e9:.2f} GB' for f in files)}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
