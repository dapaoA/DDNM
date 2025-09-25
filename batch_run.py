import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import yaml


def ensure_symlink(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if target_path.is_symlink() or target_path.exists():
            # Replace only if incorrect
            try:
                current = target_path.resolve(strict=False)
            except Exception:
                current = None
            if current != source_path:
                if target_path.exists() or target_path.is_symlink():
                    if target_path.is_dir() and not target_path.is_symlink():
                        # Avoid deleting non-symlink directories
                        raise RuntimeError(f"Refusing to replace existing non-symlink directory: {target_path}")
                    target_path.unlink()
                target_path.symlink_to(source_path)
        else:
            target_path.symlink_to(source_path)
    except OSError as exc:
        print(f"[warn] Failed to create symlink {target_path} -> {source_path}: {exc}")


def write_overridden_config(
    base_config_path: Path,
    override_steps: int,
    override_subset_1k: Optional[bool],
    dataset_name_override: Optional[str],
    out_dir: Path,
) -> Path:
    with base_config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    # Override sampling steps
    if "time_travel" not in cfg:
        cfg["time_travel"] = {}
    cfg["time_travel"]["T_sampling"] = int(override_steps)

    # Also override diffusion steps if present (user prefers this control)
    if "diffusion" not in cfg:
        cfg["diffusion"] = {}
    cfg["diffusion"]["num_diffusion_timesteps"] = int(override_steps)

    # Optionally override ImageNet subset flag
    if override_subset_1k is not None and "data" in cfg:
        cfg["data"]["subset_1k"] = bool(override_subset_1k)

    # Optionally override dataset name to use folder-based loader (e.g., FFHQ branch)
    if dataset_name_override is not None and "data" in cfg:
        cfg["data"]["dataset"] = dataset_name_override

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base_config_path.stem}_T{override_steps}.yml"
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def run_command(command: List[str], cwd: Path) -> int:
    print("[run]", shlex.join(command))
    process = subprocess.Popen(command, cwd=cwd, stdout=sys.stdout, stderr=sys.stderr)
    return process.wait()


def maybe_make_inpaint_mask(repo_root: Path) -> None:
    mask_path = repo_root / "exp/inp_masks/mask.npy"
    if mask_path.exists():
        return
    (repo_root / "exp/inp_masks").mkdir(parents=True, exist_ok=True)
    import numpy as np

    height = width = 256
    mask = np.ones((height, width), dtype=np.uint8)
    hole_size = 96
    y0 = (height - hole_size) // 2
    x0 = (width - hole_size) // 2
    mask[y0 : y0 + hole_size, x0 : x0 + hole_size] = 0
    np.save(mask_path, mask)
    print(f"[info] Saved default inpainting mask at {mask_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch runner for DDNM")
    parser.add_argument("--eta", type=float, default=0.85, help="DDIM eta")
    parser.add_argument(
        "--simplified", action="store_true", default=True, help="Use simplified DDNM"
    )
    parser.add_argument(
        "--sigma_y", type=float, default=0.0, help="Observation noise level"
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Print commands without executing"
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent

    # User-editable sections
    # Models to run. Keys: 'imagenet', 'ffhq'
    models: List[str] = [
        "imagenet",
        # "ffhq",  # FFHQ 256x256 (OpenAI UNet architecture)
    ]

    # Tasks to run: align with README ImageNet best settings
    tasks: List[str] = [
        # "deblur_gauss",
        # # "colorization",
        # # "inpainting",
        # # Extra custom tasks
        # "inpainting_box",
        "inpainting_rand",

        # "sr_bicubic",         # 4x
        # "sr_bicubic_x8",      # 8x alias
    ]

    # Sampling steps to sweep
    steps_list: List[int] = [1000, 100, 20]
    # steps_list: List[int] = [20]
    # Extra per-task arguments (README defaults)
    task_extra_args: Dict[str, Dict[str, object]] = {  
        # "deblur_gauss": {"--sigma_y": 0.05},
        # # "colorization": {"--sigma_y": 0.05},
        # # "inpainting": {"--sigma_y": 0.05},
        # # Optional custom tasks:
        # "inpainting_box": {"--mask_len_range": [128, 129], "--sigma_y": 0.05},
        "inpainting_rand": {"--mask_prob_range": [0.9, 0.95], "--sigma_y": 0.05},

        # "sr_bicubic": {"--deg_scale": 4, "--sigma_y": 0.05},
        # "sr_bicubic_x8": {"--deg_scale": 8, "--sigma_y": 0.05},
    }

    # Dataset mapping for each model (relative to exp/datasets). Adjust if needed.
    model_to_path_y: Dict[str, str] = {
        "imagenet": "imagenet",
        "ffhq": "ffhq",
    }

    # Base config file per model
    model_to_config: Dict[str, str] = {
        "imagenet": "configs/imagenet_256.yml",
        "ffhq": "configs/ffhq_256.yml",
    }

    # Checkpoint sources (edit paths if your files live elsewhere)
    model_ckpt_sources: Dict[str, Path] = {
        "imagenet": Path("/workspace/InverseBench/checkpoints/imagenet256.pt"),
        "ffhq": Path("/workspace/InverseBench/checkpoints/ffhq256.pt"),
    }

    # Checkpoint targets expected by DDNM
    model_ckpt_targets: Dict[str, Path] = {
        "imagenet": repo_root / "exp/logs/imagenet/256x256_diffusion_uncond.pt",
        # For FFHQ, we route to the OpenAI ImageNet UNet path, since the model
        # uses the same structure as ImageNet 256x256 uncond checkpoint.
        "ffhq": repo_root / "exp/logs/imagenet/256x256_diffusion_uncond.pt",
    }

    # Ensure checkpoints are wired
    for model_name in models:
        src = model_ckpt_sources.get(model_name)
        tgt = model_ckpt_targets.get(model_name)
        if src is None or tgt is None:
            print(f"[warn] Missing checkpoint mapping for model {model_name}, skipping wire step")
            continue
        if not src.exists():
            print(f"[warn] Checkpoint source does not exist: {src} (model={model_name})")
        ensure_symlink(src, tgt)

    # Ensure datasets exist (user may have symlinked already)
    for model_name in models:
        path_y = model_to_path_y[model_name]
        ds_dir = repo_root / "exp/datasets" / path_y
        if not ds_dir.exists():
            print(f"[warn] Dataset directory not found: {ds_dir}. Please create or symlink your data.")

    # Create default inpainting mask if needed
    if any(task == "inpainting" for task in tasks):
        maybe_make_inpaint_mask(repo_root)

    auto_cfg_dir = repo_root / "configs/auto"

    # Run sweeps
    for model_name in models:
        base_cfg_rel = model_to_config[model_name]
        base_cfg_path = repo_root / base_cfg_rel
        if not base_cfg_path.exists():
            print(f"[error] Base config not found: {base_cfg_path}. Skipping model {model_name}.")
            continue

        # Do not override subset flag by default; honor what's in the base config
        override_subset_1k = None

        for steps in steps_list:
            # For ImageNet and FFHQ models, use folder-based loader 'FFHQ' when running on FFHQ
            dataset_override = "FFHQ" if model_name in ("imagenet", "ffhq") else None

            overridden_cfg = write_overridden_config(
                base_config_path=base_cfg_path,
                override_steps=steps,
                override_subset_1k=override_subset_1k,
                dataset_name_override=dataset_override,
                out_dir=auto_cfg_dir,
            )

            for task_name in tasks:
                extra_args = task_extra_args.get(task_name, {})
                out_name = f"{model_name}_{task_name}_T{steps}"

                # Map custom inpainting tasks to base 'inpainting' and materialize mask.npy
                effective_task = task_name
                if task_name in ("inpainting_box", "inpainting_rand"):
                    # Load current config to get image size
                    with overridden_cfg.open("r") as f:
                        cfg_now = yaml.safe_load(f)
                    H = W = int(cfg_now.get("data", {}).get("image_size", 256))
                    (repo_root / "exp/inp_masks").mkdir(parents=True, exist_ok=True)
                    import numpy as np
                    if task_name == "inpainting_box":
                        min_len, max_len = [128, 129]
                        if isinstance(extra_args.get("--mask_len_range"), (list, tuple)) and len(extra_args["--mask_len_range"]) == 2:
                            min_len, max_len = extra_args["--mask_len_range"]
                        side = int(np.random.randint(int(min_len), int(max_len) + 1))
                        y0 = (H - side) // 2
                        x0 = (W - side) // 2
                        mask_np = np.ones((H, W), dtype=np.float32)
                        mask_np[y0:y0+side, x0:x0+side] = 0.0
                    else:  # inpainting_rand
                        pmin, pmax = [0.3, 0.7]
                        if isinstance(extra_args.get("--mask_prob_range"), (list, tuple)) and len(extra_args["--mask_prob_range"]) == 2:
                            pmin, pmax = extra_args["--mask_prob_range"]
                        keep_p = float(np.random.uniform(float(pmin), float(pmax)))
                        mask_np = (np.random.rand(H, W) < keep_p).astype(np.float32)
                    np.save(repo_root / "exp/inp_masks/mask.npy", mask_np)
                    effective_task = "inpainting"

                # Map SR alias to base task name
                if task_name == "sr_bicubic_x8":
                    effective_task = "sr_bicubic"

                # Filter out mask-only args; keep others like --sigma_y, --deg_scale
                filtered_extra_args = {k: v for k, v in extra_args.items() if not str(k).startswith("--mask_")}
                cmd: List[str] = [
                    sys.executable,
                    "main.py",
                    "--ni",
                    "--config",
                    # main.py joins "configs/" + args.config, so pass path relative to configs/
                    str(overridden_cfg.relative_to(repo_root / "configs")),
                    "--path_y",
                    model_to_path_y[model_name],
                    "--eta",
                    str(args.eta),
                    "--deg",
                    effective_task,
                    "--sigma_y",
                    str(args.sigma_y),
                    "-i",
                    out_name,
                ]

                # Use simplified path for all tasks
                cmd.insert(3, "--simplified")
                # Append extra args for the task
                for key, value in filtered_extra_args.items():
                    if isinstance(value, (list, tuple)):
                        cmd.append(str(key))
                        cmd.extend([str(v) for v in value])
                    else:
                        cmd.extend([str(key), str(value)])

                if args.dry_run:
                    print("[dry-run]", shlex.join(cmd))
                else:
                    ret = run_command(cmd, cwd=repo_root)
                    if ret != 0:
                        print(f"[error] Command failed (code={ret}). Skipping remainder of this combo: model={model_name}, task={task_name}, steps={steps}")
                        # Continue to next combo instead of aborting entire run

    print("[done] Batch run completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


