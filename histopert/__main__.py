"""CLI: python -m histopert --help."""

import argparse
import json
import os
import shutil
from pathlib import Path

# Set before any pyplot import. No plotting windows during training.
os.environ.setdefault("MPLBACKEND", "Agg")

from .config import load_config, stage_config
from .io import output_path, write_json


def run_stage(config, stage, resume=False):
    settings = stage_config(config, stage, resume=resume)
    stage_directory = output_path(settings["save_dir"])
    if stage != 3 and resume:
        raise ValueError("--resume currently applies only to Stage 3")
    if not resume and stage_directory.exists() and any(stage_directory.glob("*.pt")):
        raise FileExistsError(
            f"Run already contains checkpoints: {stage_directory}. Use a new run_dir."
        )
    stage_directory.mkdir(parents=True, exist_ok=True)
    if stage == 1:
        from .stage1 import train_patient_stage1_only

        split_destination = Path(settings["mode_dir"]) / settings["split_filename"]
        split_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config["data"]["split_path"], split_destination)
        return train_patient_stage1_only(settings)
    if stage == 2:
        from .stage2 import train_patient_stage2_only

        return train_patient_stage2_only(settings)
    from . import global_fallback, stage3

    global_fallback.RESULT_DIR = stage_directory
    global_fallback._RAW_FALLBACK_AUDIT = (
        stage_directory / "base_proxy_global_fallback_audit_raw.csv"
    )
    global_fallback.install_extension()
    if not resume:
        stage3.train_patient_model_text_fusion(
            stage_config(config, 3, mode="preflight")
        )
    result = stage3.train_patient_model_text_fusion(settings)
    global_fallback.finalize_fallback_audit()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="TCGA Stage 1/2/3 from precomputed features"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "train", "stage1", "stage2", "stage3"):
        command_parser = sub.add_parser(name)
        command_parser.add_argument("--config", required=True)
        command_parser.add_argument(
            "--device", help="cpu or cuda:N; defaults to the JSON setting"
        )
        if name == "stage3":
            command_parser.add_argument(
                "--resume",
                action="store_true",
                help="Resume this run's last Stage 3 checkpoint",
            )
    command_parser = sub.add_parser("predict")
    for flag in (
        "checkpoint",
        "pairs",
        "he",
        "context",
        "context-index",
        "drugs",
        "output",
    ):
        command_parser.add_argument("--" + flag, required=True)
    command_parser.add_argument("--device", default="cpu")
    command_parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.command == "predict":
        from .predict import predict

        result = predict(
            args.checkpoint,
            args.pairs,
            args.he,
            args.context,
            args.context_index,
            args.drugs,
            args.output,
            args.device,
            args.batch_size,
        )
        print(json.dumps(result, indent=2))
        return
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    if args.command in ("validate", "train"):
        from .validate import validate_inputs

        result = validate_inputs(config)
        print(json.dumps(result, indent=2))
        run_dir = output_path(config["run_dir"])
        if (
            args.command == "train"
            and run_dir.exists()
            and any(run_dir.glob("stage*/*.pt"))
        ):
            raise FileExistsError(
                "Training outputs already exist; choose a new run_dir"
            )
        write_json(run_dir / "input_validation.json", result)
        if args.command == "validate":
            return
    stages = (1, 2, 3) if args.command == "train" else (int(args.command[-1]),)
    for stage in stages:
        run_stage(config, stage, getattr(args, "resume", False))


if __name__ == "__main__":
    main()
