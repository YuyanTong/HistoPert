"""Resolve portable data paths and stage-specific training settings."""

import json
from pathlib import Path

from .io import output_path

DATA_KEYS = {
    "features_output_dir",
    "txt_file_path",
    "patient_csv_path",
    "patient_uce_path",
    "patient_text_embedding_path",
    "patient_text_embedding_index_path",
    "cell_emb_path",
    "cell_cancer_path",
    "cell_drug_dose_expr_parquet",
    "cell_control_parquet",
    "drug_embedding_path",
    "split_path",
}


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    unknown = set(config) - {
        "data",
        "run_dir",
        "seed",
        "device",
        "model",
        "stage1",
        "stage2",
        "stage3",
        "description",
    }
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    if set(config.get("data", {})) != DATA_KEYS:
        raise ValueError(f"Data keys must be exactly {sorted(DATA_KEYS)}")
    config["data"] = {
        k: str((path.parent / v).resolve()) for k, v in config["data"].items()
    }
    config["run_dir"] = str(output_path(path.parent / config["run_dir"]))
    config.setdefault("seed", 11)
    config.setdefault("device", "cpu")
    return config


def stage_config(config, stage, *, mode="train", resume=False):
    defaults = json.loads(Path(__file__).with_name("defaults.json").read_text())
    settings = defaults[f"stage{stage}"]
    model = config.get("model", {})
    model_keys = {
        "hidden_dim",
        "cancer_emb_dim",
        "dropout",
        "patient_text_dim",
        "stage3_output_dim",
        "perturb_hidden_dim",
    }
    if set(model) - model_keys:
        raise ValueError(f"Unknown model fields: {sorted(set(model) - model_keys)}")
    overrides = config.get(f"stage{stage}", {})
    # Paths, initialization and protocol flags are owned by the release.
    allowed = {
        k for k in settings if k.startswith(f"stage{stage}_") or k.startswith("lambda_")
    }
    if set(overrides) - allowed:
        raise ValueError(
            f"Unsupported Stage {stage} settings: {sorted(set(overrides) - allowed)}"
        )
    settings.update(overrides)
    settings.update(model)
    settings.update(config["data"])
    run_directory = output_path(config["run_dir"])
    stage_directory = run_directory / f"stage{stage}"
    stage1_directory, stage2_directory = (
        run_directory / "stage1",
        run_directory / "stage2",
    )
    settings.update(
        {
            "seed": config["seed"],
            "device": config["device"],
            "save_dir": str(stage_directory),
            "experiment_root": str(stage_directory),
            "mode_dir": str(stage_directory / "splits"),
            "split_filename": "patient_split_main_train_val_test.json",
            "stage1_dir": str(stage1_directory),
            "stage2_dir": str(stage2_directory),
            "main_split_path": str(stage1_directory / "main_split_used.json"),
            "stage1_cancer_encoder_path": str(
                stage1_directory / "stage1_cancer_encoder.json"
            ),
            "resume_stage1_ckpt": str(stage1_directory / "stage1_alignment_model.pt"),
            "resume_stage2_ckpt": str(stage2_directory / "stage2_model.pt"),
            "resume_stage1_ckpt_path": None,
            "resume_from_stage1_ckpt": False,
            "old_stage3_ckpt_path": None,
            "load_old_stage3_partial": False,
            "resume_stage3_training": resume,
            "resume_stage3_ckpt_type": "last",
            "resume_stage3_last_ckpt": str(
                stage_directory / "stage3_text_fusion_unicure_last.pt"
            ),
            "resume_stage3_best_ckpt": str(
                stage_directory / "stage3_text_fusion_unicure_best.pt"
            ),
            "execution_mode": mode,
            "preflight_only": mode == "preflight",
            "final_test_mode": False,
            "build_main_test_cache": False,
            "run_stage3": mode == "train",
            "require_preflight_manifest": mode != "preflight",
            "force_rebuild_condition_store": False,
            "force_rebuild_record_caches": False,
            "use_clinical": False,
            "clin_dim": 0,
        }
    )
    return settings
