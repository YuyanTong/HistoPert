"""Stage 2: align H&E embeddings with the molecular teacher representation."""

import copy
import json
import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .alignment import PatientAlignmentModel
from .data import load_patient_embeddings
from .io import load_checkpoint

# Basic utils


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_module(module):
    for parameter in module.parameters():
        parameter.requires_grad = False


def unfreeze_module(module):
    for parameter in module.parameters():
        parameter.requires_grad = True


def make_optimizer(model, lr, weight_decay=1e-4):
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    print(f"[optimizer] trainable parameters = {trainable_parameter_count:,}")

    if trainable_parameter_count == 0:
        raise RuntimeError(
            "No trainable parameters. Please check freeze/unfreeze logic."
        )

    return torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)


def save_checkpoint(path, model, optimizer, epoch, metrics, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
            if optimizer is not None
            else None,
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        },
        path,
    )
    print(f"[save_checkpoint] {path}")


# Load Stage1 encoder


def load_stage1_encoder(config):
    encoder_path = config["stage1_cancer_encoder_path"]

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Stage1 cancer encoder not found: {encoder_path}")

    with open(encoder_path, "r", encoding="utf-8") as f:
        encoder = json.load(f)

    encoder = {str(k): int(v) for k, v in encoder.items()}

    print("\nLoad Stage1 cancer encoder")
    print(f"[stage1 encoder path] {encoder_path}")
    print(f"[stage1 encoder n] {len(encoder)}")

    index_to_cancer = {v: k for k, v in encoder.items()}
    for i in sorted(index_to_cancer):
        print(f"  {i}: {index_to_cancer[i]}")

    return encoder, encoder_path


# Load main split


def load_main_split(patient_ids, split_path):
    """Load train/validation/test indices and patient IDs from the Stage 1 split."""
    patient_ids = list(patient_ids)

    if len(set(patient_ids)) != len(patient_ids):
        dup = [
            patient_id for patient_id, cnt in Counter(patient_ids).items() if cnt > 1
        ]
        raise RuntimeError(
            f"[main split] current patient_ids contains duplicates, examples={dup[:20]}"
        )

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"[main split] split json not found: {split_path}")

    print(f"\n[main split] loading: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        split_metadata = json.load(f)

    required_fields = [
        "main_train_idx",
        "main_val_idx",
        "main_test_idx",
        "main_train_patient_ids",
        "main_val_patient_ids",
        "main_test_patient_ids",
    ]

    missing_fields = [field for field in required_fields if field not in split_metadata]
    if missing_fields:
        raise RuntimeError(
            f"[main split] missing fields in split json: {missing_fields}"
        )

    split_patient_ids = {
        "main_train": list(split_metadata["main_train_patient_ids"]),
        "main_val": list(split_metadata["main_val_patient_ids"]),
        "main_test": list(split_metadata["main_test_patient_ids"]),
    }

    split_indices = {
        "main_train": np.array(split_metadata["main_train_idx"], dtype=int),
        "main_val": np.array(split_metadata["main_val_idx"], dtype=int),
        "main_test": np.array(split_metadata["main_test_idx"], dtype=int),
    }

    # Check overlap by patient id.
    split_sets = {k: set(v) for k, v in split_patient_ids.items()}

    overlaps = {
        "main_train/main_val": split_sets["main_train"] & split_sets["main_val"],
        "main_train/main_test": split_sets["main_train"] & split_sets["main_test"],
        "main_val/main_test": split_sets["main_val"] & split_sets["main_test"],
    }

    any_overlap = {k: v for k, v in overlaps.items() if len(v) > 0}
    if any_overlap:
        details = {k: list(v)[:20] for k, v in any_overlap.items()}
        raise RuntimeError(f"[main split] patient_id overlap across splits: {details}")

    # Check that split covers exactly current patients.
    all_split_ids = (
        split_sets["main_train"] | split_sets["main_val"] | split_sets["main_test"]
    )
    current_ids = set(patient_ids)

    missing_in_current = all_split_ids - current_ids
    extra_current = current_ids - all_split_ids

    if missing_in_current:
        raise RuntimeError(
            "[main split] split contains patient_id no longer in current patient_ids: "
            f"{list(missing_in_current)[:20]}"
        )

    if extra_current:
        raise RuntimeError(
            "[main split] current patient_ids contains patient_id not covered by split: "
            f"{list(extra_current)[:20]}"
        )

    # Check index-to-patient-id exact order.
    for name, expected_patient_ids in split_patient_ids.items():
        idx = split_indices[name]

        if np.any(idx < 0) or np.any(idx >= len(patient_ids)):
            raise RuntimeError(f"[main split] {name}_idx contains out-of-range indices")

        ids_from_idx = [patient_ids[i] for i in idx.tolist()]
        exact_order_match = ids_from_idx == expected_patient_ids
        same_set_match = set(ids_from_idx) == set(expected_patient_ids)

        print(
            f"[main split] {name}: "
            f"saved={len(expected_patient_ids)}, "
            f"idx={len(idx)}, "
            f"exact_order_match={exact_order_match}, "
            f"same_set_match={same_set_match}"
        )

        if not exact_order_match:
            print(f"[main split] {name}_idx/patient_id order mismatch")
            print(
                f"[main split] saved patient_id first20 = {expected_patient_ids[:20]}"
            )
            print(f"[main split] idx patient_id first20 = {ids_from_idx[:20]}")
            raise RuntimeError(
                f"[main split] {name}_idx does not match {name}_patient_ids"
            )

    print("\nMain split summary")
    print(f"main_train patients = {len(split_indices['main_train'])}")
    print(f"main_val patients = {len(split_indices['main_val'])}")
    print(f"main_test patients = {len(split_indices['main_test'])}")
    print(
        "[main split] main_test reserved for evaluation"
    )

    return (
        split_indices["main_train"],
        split_indices["main_val"],
        split_indices["main_test"],
        split_metadata,
    )


# Dataset builders


def build_stage2_dataset(patient_img, patient_gene, selected_patient_indices):
    idx = np.asarray([int(x) for x in selected_patient_indices], dtype=int)

    if len(set(idx.tolist())) != len(idx):
        raise RuntimeError("selected_patient_indices contains duplicated indices")

    return TensorDataset(
        patient_img[idx],
        patient_gene[idx],
    )


def build_training_resources(config):
    print("\nBuild Stage2 resources")

    # Load the Stage 1 encoder to recover the model's cancer count.

    stage1_cancer_encoder, encoder_path = load_stage1_encoder(config)

    # Load paired H&E and molecular embeddings.

    patient_img, patient_gene, patient_ids, patient_cancers_raw = (
        load_patient_embeddings(
            features_output_dir=config["features_output_dir"],
            txt_file_path=config["txt_file_path"],
            patient_csv_path=config["patient_csv_path"],
            patient_uce_path=config["patient_uce_path"],
            id_mode="case",
            dedup_case=True,
        )
    )

    print("\nLoaded TCGA patient embeddings")
    print(f"[patient_n] {len(patient_ids)}")
    print(f"[patient_img] {tuple(patient_img.shape)}")
    print(f"[patient_gene] {tuple(patient_gene.shape)}")

    print("\n[Raw TCGA project counts]")
    for k, v in sorted(
        Counter([str(x) for x in patient_cancers_raw]).items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"  {k}: {v}")

    # Load main split from Stage1.

    train_idx, val_idx, main_test_idx, split_payload = load_main_split(
        patient_ids=list(patient_ids),
        split_path=config["main_split_path"],
    )

    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    main_test_idx = np.asarray(main_test_idx, dtype=int)

    stage2_train_dataset = build_stage2_dataset(
        patient_img=patient_img,
        patient_gene=patient_gene,
        selected_patient_indices=train_idx,
    )

    stage2_val_dataset = build_stage2_dataset(
        patient_img=patient_img,
        patient_gene=patient_gene,
        selected_patient_indices=val_idx,
    )

    # Save diagnostics.
    os.makedirs(config["save_dir"], exist_ok=True)

    with open(
        os.path.join(config["save_dir"], "stage1_cancer_encoder_used_for_stage2.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(stage1_cancer_encoder, f, ensure_ascii=False, indent=2)

    with open(
        os.path.join(config["save_dir"], "stage2_split_used.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(split_payload, f, ensure_ascii=False, indent=2)

    with open(
        os.path.join(config["save_dir"], "stage2_patient_mapping_summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "stage1_encoder_path": encoder_path,
                "main_split_path": config["main_split_path"],
                "patient_n": int(len(patient_ids)),
                "patient_img_shape": list(patient_img.shape),
                "patient_gene_shape": list(patient_gene.shape),
                "main_train_n": int(len(train_idx)),
                "main_val_n": int(len(val_idx)),
                "main_test_n": int(len(main_test_idx)),
                "raw_tcga_project_counts": dict(
                    Counter([str(x) for x in patient_cancers_raw])
                ),
                "note": "Stage 2 aligns patient_img with patient_gene.",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "stage2_train_dataset": stage2_train_dataset,
        "stage2_val_dataset": stage2_val_dataset,
        "stage1_cancer_encoder": stage1_cancer_encoder,
        "patient_img": patient_img,
        "patient_gene": patient_gene,
        "patient_ids": patient_ids,
        "patient_cancers_raw": patient_cancers_raw,
        "main_train_idx": train_idx,
        "main_val_idx": val_idx,
        "main_test_idx": main_test_idx,
        "split_payload": split_payload,
    }


# Load Stage1 checkpoint


def resume_stage1_model_if_needed(model, config, device):
    checkpoint_path = config.get("resume_stage1_ckpt", None)

    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        raise ValueError("resume_stage1_ckpt is not set")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Stage1 checkpoint not found: {checkpoint_path}")

    print("\nLoad Stage1 checkpoint")
    print(f"[resume stage1] ckpt_path = {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    incompatible = model.load_state_dict(
        state_dict,
        strict=False,
    )

    source_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    source_metrics = checkpoint.get("metrics") if isinstance(checkpoint, dict) else None

    print(f"[resume stage1] loaded checkpoint from: {checkpoint_path}")
    print(f"[resume stage1] checkpoint epoch = {source_epoch}")
    print(f"[resume stage1] metrics = {source_metrics}")
    print(f"[resume stage1] missing keys n = {len(incompatible.missing_keys)}")
    print(f"[resume stage1] unexpected keys n = {len(incompatible.unexpected_keys)}")

    if len(incompatible.missing_keys) > 0:
        print("[resume stage1] first missing keys:")
        for k in incompatible.missing_keys[:30]:
            print(f"  {k}")

    if len(incompatible.unexpected_keys) > 0:
        print("[resume stage1] first unexpected keys:")
        for k in incompatible.unexpected_keys[:30]:
            print(f"  {k}")

    model = model.to(device)
    return model, checkpoint


# Stage2 training


def train_stage2(model, train_dataset, val_dataset, config, device):
    print("\nStage 2: image -> aligned gene/cell embedding")
    print(f"len(train_dataset) = {len(train_dataset)}")
    print(f"len(val_dataset) = {len(val_dataset)}")

    sample = train_dataset[0]
    print(f"train sample img shape = {tuple(sample[0].shape)}")
    print(f"train sample gene shape = {tuple(sample[1].shape)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["stage2_batch_size"],
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["stage2_batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # Freeze all parameters.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Only train image alignment head.
    unfreeze_module(model.img_align_head)

    optimizer = make_optimizer(
        model,
        lr=config["stage2_lr"],
        weight_decay=1e-4,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None
    patience_counter = 0

    patience = config.get("stage2_early_stop_patience", 30)
    min_delta = config.get("stage2_min_delta", 1e-4)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_img_align": [],
        "train_img_cos": [],
        "val_loss": [],
        "val_img_align": [],
        "val_img_cos": [],
    }

    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    print("\nStage2 effective settings")
    print(f"[Stage2] batch_size = {config['stage2_batch_size']}")
    print(f"[Stage2] lr = {config['stage2_lr']}")
    print(f"[Stage2] epochs = {config['stage2_epochs']}")
    print(f"[Stage2] patience = {patience}")
    print(f"[Stage2] min_delta = {min_delta}")
    print(f"[Stage2] lambda_img_align = {config['lambda_img_align']}")
    print(f"[Stage2] lambda_img_cos = {config['lambda_img_cos']}")

    for epoch in range(config["stage2_epochs"]):
        # Train

        model.train()

        meters = {
            "loss": 0.0,
            "loss_img_align": 0.0,
            "loss_img_cos": 0.0,
        }
        n_batches = 0

        for image_embeddings, gene_embeddings in tqdm(
            train_loader,
            desc=f"Stage2 Train {epoch + 1}",
            dynamic_ncols=True,
        ):
            image_embeddings = image_embeddings.to(device, non_blocking=True)
            gene_embeddings = gene_embeddings.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            losses = model.stage2_losses(
                img=image_embeddings,
                gene=gene_embeddings,
                lambda_img_align=config["lambda_img_align"],
                lambda_img_cos=config["lambda_img_cos"],
            )

            loss = losses["loss"]
            loss.backward()
            optimizer.step()

            meters["loss"] += float(losses["loss"].item())
            meters["loss_img_align"] += float(losses["loss_img_align"].item())
            meters["loss_img_cos"] += float(losses["loss_img_cos"].item())
            n_batches += 1

        avg_train_loss = meters["loss"] / max(n_batches, 1)
        avg_train_img_align = meters["loss_img_align"] / max(n_batches, 1)
        avg_train_img_cos = meters["loss_img_cos"] / max(n_batches, 1)

        print(
            f"[Stage2 Train {epoch + 1}] "
            f"loss={avg_train_loss:.6f} "
            f"img_align={avg_train_img_align:.6f} "
            f"img_cos={avg_train_img_cos:.6f}"
        )

        # Val

        model.eval()

        val_meters = {
            "loss": 0.0,
            "loss_img_align": 0.0,
            "loss_img_cos": 0.0,
        }
        val_batches = 0

        with torch.no_grad():
            for image_embeddings, gene_embeddings in tqdm(
                val_loader,
                desc=f"Stage2 Val {epoch + 1}",
                dynamic_ncols=True,
            ):
                image_embeddings = image_embeddings.to(device, non_blocking=True)
                gene_embeddings = gene_embeddings.to(device, non_blocking=True)

                losses = model.stage2_losses(
                    img=image_embeddings,
                    gene=gene_embeddings,
                    lambda_img_align=config["lambda_img_align"],
                    lambda_img_cos=config["lambda_img_cos"],
                )

                val_meters["loss"] += float(losses["loss"].item())
                val_meters["loss_img_align"] += float(losses["loss_img_align"].item())
                val_meters["loss_img_cos"] += float(losses["loss_img_cos"].item())
                val_batches += 1

        avg_val_loss = val_meters["loss"] / max(val_batches, 1)
        avg_val_img_align = val_meters["loss_img_align"] / max(val_batches, 1)
        avg_val_img_cos = val_meters["loss_img_cos"] / max(val_batches, 1)

        print(
            f"[Stage2 Val {epoch + 1}] "
            f"loss={avg_val_loss:.6f} "
            f"img_align={avg_val_img_align:.6f} "
            f"img_cos={avg_val_img_cos:.6f}"
        )

        # Save history

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["train_img_align"].append(avg_train_img_align)
        history["train_img_cos"].append(avg_train_img_cos)
        history["val_loss"].append(avg_val_loss)
        history["val_img_align"].append(avg_val_img_align)
        history["val_img_cos"].append(avg_val_img_cos)

        with open(
            os.path.join(save_dir, "stage2_history.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        pd.DataFrame(history).to_csv(
            os.path.join(save_dir, "stage2_history.csv"),
            index=False,
        )

        # Save last

        save_checkpoint(
            os.path.join(save_dir, "stage2_last.pt"),
            model,
            optimizer=optimizer,
            epoch=epoch + 1,
            metrics={
                "stage": "stage2_v21_main_split_75_10_15",
                "stage2_val_loss": avg_val_loss,
                "stage2_val_img_align": avg_val_img_align,
                "stage2_val_img_cos": avg_val_img_cos,
            },
            config=config,
        )

        # Save best

        improved = (best_val_loss - avg_val_loss) > min_delta

        if improved:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_state_dict = copy.deepcopy(model.state_dict())

            save_checkpoint(
                os.path.join(save_dir, "stage2_model.pt"),
                model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics={
                    "stage": "stage2_v21_main_split_75_10_15",
                    "stage2_val_loss": avg_val_loss,
                    "stage2_val_img_align": avg_val_img_align,
                    "stage2_val_img_cos": avg_val_img_cos,
                },
                config=config,
            )

            print(
                f"[Stage2] new best val loss = {best_val_loss:.6f} at epoch {best_epoch}"
            )

        else:
            patience_counter += 1
            print(
                f"[Stage2] no improvement. "
                f"best_val_loss={best_val_loss:.6f} at epoch {best_epoch}. "
                f"patience {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print(
                f"[Stage2] Early stopping triggered at epoch {epoch + 1}, best epoch={best_epoch}"
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict, strict=False)
        print(
            f"[Stage2] loaded best model from epoch {best_epoch} "
            f"with val_loss={best_val_loss:.6f}"
        )

    plot_stage2_history(history, save_dir)

    print("\nFinished Stage2 training")
    print(f"[best epoch] {best_epoch}")
    print(f"[best val loss] {best_val_loss:.6f}")
    print(f"[best checkpoint] {os.path.join(save_dir, 'stage2_model.pt')}")
    print(f"[last checkpoint] {os.path.join(save_dir, 'stage2_last.pt')}")

    return history


def plot_stage2_history(history, save_dir):
    if len(history["epoch"]) == 0:
        print("[plot_stage2_history] empty history, skip")
        return

    epochs = history["epoch"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Stage2 Total Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "stage2_total_loss.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_img_align"], label="train_img_align")
    plt.plot(epochs, history["val_img_align"], label="val_img_align")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Stage2 Image Alignment Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "stage2_img_align_loss.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_img_cos"], label="train_img_cos")
    plt.plot(epochs, history["val_img_cos"], label="val_img_cos")
    plt.xlabel("Epoch")
    plt.ylabel("Cosine Loss")
    plt.title("Stage2 Image Cosine Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "stage2_img_cos_loss.png"), dpi=200)
    plt.close()

    print(f"[plot_stage2_history] saved plots to {save_dir}")


# Main


def train_patient_stage2_only(config=None):
    """Run image alignment using the trained molecular alignment checkpoint."""
    if config is None:
        raise ValueError("Use python -m histopert with an explicit configuration")

    print("\nStage 2: histology alignment")
    print(f"save_dir = {config['save_dir']}")
    print(f"stage1_dir = {config['stage1_dir']}")
    print(f"resume_stage1_ckpt = {config['resume_stage1_ckpt']}")
    print(f"main_split_path = {config['main_split_path']}")
    print(f"stage1_cancer_encoder_path = {config['stage1_cancer_encoder_path']}")
    print(f"device = {config['device']}")
    print(f"patient_csv_path = {config['patient_csv_path']}")
    print(f"patient_uce_path = {config['patient_uce_path']}")

    os.makedirs(config["save_dir"], exist_ok=True)

    with open(
        os.path.join(config["save_dir"], "config_stage2_from_stage1_75_10_15.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    set_seed(config["seed"])
    device = torch.device(config["device"])

    resources = build_training_resources(config)

    stage2_train_dataset = resources["stage2_train_dataset"]
    stage2_val_dataset = resources["stage2_val_dataset"]
    stage1_cancer_encoder = resources["stage1_cancer_encoder"]

    sample = stage2_train_dataset[0]

    img_dim = sample[0].numel()
    gene_dim = sample[1].numel()

    # Stage2 does not use clinical features.
    clin_dim = int(config["clin_dim"])

    # Stage1 cell space dim is 1280.
    # In current data, gene_dim is also 1280.
    cell_dim = gene_dim

    num_cancers = len(stage1_cancer_encoder)

    print("\nModel dim check")
    print(f"img_dim = {img_dim}")
    print(f"gene_dim = {gene_dim}")
    print(f"clin_dim = {clin_dim}")
    print(f"cell_dim = {cell_dim}")
    print(f"num_cancers = {num_cancers}")

    if cell_dim != 1280:
        raise ValueError(f"Expected cell_dim=1280, got {cell_dim}")

    model = PatientAlignmentModel(
        img_dim=img_dim,
        gene_dim=gene_dim,
        cell_dim=cell_dim,
        drug_dim=1,
        output_dim=1,
        clin_dim=clin_dim,
        num_cancers=num_cancers,
        hidden_dim=config["hidden_dim"],
        cancer_emb_dim=config["cancer_emb_dim"],
        dropout=config["dropout"],
        use_clinical=config["use_clinical"],
        unicure_model=None,
    ).to(device)

    model, stage1_ckpt = resume_stage1_model_if_needed(
        model=model,
        config=config,
        device=device,
    )

    history = train_stage2(
        model=model,
        train_dataset=stage2_train_dataset,
        val_dataset=stage2_val_dataset,
        config=config,
        device=device,
    )

    return history
