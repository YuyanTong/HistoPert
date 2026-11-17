"""Stage 1: align patient molecular embeddings with the cell reference space."""

import copy
import json
import os
import random
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler, TensorDataset
from tqdm import tqdm

from .alignment import PatientAlignmentModel
from .data import (
    build_shared_cancer_label_encoder,
    encode_cancer_labels,
    load_cell_cancer_map,
    load_patient_embeddings,
)
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


def resume_stage1_from_checkpoint(model, config, device):
    checkpoint_path = config["resume_stage1_ckpt_path"]

    print("\nResume Stage1 from checkpoint")
    print(f"ckpt_path = {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Stage1 checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    incompatible = model.load_state_dict(state_dict, strict=False)

    source_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    source_metrics = checkpoint.get("metrics") if isinstance(checkpoint, dict) else None

    print("loaded model weights")
    print(f"missing keys count = {len(incompatible.missing_keys)}")
    print(f"unexpected keys count = {len(incompatible.unexpected_keys)}")
    print(f"source epoch = {source_epoch}")
    print(f"source metrics = {source_metrics}")

    if len(incompatible.missing_keys) > 0:
        print("[resume] first missing keys:")
        for k in incompatible.missing_keys[:30]:
            print(f"  {k}")

    if len(incompatible.unexpected_keys) > 0:
        print("[resume] first unexpected keys:")
        for k in incompatible.unexpected_keys[:30]:
            print(f"  {k}")

    model = model.to(device)
    return model, checkpoint


# Cancer label normalization


def normalize_cell_cancer(x: str) -> str:
    """Normalize LINCS cancer labels for molecular alignment."""
    x = str(x).strip()

    cell_map = {
        "Unknown": "UNKNOWN",
        "Non-Cancerous": "OTHER",
        "Bladder Urothelial Carcinoma": "BLCA",
        "Invasive Breast Carcinoma": "BRCA",
        "Cervical Adenocarcinoma": "CESC",
        "Colorectal Adenocarcinoma": "COAD_READ",
        "Diffuse Glioma": "GBM_LGG",
        "Head and Neck Squamous Cell Carcinoma": "HNSC",
        "Hepatocellular Carcinoma": "LIHC",
        "Non-Small Cell Lung Cancer": "LUAD_LUSC",
        "Ovarian Epithelial Tumor": "OV",
        "Pancreatic Adenocarcinoma": "PAAD",
        "Prostate Adenocarcinoma": "PRAD",
        "Melanoma": "SKCM",
        "Esophagogastric Adenocarcinoma": "STAD_ESCA",
        "Endometrial Carcinoma": "UCEC_UCS",
        "Renal Cell Carcinoma": "KIDNEY",
        "Acute Myeloid Leukemia": "HEME",
        "B-Cell Acute Lymphoblastic Leukemia": "HEME",
        "Mature B-Cell Neoplasms": "HEME",
        "Myeloproliferative Neoplasms": "HEME",
        "T-Lymphoblastic Leukemia/Lymphoma": "HEME",
        "Ewing Sarcoma": "SARCOMA",
        "Osteosarcoma": "SARCOMA",
        "Rhabdoid Cancer": "SARCOMA",
        "Hepatoblastoma": "OTHER",
        "Neuroblastoma": "OTHER",
        "SMARCA4-deficient undifferentiated tumor": "OTHER",
    }

    return cell_map.get(x, "UNKNOWN")


def normalize_patient_cancer(x: str) -> str:
    """Map TCGA projects to reference cancer groups or TCGA_OTHER_UNMATCHED."""
    x = str(x).strip()

    patient_map = {
        "TCGA-ACC": "TCGA_OTHER_UNMATCHED",
        "TCGA-BLCA": "BLCA",
        "TCGA-BRCA": "BRCA",
        "TCGA-CESC": "CESC",
        "TCGA-CHOL": "TCGA_OTHER_UNMATCHED",
        "TCGA-COAD": "COAD_READ",
        "TCGA-READ": "COAD_READ",
        "TCGA-DLBC": "HEME",
        "TCGA-ESCA": "STAD_ESCA",
        "TCGA-STAD": "STAD_ESCA",
        "TCGA-GBM": "GBM_LGG",
        "TCGA-LGG": "GBM_LGG",
        "TCGA-HNSC": "HNSC",
        "TCGA-KICH": "KIDNEY",
        "TCGA-KIRC": "KIDNEY",
        "TCGA-KIRP": "KIDNEY",
        "TCGA-LIHC": "LIHC",
        "TCGA-LUAD": "LUAD_LUSC",
        "TCGA-LUSC": "LUAD_LUSC",
        "TCGA-OV": "OV",
        "TCGA-PAAD": "PAAD",
        "TCGA-PRAD": "PRAD",
        "TCGA-SARC": "SARCOMA",
        "TCGA-SKCM": "SKCM",
        "TCGA-UCEC": "UCEC_UCS",
        "TCGA-UCS": "UCEC_UCS",
        # No reliable LINCS same-cancer target.
        "TCGA-MESO": "TCGA_OTHER_UNMATCHED",
        "TCGA-PCPG": "TCGA_OTHER_UNMATCHED",
        "TCGA-TGCT": "TCGA_OTHER_UNMATCHED",
        "TCGA-THCA": "TCGA_OTHER_UNMATCHED",
        "TCGA-THYM": "TCGA_OTHER_UNMATCHED",
        "TCGA-UVM": "TCGA_OTHER_UNMATCHED",
    }

    return patient_map.get(x, "UNKNOWN")


# Split utilities


def _stratified_sample(
    indices: np.ndarray, cancers: np.ndarray, n_pick: int, rng: np.random.RandomState
) -> np.ndarray:
    indices = np.array(indices, dtype=int)

    if n_pick <= 0 or len(indices) == 0:
        return np.array([], dtype=int)

    if n_pick >= len(indices):
        out = indices.copy()
        rng.shuffle(out)
        return out

    cancers_sub = cancers[indices]
    uniq, counts = np.unique(cancers_sub, return_counts=True)
    props = counts / counts.sum()

    alloc = np.floor(props * n_pick).astype(int)
    remainder = int(n_pick - alloc.sum())

    frac = props * n_pick - alloc
    order = np.argsort(-frac)

    for k in range(remainder):
        alloc[order[k % len(order)]] += 1

    picked = []

    for c, k in zip(uniq, alloc):
        idx_c = indices[cancers_sub == c]
        rng.shuffle(idx_c)
        picked.append(idx_c[:k])

    picked = np.concatenate(picked) if len(picked) else np.array([], dtype=int)
    rng.shuffle(picked)

    return picked.astype(int)


def make_or_load_main_split_train_val_test(
    patient_ids,
    patient_cancers_stage1,
    split_path,
    seed=42,
    train_ratio=0.70,
    val_ratio=0.10,
    test_ratio=0.20,
):
    """Create or load patient splits stratified by Stage 1 cancer label.

    Return main_train, main_val and main_test. Stage 1 trains on main_train
    and uses main_val for early stopping.
    """
    patient_ids = list(patient_ids)

    if len(set(patient_ids)) != len(patient_ids):
        dup = [
            patient_id for patient_id, cnt in Counter(patient_ids).items() if cnt > 1
        ]
        raise RuntimeError(
            f"[main split] current patient_ids contains duplicates, examples={dup[:20]}"
        )

    ratios_sum = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if abs(ratios_sum - 1.0) > 1e-6:
        raise ValueError(
            f"[main split] train/val/test ratios must sum to 1.0, got {ratios_sum}"
        )

    if os.path.exists(split_path):
        print(f"[main split] loading existing split: {split_path}")
        with open(split_path, "r", encoding="utf-8") as f:
            split_metadata = json.load(f)

        required_fields = [
            "main_train_idx",
            "main_val_idx",
            "main_test_idx",
            "main_train_patient_ids",
            "main_val_patient_ids",
            "main_test_patient_ids",
            "seed",
            "split_note",
        ]
        missing_fields = [
            field for field in required_fields if field not in split_metadata
        ]
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

        split_sets = {k: set(v) for k, v in split_patient_ids.items()}
        overlaps = {
            "main_train/main_val": split_sets["main_train"] & split_sets["main_val"],
            "main_train/main_test": split_sets["main_train"] & split_sets["main_test"],
            "main_val/main_test": split_sets["main_val"] & split_sets["main_test"],
        }

        any_overlap = {k: v for k, v in overlaps.items() if len(v) > 0}
        if any_overlap:
            details = {k: list(v)[:20] for k, v in any_overlap.items()}
            raise RuntimeError(
                f"[main split] patient_id overlap across splits: {details}"
            )

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
                "[main split] current patient_ids contains patient_id not covered by existing split: "
                f"{list(extra_current)[:20]}"
            )

        for name, saved_patient_ids in split_patient_ids.items():
            saved_idx = split_indices[name]

            if np.any(saved_idx < 0) or np.any(saved_idx >= len(patient_ids)):
                raise RuntimeError(
                    f"[main split] {name}_idx contains out-of-range indices"
                )

            ids_from_saved_idx = [patient_ids[i] for i in saved_idx.tolist()]
            exact_order_match = ids_from_saved_idx == saved_patient_ids
            same_set_match = set(ids_from_saved_idx) == set(saved_patient_ids)

            print(
                f"[main split] {name}: "
                f"saved={len(saved_patient_ids)}, "
                f"idx={len(saved_idx)}, "
                f"exact_order_match={exact_order_match}, "
                f"same_set_match={same_set_match}"
            )

            if not exact_order_match:
                print(f"[main split] {name}_idx/patient_id order mismatch")
                print(
                    f"[main split] saved patient_id first20 = {saved_patient_ids[:20]}"
                )
                print(
                    f"[main split] idx patient_id first20 = {ids_from_saved_idx[:20]}"
                )
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

    rng = np.random.RandomState(seed)
    N = len(patient_ids)

    cancers_arr = np.asarray(patient_cancers_stage1, dtype=object)
    all_idx = np.arange(N, dtype=int)

    print(f"[main split] current patient count = {N}")
    print(f"[main split] train_ratio = {train_ratio}")
    print(f"[main split] val_ratio = {val_ratio}")
    print(f"[main split] test_ratio = {test_ratio}")
    print("[main split] stratify by Stage1 cancer label")
    print(
        "[main split] cohort: patients with H&E and molecular embeddings"
    )

    n_test = int(round(N * test_ratio))
    n_val = int(round(N * val_ratio))

    n_test = max(1, min(n_test, N - 2))
    n_val = max(1, min(n_val, N - n_test - 1))

    main_test_idx = _stratified_sample(
        all_idx,
        cancers_arr,
        n_test,
        rng,
    )

    remain_mask = np.ones(N, dtype=bool)
    remain_mask[main_test_idx] = False
    remain_idx = np.where(remain_mask)[0]

    main_val_idx = _stratified_sample(
        remain_idx,
        cancers_arr,
        n_val,
        rng,
    )

    main_train_idx = np.array(
        sorted(list(set(remain_idx.tolist()) - set(main_val_idx.tolist()))),
        dtype=int,
    )

    main_train_ids = set(patient_ids[i] for i in main_train_idx.tolist())
    main_val_ids = set(patient_ids[i] for i in main_val_idx.tolist())
    main_test_ids = set(patient_ids[i] for i in main_test_idx.tolist())

    overlaps = {
        "main_train/main_val": main_train_ids & main_val_ids,
        "main_train/main_test": main_train_ids & main_test_ids,
        "main_val/main_test": main_val_ids & main_test_ids,
    }

    any_overlap = {k: v for k, v in overlaps.items() if len(v) > 0}
    if any_overlap:
        details = {k: list(v)[:20] for k, v in any_overlap.items()}
        raise RuntimeError(f"[main split] patient_id overlap across splits: {details}")

    all_split_ids = main_train_ids | main_val_ids | main_test_ids
    if all_split_ids != set(patient_ids):
        missing = set(patient_ids) - all_split_ids
        extra = all_split_ids - set(patient_ids)
        raise RuntimeError(
            f"[main split] split coverage error. "
            f"missing={list(missing)[:20]}, extra={list(extra)[:20]}"
        )

    payload = dict(
        main_train_idx=main_train_idx.astype(int).tolist(),
        main_val_idx=main_val_idx.astype(int).tolist(),
        main_test_idx=main_test_idx.astype(int).tolist(),
        main_train_patient_ids=[patient_ids[i] for i in main_train_idx.tolist()],
        main_val_patient_ids=[patient_ids[i] for i in main_val_idx.tolist()],
        main_test_patient_ids=[patient_ids[i] for i in main_test_idx.tolist()],
        main_train_cancer_counts=dict(
            Counter([cancers_arr[i] for i in main_train_idx.tolist()])
        ),
        main_val_cancer_counts=dict(
            Counter([cancers_arr[i] for i in main_val_idx.tolist()])
        ),
        main_test_cancer_counts=dict(
            Counter([cancers_arr[i] for i in main_test_idx.tolist()])
        ),
        all_patient_n=int(N),
        seed=int(seed),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        stratify_label="patient_cancers_stage1",
        split_note="TCGA patient split stratified by Stage 1 cancer label.",
    )

    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[main split] saved new split: {split_path}")

    print("\nMain split summary")
    print(f"main_train patients = {len(main_train_idx)}")
    print(f"main_val patients = {len(main_val_idx)}")
    print(f"main_test patients = {len(main_test_idx)}")
    print(
        "[main split] main_test reserved for evaluation"
    )

    print("\n[main split] main_train cancer counts")
    print(
        pd.Series([cancers_arr[i] for i in main_train_idx.tolist()])
        .value_counts()
        .sort_index()
    )

    print("\n[main split] main_val cancer counts")
    print(
        pd.Series([cancers_arr[i] for i in main_val_idx.tolist()])
        .value_counts()
        .sort_index()
    )

    print("\n[main split] main_test cancer counts")
    print(
        pd.Series([cancers_arr[i] for i in main_test_idx.tolist()])
        .value_counts()
        .sort_index()
    )

    return main_train_idx, main_val_idx, main_test_idx, payload


# Dataset / sampler


def build_stage1_alignment_dataset(
    patient_img,
    patient_gene,
    patient_clin,
    patient_cancer_idx,
    selected_patient_indices,
):
    idx = np.array(sorted(set(int(x) for x in selected_patient_indices)), dtype=int)

    return TensorDataset(
        patient_img[idx],
        patient_gene[idx],
        patient_clin[idx],
        torch.tensor(patient_cancer_idx[idx], dtype=torch.long),
    )


class StratifiedCancerBatchSampler(Sampler):
    """Sample cancer-balanced batches of patient embeddings.

    Each batch selects cancers_per_batch groups and draws
    batch_size / cancers_per_batch patients from each selected group.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        cancers_per_batch,
        seed=42,
        drop_last=True,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.cancers_per_batch = int(cancers_per_batch)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        assert self.batch_size % self.cancers_per_batch == 0, (
            f"batch_size ({self.batch_size}) must be divisible by "
            f"cancers_per_batch ({self.cancers_per_batch})"
        )

        self.samples_per_cancer = self.batch_size // self.cancers_per_batch
        self.cancer_to_indices = defaultdict(list)

        for idx in range(len(dataset)):
            cancer_idx = int(dataset[idx][3])
            self.cancer_to_indices[cancer_idx].append(idx)

        self.cancer_keys = sorted(self.cancer_to_indices.keys())

        assert len(self.cancer_keys) >= self.cancers_per_batch, (
            f"Need at least {self.cancers_per_batch} cancers, "
            f"but got {len(self.cancer_keys)}"
        )

        self.num_batches = len(dataset) // self.batch_size
        if not self.drop_last and len(dataset) % self.batch_size != 0:
            self.num_batches += 1

    def __iter__(self):
        rng = np.random.RandomState(self.seed)

        for _ in range(self.num_batches):
            chosen_cancers = rng.choice(
                self.cancer_keys,
                size=self.cancers_per_batch,
                replace=False,
            )

            batch_indices = []

            for c in chosen_cancers:
                indices = self.cancer_to_indices[c]

                if len(indices) >= self.samples_per_cancer:
                    chosen = rng.choice(
                        indices,
                        size=self.samples_per_cancer,
                        replace=False,
                    )
                else:
                    chosen = rng.choice(
                        indices,
                        size=self.samples_per_cancer,
                        replace=True,
                    )

                batch_indices.extend(chosen.tolist())

            rng.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.num_batches


# Build resources


def build_training_resources(config):
    patient_img, patient_gene, patient_ids, patient_cancers = load_patient_embeddings(
        features_output_dir=config["features_output_dir"],
        txt_file_path=config["txt_file_path"],
        patient_csv_path=config["patient_csv_path"],
        patient_uce_path=config["patient_uce_path"],
        id_mode="case",
        dedup_case=True,
    )

    patient_clin = torch.zeros((len(patient_ids), 0), dtype=torch.float32)

    cell_emb_df_raw = pd.read_parquet(config["cell_emb_path"])
    cell_emb_df_raw.index = cell_emb_df_raw.index.astype(str)

    n_total = len(cell_emb_df_raw)
    n_unique = cell_emb_df_raw.index.nunique()
    n_dup = n_total - n_unique

    print(f"[cell_emb] total rows = {n_total}")
    print(f"[cell_emb] unique cell_ids/new_cid = {n_unique}")
    print(f"[cell_emb] duplicate rows/control replicates = {n_dup}")

    # same-cancer MMD uses all control rows.
    cell_emb_df_for_mmd = cell_emb_df_raw.copy()

    # centroid uses unique new_cid mean. lambda_center is currently 0.0.
    if n_dup > 0:
        cell_emb_df_mean = cell_emb_df_raw.groupby(level=0).mean()
    else:
        cell_emb_df_mean = cell_emb_df_raw.copy()

    print(f"[cell_emb_mmd] shape = {cell_emb_df_for_mmd.shape}")
    print(f"[cell_emb_mean] shape = {cell_emb_df_mean.shape}")

    cell_ids_mmd = list(cell_emb_df_for_mmd.index)
    cell_ids_mean = list(cell_emb_df_mean.index)

    cell_cancer_map_mmd = load_cell_cancer_map(config["cell_cancer_path"], cell_ids_mmd)
    cell_cancer_map_mean = load_cell_cancer_map(
        config["cell_cancer_path"], cell_ids_mean
    )

    cell_cancers_mmd = [cell_cancer_map_mmd[cid] for cid in cell_ids_mmd]
    cell_cancers_mean = [cell_cancer_map_mean[cid] for cid in cell_ids_mean]

    patient_cancers_raw = list(patient_cancers)

    patient_cancers_std = [normalize_patient_cancer(x) for x in patient_cancers_raw]
    cell_cancers_mmd_std = [normalize_cell_cancer(x) for x in cell_cancers_mmd]
    cell_cancers_mean_std = [normalize_cell_cancer(x) for x in cell_cancers_mean]

    print("\nTCGA cancer counts after normalization")
    print(pd.Series(patient_cancers_std).value_counts().sort_index())

    print("\nCell cancer counts after normalization, all control rows")
    print(pd.Series(cell_cancers_mmd_std).value_counts().sort_index())

    # Cleaned global tumor cell pool for UNMATCHED fallback

    GLOBAL_EXCLUDE_RAW = {
        "Unknown",
        "Non-Cancerous",
        "Hepatoblastoma",
        "Neuroblastoma",
        "SMARCA4-deficient undifferentiated tumor",
    }

    global_keep_mask = []

    for raw_cancer in cell_cancers_mmd:
        raw_cancer = str(raw_cancer).strip()
        std_cancer = normalize_cell_cancer(raw_cancer)

        keep = True

        if raw_cancer in GLOBAL_EXCLUDE_RAW:
            keep = False

        if std_cancer in {"UNKNOWN", "OTHER"}:
            keep = False

        global_keep_mask.append(keep)

    global_keep_mask = np.asarray(global_keep_mask, dtype=bool)

    cell_emb_df_global = cell_emb_df_for_mmd.iloc[global_keep_mask].copy()
    cell_cancers_global = [
        c for c, keep in zip(cell_cancers_mmd, global_keep_mask) if keep
    ]
    cell_cancers_global_std = [normalize_cell_cancer(x) for x in cell_cancers_global]

    print("\nCleaned global tumor cell pool for UNMATCHED fallback")
    print(f"global tumor rows = {len(cell_emb_df_global)}")
    print(f"global tumor unique new_cid = {cell_emb_df_global.index.nunique()}")
    print(pd.Series(cell_cancers_global_std).value_counts().sort_index())

    if len(cell_emb_df_global) < 2:
        raise RuntimeError("[global cell pool] cleaned global tumor pool has < 2 rows")

    # Matchable cancer decision

    UNMATCHED_LABEL = "UNMATCHED"

    patient_counter_raw = Counter(patient_cancers_std)
    cell_counter_raw = Counter(cell_cancers_mmd_std)

    matchable_cancers = set()

    SPECIAL_NO_SAME_CANCER_MMD = {
        "OTHER",
        "UNKNOWN",
        "TCGA_OTHER_UNMATCHED",
        "UNMATCHED",
    }

    for cancer in set(patient_cancers_std) | set(cell_cancers_mmd_std):
        if cancer in SPECIAL_NO_SAME_CANCER_MMD:
            continue

        patient_n = patient_counter_raw.get(cancer, 0)
        cell_n = cell_counter_raw.get(cancer, 0)

        if (
            cell_n > 0
            and patient_n >= config["stage1_min_tcga_samples_for_match"]
            and cell_n >= config["stage1_min_cell_samples_for_match"]
        ):
            matchable_cancers.add(cancer)

    print("\nMatchable cancers for same-cancer MMD")
    print(sorted(matchable_cancers))

    # Non-matchable TCGA cancers are mapped to UNMATCHED.
    patient_cancers_stage1 = [
        c if c in matchable_cancers else UNMATCHED_LABEL for c in patient_cancers_std
    ]

    print("\nStage1 patient cancer counts")
    print(pd.Series(patient_cancers_stage1).value_counts().sort_index())

    # real encoder preserves biological normalized labels.
    real_cancer_encoder = build_shared_cancer_label_encoder(
        patient_cancers_std,
        cell_cancers_mmd_std,
    )

    # stage1 encoder is used for training and checkpoint compatibility.
    stage1_cancer_encoder = build_shared_cancer_label_encoder(
        patient_cancers_stage1,
        cell_cancers_mmd_std,
    )

    patient_cancer_idx_stage1 = encode_cancer_labels(
        patient_cancers_stage1,
        stage1_cancer_encoder,
    )

    cell_cancer_idx_mmd = encode_cancer_labels(
        cell_cancers_mmd_std,
        stage1_cancer_encoder,
    )

    cell_cancer_idx_mean = encode_cancer_labels(
        cell_cancers_mean_std,
        stage1_cancer_encoder,
    )

    cell_cancer_idx_tensor_mmd = torch.tensor(cell_cancer_idx_mmd, dtype=torch.long)
    cell_cancer_idx_tensor_mean = torch.tensor(cell_cancer_idx_mean, dtype=torch.long)

    unmatched_cancer_idx = stage1_cancer_encoder.get(UNMATCHED_LABEL, None)

    matchable_cancer_indices = sorted(
        stage1_cancer_encoder[c]
        for c in matchable_cancers
        if c in stage1_cancer_encoder
    )

    os.makedirs(config["save_dir"], exist_ok=True)

    with open(
        os.path.join(config["save_dir"], "real_cancer_encoder.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(real_cancer_encoder, f, ensure_ascii=False, indent=2)

    with open(
        os.path.join(config["save_dir"], "stage1_cancer_encoder.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(stage1_cancer_encoder, f, ensure_ascii=False, indent=2)

    with open(
        os.path.join(config["save_dir"], "patient_cancers_raw_std_stage1.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "patient_cancers_raw": patient_cancers_raw,
                "patient_cancers_std": patient_cancers_std,
                "patient_cancers_stage1": patient_cancers_stage1,
                "matchable_cancers": sorted(matchable_cancers),
                "unmatched_cancer_idx": unmatched_cancer_idx,
                "matchable_cancer_indices": matchable_cancer_indices,
                "global_pool_note": "LINCS tumor controls with eligible cancer labels.",
                "main_split_note": "TCGA patients with H&E and molecular embeddings.",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Main train / val / test split

    split_path = os.path.join(config["mode_dir"], config["split_filename"])

    train_idx, val_idx, main_test_idx, split_payload = (
        make_or_load_main_split_train_val_test(
            patient_ids=list(patient_ids),
            patient_cancers_stage1=list(patient_cancers_stage1),
            split_path=split_path,
            seed=config["seed"],
            train_ratio=config["main_train_ratio"],
            val_ratio=config["main_val_ratio"],
            test_ratio=config["main_test_ratio"],
        )
    )

    # Save a copy of the split in the Stage1 output directory.
    with open(
        os.path.join(config["save_dir"], "main_split_used.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(split_payload, f, ensure_ascii=False, indent=2)

    # Validate saved split indices against patient IDs.
    split_checks = [
        ("main_train", train_idx, split_payload["main_train_patient_ids"]),
        ("main_val", val_idx, split_payload["main_val_patient_ids"]),
        ("main_test", main_test_idx, split_payload["main_test_patient_ids"]),
    ]

    patient_ids_list = list(patient_ids)

    for split_name, split_idx, expected_patient_ids in split_checks:
        current_patient_ids = [
            patient_ids_list[int(i)] for i in np.asarray(split_idx, dtype=int).tolist()
        ]
        if current_patient_ids != expected_patient_ids:
            print(f"[split-check] {split_name} patient_id order mismatch")
            print(f"[split-check] expected first20 = {expected_patient_ids[:20]}")
            print(f"[split-check] current first20 = {current_patient_ids[:20]}")
            raise RuntimeError(
                f"[split-check] {split_name} patient_id list does not match split json"
            )

    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    main_test_idx = np.asarray(main_test_idx, dtype=int)

    cell_emb_tensor_mmd = torch.tensor(cell_emb_df_for_mmd.values, dtype=torch.float32)
    cell_emb_tensor_mean = torch.tensor(cell_emb_df_mean.values, dtype=torch.float32)
    cell_emb_tensor_global = torch.tensor(
        cell_emb_df_global.values, dtype=torch.float32
    )

    stage1_train_dataset = build_stage1_alignment_dataset(
        patient_img=patient_img,
        patient_gene=patient_gene,
        patient_clin=patient_clin,
        patient_cancer_idx=np.asarray(patient_cancer_idx_stage1),
        selected_patient_indices=train_idx,
    )

    stage1_val_dataset = build_stage1_alignment_dataset(
        patient_img=patient_img,
        patient_gene=patient_gene,
        patient_clin=patient_clin,
        patient_cancer_idx=np.asarray(patient_cancer_idx_stage1),
        selected_patient_indices=val_idx,
    )

    num_cancers = len(stage1_cancer_encoder)

    cell_centroids = []

    for c in range(num_cancers):
        mask = cell_cancer_idx_tensor_mean == c

        if mask.sum() == 0:
            cell_centroids.append(
                torch.zeros(cell_emb_tensor_mean.shape[1], dtype=torch.float32)
            )
        else:
            cell_centroids.append(cell_emb_tensor_mean[mask].mean(dim=0))

    cell_centroids = torch.stack(cell_centroids, dim=0)

    print("\nMain split summary")
    print(f"main_train patients = {len(train_idx)}")
    print(f"main_val patients = {len(val_idx)}")
    print(f"main_test patients = {len(main_test_idx)}")
    print(
        "[main split] main_test reserved for evaluation"
    )

    return {
        "stage1_train_dataset": stage1_train_dataset,
        "stage1_val_dataset": stage1_val_dataset,
        "cell_emb_tensor_mmd": cell_emb_tensor_mmd,
        "cell_cancer_idx_tensor_mmd": cell_cancer_idx_tensor_mmd,
        "cell_emb_tensor_global": cell_emb_tensor_global,
        "cell_centroids": cell_centroids,
        "real_cancer_encoder": real_cancer_encoder,
        "stage1_cancer_encoder": stage1_cancer_encoder,
        "unmatched_cancer_idx": unmatched_cancer_idx,
        "matchable_cancer_indices": matchable_cancer_indices,
        "patient_img": patient_img,
        "patient_gene": patient_gene,
        "patient_clin": patient_clin,
        "patient_ids": patient_ids,
        "patient_cancers_raw": patient_cancers_raw,
        "patient_cancers_std": patient_cancers_std,
        "patient_cancers_stage1": patient_cancers_stage1,
        "main_train_idx": train_idx,
        "main_val_idx": val_idx,
        "main_test_idx": main_test_idx,
        "split_payload": split_payload,
    }


# Plot


def plot_stage1_history(history, save_dir):
    if len(history["epoch"]) == 0:
        return

    epochs = history["epoch"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Stage1 Total Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "stage1_total_loss.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_mmd_cancer"], label="train_mmd_cancer")
    plt.plot(epochs, history["val_mmd_cancer"], label="val_mmd_cancer")
    plt.xlabel("Epoch")
    plt.ylabel("MMD Cancer")
    plt.title("Stage1 Same-cancer MMD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "stage1_mmd_cancer.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_mmd_global"], label="train_mmd_global")
    plt.plot(epochs, history["val_mmd_global"], label="val_mmd_global")
    plt.xlabel("Epoch")
    plt.ylabel("MMD Global")
    plt.title("Stage1 UNMATCHED Global Fallback MMD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "stage1_mmd_global.png"), dpi=200)
    plt.close()

    print(f"[plot_stage1_history] saved plots to {save_dir}")


# Stage1 train


def train_stage1_alignment(
    model,
    train_dataset,
    val_dataset,
    cell_emb_tensor,
    cell_cancer_idx_tensor,
    cell_emb_tensor_global,
    cell_centroids,
    unmatched_cancer_idx,
    matchable_cancer_indices,
    config,
    device,
):
    print("\nStage 1: TCGA -> cell space alignment")
    print(f"len(train_dataset) = {len(train_dataset)}")
    print(f"len(val_dataset) = {len(val_dataset)}")
    print(f"cell_emb_tensor.shape = {cell_emb_tensor.shape}")
    print(f"cell_cancer_idx_tensor.shape = {cell_cancer_idx_tensor.shape}")
    print(f"cell_emb_tensor_global.shape = {cell_emb_tensor_global.shape}")
    print(f"cell_centroids.shape = {cell_centroids.shape}")

    sample = train_dataset[0]
    print(f"train_dataset[0] img shape = {sample[0].shape}")
    print(f"train_dataset[0] gene shape = {sample[1].shape}")
    print(f"train_dataset[0] clin shape = {sample[2].shape}")
    print(f"train_dataset[0] cancer_idx = {sample[3]}")

    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    train_batch_sampler = StratifiedCancerBatchSampler(
        dataset=train_dataset,
        batch_size=config["stage1_batch_size"],
        cancers_per_batch=config["stage1_cancers_per_batch"],
        seed=config["seed"],
        drop_last=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["stage1_batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # Freeze all parameters. Only train tcga_align_head.
    for parameter in model.parameters():
        parameter.requires_grad = False

    unfreeze_module(model.tcga_align_head)

    optimizer = make_optimizer(
        model,
        lr=config["stage1_lr"],
        weight_decay=1e-4,
    )

    cell_emb_tensor = cell_emb_tensor.to(device)
    cell_cancer_idx_tensor = cell_cancer_idx_tensor.to(device)
    cell_emb_tensor_global = cell_emb_tensor_global.to(device)
    cell_centroids = cell_centroids.to(device)

    print("\nStage1 effective settings")
    print(f"[Stage1] batch_size = {config['stage1_batch_size']}")
    print(f"[Stage1] cancers_per_batch = {config['stage1_cancers_per_batch']}")
    print(
        f"[Stage1] samples_per_cancer = {config['stage1_batch_size'] // config['stage1_cancers_per_batch']}"
    )
    print(f"[Stage1] lr = {config['stage1_lr']}")
    print(f"[Stage1] epochs = {config['stage1_epochs']}")
    print(f"[Stage1] lambda_mmd_cancer = {config['lambda_mmd_cancer']}")
    print(f"[Stage1] lambda_center = {config['lambda_center']}")
    print(f"[Stage1] lambda_mmd_global = {config['lambda_mmd_global']}")
    print(f"[Stage1] same_cancer_sample_n = {config['stage1_same_cancer_sample_n']}")
    print(f"[Stage1] global_sample_n = {config['stage1_global_sample_n']}")
    print(f"[Stage1] unmatched_cancer_idx = {unmatched_cancer_idx}")
    print(f"[Stage1] matchable_cancer_indices = {matchable_cancer_indices}")

    history = {
        "epoch": [],
        "train_loss": [],
        "train_mmd_cancer": [],
        "train_mmd_global": [],
        "train_center": [],
        "val_loss": [],
        "val_mmd_cancer": [],
        "val_mmd_global": [],
        "val_center": [],
    }

    best_val_metric = float("inf")
    best_epoch = -1
    best_state_dict = None

    patience_counter = 0
    patience = config["stage1_early_stop_patience"]
    min_delta = config["stage1_min_delta"]

    for epoch in range(config["stage1_epochs"]):
        print(
            f"\n[Stage1] epoch={epoch + 1} "
            f"batch_size={config['stage1_batch_size']} "
            f"cancers_per_batch={config['stage1_cancers_per_batch']}"
        )

        # Train

        model.train()

        train_meters = {
            "loss": 0.0,
            "loss_mmd_cancer": 0.0,
            "loss_mmd_global": 0.0,
            "loss_center": 0.0,
        }

        n_train_batches = 0

        for _, gene_embeddings, _, cancer_idx in tqdm(
            train_loader,
            desc=f"Stage1 Train {epoch + 1}",
            dynamic_ncols=True,
        ):
            gene_embeddings = gene_embeddings.to(device, non_blocking=True)
            cancer_idx = cancer_idx.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            losses = model.stage1_align_losses(
                tcga_emb=gene_embeddings,
                tcga_cancer_idx=cancer_idx,
                cell_centroids=cell_centroids,
                cell_emb_all=cell_emb_tensor,
                cell_cancer_idx_all=cell_cancer_idx_tensor,
                cell_emb_global=cell_emb_tensor_global,
                unmatched_cancer_idx=unmatched_cancer_idx,
                matchable_cancer_indices=matchable_cancer_indices,
                lambda_center=config["lambda_center"],
                lambda_mmd_cancer=config["lambda_mmd_cancer"],
                lambda_mmd_global=config["lambda_mmd_global"],
                same_cancer_sample_n=config["stage1_same_cancer_sample_n"],
                global_sample_n=config["stage1_global_sample_n"],
            )

            loss = losses["loss"]
            loss.backward()
            optimizer.step()

            train_meters["loss"] += float(losses["loss"].item())
            train_meters["loss_mmd_cancer"] += float(losses["loss_mmd_cancer"].item())
            train_meters["loss_mmd_global"] += float(losses["loss_mmd_global"].item())
            train_meters["loss_center"] += float(losses["loss_center"].item())

            n_train_batches += 1

        train_loss = train_meters["loss"] / max(n_train_batches, 1)
        train_mmd_cancer = train_meters["loss_mmd_cancer"] / max(n_train_batches, 1)
        train_mmd_global = train_meters["loss_mmd_global"] / max(n_train_batches, 1)
        train_center = train_meters["loss_center"] / max(n_train_batches, 1)

        print(
            f"[Stage1 Train {epoch + 1}] "
            f"loss={train_loss:.6f} "
            f"mmd_cancer={train_mmd_cancer:.6f} "
            f"mmd_global={train_mmd_global:.6f} "
            f"center={train_center:.6f}"
        )

        # Val

        model.eval()

        val_meters = {
            "loss": 0.0,
            "loss_mmd_cancer": 0.0,
            "loss_mmd_global": 0.0,
            "loss_center": 0.0,
        }

        n_val_batches = 0

        with torch.no_grad():
            for _, gene_embeddings, _, cancer_idx in tqdm(
                val_loader,
                desc=f"Stage1 Val {epoch + 1}",
                dynamic_ncols=True,
            ):
                gene_embeddings = gene_embeddings.to(device, non_blocking=True)
                cancer_idx = cancer_idx.to(device, non_blocking=True)

                losses = model.stage1_align_losses(
                    tcga_emb=gene_embeddings,
                    tcga_cancer_idx=cancer_idx,
                    cell_centroids=cell_centroids,
                    cell_emb_all=cell_emb_tensor,
                    cell_cancer_idx_all=cell_cancer_idx_tensor,
                    cell_emb_global=cell_emb_tensor_global,
                    unmatched_cancer_idx=unmatched_cancer_idx,
                    matchable_cancer_indices=matchable_cancer_indices,
                    lambda_center=config["lambda_center"],
                    lambda_mmd_cancer=config["lambda_mmd_cancer"],
                    lambda_mmd_global=config["lambda_mmd_global"],
                    same_cancer_sample_n=config["stage1_same_cancer_sample_n"],
                    global_sample_n=config["stage1_global_sample_n"],
                )

                val_meters["loss"] += float(losses["loss"].item())
                val_meters["loss_mmd_cancer"] += float(losses["loss_mmd_cancer"].item())
                val_meters["loss_mmd_global"] += float(losses["loss_mmd_global"].item())
                val_meters["loss_center"] += float(losses["loss_center"].item())

                n_val_batches += 1

        val_loss = val_meters["loss"] / max(n_val_batches, 1)
        val_mmd_cancer = val_meters["loss_mmd_cancer"] / max(n_val_batches, 1)
        val_mmd_global = val_meters["loss_mmd_global"] / max(n_val_batches, 1)
        val_center = val_meters["loss_center"] / max(n_val_batches, 1)

        print(
            f"[Stage1 Val {epoch + 1}] "
            f"loss={val_loss:.6f} "
            f"mmd_cancer={val_mmd_cancer:.6f} "
            f"mmd_global={val_mmd_global:.6f} "
            f"center={val_center:.6f}"
        )

        # History

        history["epoch"].append(epoch + 1)

        history["train_loss"].append(train_loss)
        history["train_mmd_cancer"].append(train_mmd_cancer)
        history["train_mmd_global"].append(train_mmd_global)
        history["train_center"].append(train_center)

        history["val_loss"].append(val_loss)
        history["val_mmd_cancer"].append(val_mmd_cancer)
        history["val_mmd_global"].append(val_mmd_global)
        history["val_center"].append(val_center)

        with open(
            os.path.join(save_dir, "stage1_history.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        pd.DataFrame(history).to_csv(
            os.path.join(save_dir, "stage1_history.csv"),
            index=False,
        )

        # Save last

        save_checkpoint(
            os.path.join(save_dir, "stage1_alignment_last.pt"),
            model,
            optimizer=optimizer,
            epoch=epoch + 1,
            metrics={
                "stage": "stage1_alignment_v21_main_split_clean_unmatched_global_lam03",
                "val_loss": val_loss,
                "val_mmd_cancer": val_mmd_cancer,
                "val_mmd_global": val_mmd_global,
                "val_center": val_center,
            },
            config=config,
        )

        # Save best by val_loss

        improved = (best_val_metric - val_loss) > min_delta

        if improved:
            best_val_metric = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_state_dict = copy.deepcopy(model.state_dict())

            save_checkpoint(
                os.path.join(save_dir, "stage1_alignment_model.pt"),
                model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics={
                    "stage": "stage1_alignment_v21_main_split_clean_unmatched_global_lam03",
                    "best_val_loss": best_val_metric,
                    "val_loss": val_loss,
                    "val_mmd_cancer": val_mmd_cancer,
                    "val_mmd_global": val_mmd_global,
                    "val_center": val_center,
                },
                config=config,
            )

            print(
                f"[Stage1] new best val_loss = {best_val_metric:.6f} at epoch {best_epoch}"
            )

        else:
            patience_counter += 1
            print(
                f"[Stage1] no improvement. "
                f"best_val_loss={best_val_metric:.6f} at epoch {best_epoch}. "
                f"patience {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print(f"[Stage1] Early stopping triggered at epoch {epoch + 1}.")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict, strict=False)
        print(
            f"[Stage1] loaded best model from epoch {best_epoch} "
            f"with val_loss={best_val_metric:.6f}"
        )

    plot_stage1_history(history, save_dir)

    return history


# Main


def train_patient_stage1_only(config=None):
    """Run molecular alignment using precomputed features and a fixed patient split."""
    if config is None:
        raise ValueError("Use python -m histopert with an explicit configuration")

    os.makedirs(config["save_dir"], exist_ok=True)

    with open(
        os.path.join(
            config["save_dir"],
            "config_stage1_v21_main_split_clean_unmatched_global_lam03.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\nStage 1: molecular alignment")
    print("experiment_root =", config["experiment_root"])
    print("save_dir =", config["save_dir"])
    print("split_dir =", config["mode_dir"])
    print("split_filename =", config["split_filename"])
    print("device =", config["device"])
    print("patient_uce_path =", config["patient_uce_path"])
    print("resume_from_stage1_ckpt =", config["resume_from_stage1_ckpt"])
    print("resume_stage1_ckpt_path =", config["resume_stage1_ckpt_path"])

    set_seed(config["seed"])
    device = torch.device(config["device"])

    resources = build_training_resources(config)

    stage1_train_dataset = resources["stage1_train_dataset"]
    stage1_val_dataset = resources["stage1_val_dataset"]
    cell_emb_tensor_mmd = resources["cell_emb_tensor_mmd"]
    cell_cancer_idx_tensor_mmd = resources["cell_cancer_idx_tensor_mmd"]
    cell_emb_tensor_global = resources["cell_emb_tensor_global"]
    cell_centroids = resources["cell_centroids"]
    stage1_cancer_encoder = resources["stage1_cancer_encoder"]

    sample = stage1_train_dataset[0]
    img_dim = sample[0].numel()
    gene_dim = sample[1].numel()
    clin_dim = sample[2].numel()
    cell_dim = cell_emb_tensor_mmd.shape[1]

    print("\nModel dim check")
    print("img_dim =", img_dim)
    print("gene_dim =", gene_dim)
    print("clin_dim =", clin_dim)
    print("cell_dim =", cell_dim)
    print("num_cancers =", len(stage1_cancer_encoder))

    if cell_dim != 1280:
        raise ValueError(f"Expected cell_dim=1280, got {cell_dim}")

    model = PatientAlignmentModel(
        img_dim=img_dim,
        gene_dim=gene_dim,
        cell_dim=cell_dim,
        drug_dim=1,
        output_dim=1,
        clin_dim=clin_dim,
        num_cancers=len(stage1_cancer_encoder),
        hidden_dim=config["hidden_dim"],
        cancer_emb_dim=config["cancer_emb_dim"],
        dropout=config["dropout"],
        use_clinical=config["use_clinical"],
        unicure_model=None,
    ).to(device)

    if config.get("resume_from_stage1_ckpt", False):
        model, _ = resume_stage1_from_checkpoint(
            model=model,
            config=config,
            device=device,
        )

    if config.get("run_stage1", True):
        history = train_stage1_alignment(
            model=model,
            train_dataset=stage1_train_dataset,
            val_dataset=stage1_val_dataset,
            cell_emb_tensor=cell_emb_tensor_mmd,
            cell_cancer_idx_tensor=cell_cancer_idx_tensor_mmd,
            cell_emb_tensor_global=cell_emb_tensor_global,
            cell_centroids=cell_centroids,
            unmatched_cancer_idx=resources["unmatched_cancer_idx"],
            matchable_cancer_indices=resources["matchable_cancer_indices"],
            config=config,
            device=device,
        )

        print("\nFinished Stage 1: molecular alignment")
        print("Best Stage1 checkpoint:")
        print(os.path.join(config["save_dir"], "stage1_alignment_model.pt"))
        print("Last Stage1 checkpoint:")
        print(os.path.join(config["save_dir"], "stage1_alignment_last.pt"))
        print("Main split used:")
        print(os.path.join(config["save_dir"], "main_split_used.json"))

        return history

    print("[Stage1] resource preparation complete")
    return None
