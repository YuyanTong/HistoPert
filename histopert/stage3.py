"""Stage 3: patient-context perturbation prediction with fixed proxy targets."""

import hashlib
import json
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .condition_specific_dataset import (
    RECORD_SCHEMA_VERSION,
    STORE_SCHEMA_VERSION,
    ConditionSpecificPatientDrugDoseDataset,
    condition_index_from_lincs_store,
    deterministic_audit_indices,
    hash_numpy_array,
    load_or_build_condition_store,
    load_or_build_split_records,
    source_file_manifest,
    strict_same_cancer_base_proxy,
)
from .data import (
    encode_cancer_labels,
    load_cell_cancer_map,
    load_patient_embeddings,
)
from .fusion import PatientContextModel
from .io import load_checkpoint
from .lbpt_solver import compute_reconstruction_metrics, solve_lbpt_simplex
from .perturbation import PerturbationPredictor

# Basic utils


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def compute_batch_pearson(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    correlations = []
    for i in range(pred_np.shape[0]):
        prediction = pred_np[i]
        target_values = target_np[i]

        if np.std(prediction) < 1e-9 or np.std(target_values) < 1e-9:
            correlations.append(0.0)
        else:
            correlations.append(np.corrcoef(prediction, target_values)[0, 1])

    return float(np.mean(correlations))


# Cancer label normalization


def normalize_cell_cancer(x: str) -> str:
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


def normalize_patient_cancer_stage1(x: str) -> str:
    """Map TCGA projects to reference cancer groups for cell matching."""
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
        "TCGA-STAD": "STAD_ESCA",
        "TCGA-UCEC": "UCEC_UCS",
        "TCGA-UCS": "UCEC_UCS",
        "TCGA-MESO": "TCGA_OTHER_UNMATCHED",
        "TCGA-PCPG": "TCGA_OTHER_UNMATCHED",
        "TCGA-TGCT": "TCGA_OTHER_UNMATCHED",
        "TCGA-THCA": "TCGA_OTHER_UNMATCHED",
        "TCGA-THYM": "TCGA_OTHER_UNMATCHED",
        "TCGA-UVM": "TCGA_OTHER_UNMATCHED",
    }

    return patient_map.get(x, "UNKNOWN")


def normalize_patient_cancer_for_neighbor(x: str) -> str:
    """Use TCGA project labels for H&E neighbor retrieval."""
    x = str(x).strip()

    if x.startswith("TCGA-"):
        return x.replace("TCGA-", "")

    return x


# Split utilities


def load_main_split_train_val_test(patient_ids, split_path):
    """Load patient indices from the Stage 1 train/validation/test split."""
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

    print(f"[main split] loading: {split_path}")
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

    for name, saved_patient_ids in split_patient_ids.items():
        saved_idx = split_indices[name]

        if np.any(saved_idx < 0) or np.any(saved_idx >= len(patient_ids)):
            raise RuntimeError(f"[main split] {name}_idx contains out-of-range indices")

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
            print(f"[main split] saved patient_id first20 = {saved_patient_ids[:20]}")
            print(f"[main split] idx patient_id first20 = {ids_from_saved_idx[:20]}")
            raise RuntimeError(
                f"[main split] {name}_idx does not match {name}_patient_ids"
            )

    all_idx = np.concatenate(
        [
            split_indices["main_train"],
            split_indices["main_val"],
            split_indices["main_test"],
        ]
    )

    if len(set(all_idx.tolist())) != len(patient_ids):
        raise RuntimeError(
            f"[main split] coverage mismatch: unique split idx={len(set(all_idx.tolist()))}, "
            f"patient_n={len(patient_ids)}"
        )

    print("\nMain split summary")
    print(f"main_train patients = {len(split_indices['main_train'])}")
    print(f"main_val patients   = {len(split_indices['main_val'])}")
    print(f"main_test patients  = {len(split_indices['main_test'])}")
    print(
        "[main split] main_test reserved for evaluation"
    )

    return (
        split_indices["main_train"],
        split_indices["main_val"],
        split_indices["main_test"],
        split_metadata,
    )


# Encoder loading


def load_stage1_cancer_encoder(config):
    stage_dir = config["stage1_dir"]

    candidates = [
        os.path.join(stage_dir, "stage1_cancer_encoder.json"),
        os.path.join(stage_dir, "stage1_cancer_encoder_for_export.json"),
    ]

    for path in candidates:
        if os.path.exists(path):
            print(f"[encoder] loading stage1 cancer encoder from: {path}")

            with open(path, "r", encoding="utf-8") as f:
                enc = json.load(f)

            enc = {str(k): int(v) for k, v in enc.items()}

            print(f"[encoder] num_cancers = {len(enc)}")
            print("[encoder] labels:")
            index_to_cancer = {v: k for k, v in enc.items()}
            for i in sorted(index_to_cancer):
                print(f"  {i}: {index_to_cancer[i]}")

            return enc

    raise FileNotFoundError(
        "Could not find stage1 cancer encoder. Tried:\n" + "\n".join(candidates)
    )


# Neighbor functions


def compute_topk_neighbors_same_project_patient(
    query_emb: torch.Tensor,
    query_project: np.ndarray,
    query_patient_ids,
    candidate_emb: torch.Tensor,
    candidate_project: np.ndarray,
    candidate_patient_ids,
    k: int = 5,
    trace_scale: float = 1e-3,
    absolute_epsilon: float = 1e-8,
):
    """HE image neighbor:
    - restrict to same TCGA project label
    - exclude the query patient by patient ID for every split
    """
    query_project = np.asarray(query_project, dtype=object)
    candidate_project = np.asarray(candidate_project, dtype=object)
    query_patient_ids = np.asarray(query_patient_ids, dtype=object)
    candidate_patient_ids = np.asarray(candidate_patient_ids, dtype=object)

    query_count = query_emb.size(0)
    candidate_count = candidate_emb.size(0)

    if len(query_project) != query_count or len(query_patient_ids) != query_count:
        raise ValueError("query metadata length does not match query_emb")
    if (
        len(candidate_project) != candidate_count
        or len(candidate_patient_ids) != candidate_count
    ):
        raise ValueError("candidate metadata length does not match candidate_emb")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    query_emb = F.normalize(query_emb.float(), dim=1)
    candidate_emb = F.normalize(candidate_emb.float(), dim=1)

    idx_out = np.zeros((query_count, k), dtype=np.int64)
    alpha_out = np.zeros((query_count, k), dtype=np.float64)
    cosine_out = np.zeros((query_count, k), dtype=np.float32)

    diagnostic_fields = {
        "regularization": np.zeros((query_count,), dtype=np.float64),
        "gram_trace": np.zeros((query_count,), dtype=np.float64),
        "objective": np.zeros((query_count,), dtype=np.float64),
        "effective_n": np.zeros((query_count,), dtype=np.float64),
        "he_reconstruction_l2_error": np.zeros((query_count,), dtype=np.float64),
        "he_reconstruction_relative_l2_error": np.zeros(
            (query_count,), dtype=np.float64
        ),
        "qp_kkt_residual": np.zeros((query_count,), dtype=np.float64),
        "qp_solver_iterations": np.zeros((query_count,), dtype=np.int32),
        "qp_solver_status": np.empty((query_count,), dtype=object),
        "qp_solver_message": np.empty((query_count,), dtype=object),
    }

    similarity_matrix = query_emb @ candidate_emb.T

    for i in range(query_count):
        q_project = query_project[i]

        candidate_indices = np.where(candidate_project == q_project)[0]
        candidate_indices = candidate_indices[
            candidate_patient_ids[candidate_indices] != query_patient_ids[i]
        ]

        if len(candidate_indices) < k:
            raise RuntimeError(
                "strict same-project patient library has fewer than k distinct "
                f"candidates: patient_id={query_patient_ids[i]!r}, "
                f"project={q_project!r}, available={len(candidate_indices)}, k={k}"
            )

        sim_values = (
            similarity_matrix[i, candidate_indices].cpu().numpy().astype(np.float64)
        )
        order = np.lexsort((candidate_indices, -sim_values))
        selected_indices = candidate_indices[order[:k]]
        cosine = similarity_matrix[i, selected_indices].cpu().numpy().astype(np.float32)
        solution = solve_lbpt_simplex(
            query_emb[i],
            candidate_emb[selected_indices],
            trace_scale=float(trace_scale),
            eps=float(absolute_epsilon),
        )
        idx_out[i] = selected_indices
        alpha_out[i] = solution.alpha
        cosine_out[i] = cosine
        diagnostic_fields["regularization"][i] = solution.regularization
        diagnostic_fields["gram_trace"][i] = solution.gram_trace
        diagnostic_fields["objective"][i] = solution.objective
        diagnostic_fields["effective_n"][i] = solution.effective_n
        diagnostic_fields["he_reconstruction_l2_error"][i] = (
            solution.reconstruction.l2_error
        )
        diagnostic_fields["he_reconstruction_relative_l2_error"][i] = (
            solution.reconstruction.relative_l2_error
        )
        diagnostic_fields["qp_kkt_residual"][i] = solution.kkt.residual
        diagnostic_fields["qp_solver_iterations"][i] = solution.solver_iterations
        diagnostic_fields["qp_solver_status"][i] = solution.solver_status
        diagnostic_fields["qp_solver_message"][i] = solution.solver_message

    selected_patient_ids = candidate_patient_ids[idx_out]
    same_patient_mask = selected_patient_ids == query_patient_ids[:, None]
    if np.any(same_patient_mask):
        bad_i, bad_j = np.argwhere(same_patient_mask)[0]
        raise AssertionError(
            "patient self-exclusion failed: "
            f"query_row={bad_i}, neighbor_slot={bad_j}, "
            f"patient_id={query_patient_ids[bad_i]!r}"
        )

    if np.any(alpha_out < -1e-9) or not np.allclose(
        alpha_out.sum(axis=1), 1.0, atol=1e-9, rtol=0.0
    ):
        raise AssertionError("verified LBPT alpha failed the batched simplex check")

    return idx_out, alpha_out, cosine_out, diagnostic_fields


def save_patient_neighbor_audit(
    save_dir,
    split_tag,
    query_indices,
    neighbor_indices_global,
    neighbor_weights,
    neighbor_cosines,
    lbpt_diagnostics,
    patient_ids,
    patient_projects,
    gene_reconstruction_l2=None,
    gene_reconstruction_relative_l2=None,
    cell_reconstruction_l2=None,
    cell_reconstruction_relative_l2=None,
):
    query_indices = np.asarray(query_indices, dtype=np.int64)
    neighbor_indices_global = np.asarray(neighbor_indices_global, dtype=np.int64)
    neighbor_weights = np.asarray(neighbor_weights, dtype=np.float64)
    neighbor_cosines = np.asarray(neighbor_cosines, dtype=np.float32)
    patient_ids_arr = np.asarray(patient_ids, dtype=object)
    patient_projects_arr = np.asarray(patient_projects, dtype=object)

    query_ids = patient_ids_arr[query_indices]
    neighbor_ids = patient_ids_arr[neighbor_indices_global]
    query_projects = patient_projects_arr[query_indices]
    neighbor_projects = patient_projects_arr[neighbor_indices_global]

    same_patient = neighbor_ids == query_ids[:, None]
    same_project = neighbor_projects == query_projects[:, None]
    weight_sums = neighbor_weights.sum(axis=1)
    if neighbor_cosines.shape != neighbor_weights.shape:
        raise ValueError(f"{split_tag}: cosine and alpha shapes differ")
    gene_reconstruction_l2 = (
        np.full((len(query_indices),), np.nan, dtype=np.float64)
        if gene_reconstruction_l2 is None
        else np.asarray(gene_reconstruction_l2, dtype=np.float64)
    )
    gene_reconstruction_relative_l2 = (
        np.full((len(query_indices),), np.nan, dtype=np.float64)
        if gene_reconstruction_relative_l2 is None
        else np.asarray(gene_reconstruction_relative_l2, dtype=np.float64)
    )
    cell_reconstruction_l2 = (
        np.full((len(query_indices),), np.nan, dtype=np.float64)
        if cell_reconstruction_l2 is None
        else np.asarray(cell_reconstruction_l2, dtype=np.float64)
    )
    cell_reconstruction_relative_l2 = (
        np.full((len(query_indices),), np.nan, dtype=np.float64)
        if cell_reconstruction_relative_l2 is None
        else np.asarray(cell_reconstruction_relative_l2, dtype=np.float64)
    )
    unique_neighbor_n = np.asarray(
        [len(set(row.tolist())) for row in neighbor_ids],
        dtype=np.int64,
    )

    rows = []
    for i in range(len(query_indices)):
        rows.append(
            {
                "query_global_idx": int(query_indices[i]),
                "query_patient_id": str(query_ids[i]),
                "query_project": str(query_projects[i]),
                "neighbor_global_indices": ";".join(
                    str(int(x)) for x in neighbor_indices_global[i]
                ),
                "neighbor_patient_ids": ";".join(str(x) for x in neighbor_ids[i]),
                "neighbor_projects": ";".join(str(x) for x in neighbor_projects[i]),
                "neighbor_weights": ";".join(
                    f"{float(x):.9f}" for x in neighbor_weights[i]
                ),
                "neighbor_cosines": ";".join(
                    f"{float(x):.9f}" for x in neighbor_cosines[i]
                ),
                "alpha_min": float(np.min(neighbor_weights[i])),
                "alpha_max": float(np.max(neighbor_weights[i])),
                "effective_neighbor_n": float(lbpt_diagnostics["effective_n"][i]),
                "lbpt_regularization": float(lbpt_diagnostics["regularization"][i]),
                "lbpt_gram_trace": float(lbpt_diagnostics["gram_trace"][i]),
                "lbpt_objective": float(lbpt_diagnostics["objective"][i]),
                "he_reconstruction_l2": float(
                    lbpt_diagnostics["he_reconstruction_l2_error"][i]
                ),
                "he_reconstruction_relative_l2": float(
                    lbpt_diagnostics["he_reconstruction_relative_l2_error"][i]
                ),
                "gene_reconstruction_l2": float(gene_reconstruction_l2[i]),
                "gene_reconstruction_relative_l2": float(
                    gene_reconstruction_relative_l2[i]
                ),
                "cell_reconstruction_l2": float(cell_reconstruction_l2[i]),
                "cell_reconstruction_relative_l2": float(
                    cell_reconstruction_relative_l2[i]
                ),
                "qp_kkt_residual": float(lbpt_diagnostics["qp_kkt_residual"][i]),
                "qp_solver_iterations": int(
                    lbpt_diagnostics["qp_solver_iterations"][i]
                ),
                "qp_solver_status": str(lbpt_diagnostics["qp_solver_status"][i]),
                "qp_solver_message": str(lbpt_diagnostics["qp_solver_message"][i]),
                "same_patient_count": int(same_patient[i].sum()),
                "same_project_count": int(same_project[i].sum()),
                "unique_neighbor_patient_n": int(unique_neighbor_n[i]),
                "weight_sum": float(weight_sums[i]),
            }
        )

    pd.DataFrame(rows).to_csv(
        os.path.join(save_dir, f"stage3_{split_tag}_patient_neighbor_audit.csv"),
        index=False,
    )

    summary = {
        "query_n": int(len(query_indices)),
        "neighbor_slots": int(neighbor_indices_global.shape[1]),
        "same_patient_occurrences": int(same_patient.sum()),
        "queries_with_same_patient": int(same_patient.any(axis=1).sum()),
        "same_project_occurrences": int(same_project.sum()),
        "same_project_ratio": float(same_project.mean()),
        "unique_neighbor_patient_n_min": int(unique_neighbor_n.min()),
        "unique_neighbor_patient_n_max": int(unique_neighbor_n.max()),
        "weight_sum_min": float(weight_sums.min()),
        "weight_sum_max": float(weight_sums.max()),
        "alpha_min": float(neighbor_weights.min()),
        "alpha_max": float(neighbor_weights.max()),
        "effective_neighbor_n_min": float(np.min(lbpt_diagnostics["effective_n"])),
        "effective_neighbor_n_median": float(
            np.median(lbpt_diagnostics["effective_n"])
        ),
        "effective_neighbor_n_max": float(np.max(lbpt_diagnostics["effective_n"])),
        "qp_kkt_residual_max": float(np.max(lbpt_diagnostics["qp_kkt_residual"])),
        "qp_failure_n": int(
            np.sum(np.asarray(lbpt_diagnostics["qp_solver_status"]) != "slsqp_success")
        ),
    }

    if summary["same_patient_occurrences"] != 0:
        raise AssertionError(f"{split_tag}: same patient remains in top-k")
    if summary["same_project_ratio"] != 1.0:
        raise AssertionError(f"{split_tag}: cross-project patient neighbor found")
    if summary["unique_neighbor_patient_n_min"] != neighbor_indices_global.shape[1]:
        raise AssertionError(
            f"{split_tag}: fewer than k distinct patient neighbors found"
        )
    if not np.allclose(weight_sums, 1.0, atol=1e-6, rtol=0.0):
        raise AssertionError(f"{split_tag}: patient neighbor weights do not sum to 1")
    if summary["alpha_min"] < -1e-9:
        raise AssertionError(f"{split_tag}: negative LBPT alpha found")
    if summary["qp_failure_n"] != 0:
        raise AssertionError(f"{split_tag}: non-successful LBPT solver status found")

    return summary


def check_patient_neighbor_same_project_ratio(
    query_indices,
    neighbor_indices_global,
    patient_projects,
    name="set",
):
    patient_projects_arr = np.asarray(patient_projects, dtype=object)

    ratios = []
    bad_count = 0

    for i, q_idx in enumerate(query_indices):
        q_project = patient_projects_arr[q_idx]
        neigh_idx = neighbor_indices_global[i]
        neigh_projects = patient_projects_arr[neigh_idx]

        same = neigh_projects == q_project
        ratio = float(np.mean(same))
        ratios.append(ratio)

        if ratio < 1.0:
            bad_count += 1

    print(f"\n[neighbor same-project check: {name}]")
    print(f"query_n = {len(query_indices)}")
    print(f"mean same-project ratio = {np.mean(ratios):.6f}")
    print(f"min same-project ratio  = {np.min(ratios):.6f}")
    print(f"project mismatch count  = {bad_count}")

    return ratios


def compute_patient_space_reconstruction_errors(
    embeddings,
    query_indices,
    neighbor_indices,
    alpha,
    normalize=True,
):
    """Audit how H&E-derived alpha reconstructs another patient space."""

    embeddings = torch.as_tensor(embeddings, dtype=torch.float32).detach().cpu()
    if normalize:
        embeddings = F.normalize(embeddings, dim=1)
    query_indices = np.asarray(query_indices, dtype=np.int64)
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    alpha = np.asarray(alpha, dtype=np.float64)
    if neighbor_indices.shape != alpha.shape:
        raise ValueError("neighbor indices and alpha shapes differ")

    l2_errors = np.zeros((len(query_indices),), dtype=np.float64)
    relative_errors = np.zeros((len(query_indices),), dtype=np.float64)
    for row, query_idx in enumerate(query_indices):
        metrics = compute_reconstruction_metrics(
            embeddings[int(query_idx)],
            embeddings[neighbor_indices[row]],
            alpha[row],
        )
        l2_errors[row] = metrics.l2_error
        relative_errors[row] = metrics.relative_l2_error
    return l2_errors, relative_errors


def compute_cell_proxy_reconstruction_errors(
    patient_aligned_embeddings,
    cell_embeddings,
    base_proxy_cell_indices,
    query_indices,
    neighbor_indices,
    alpha,
):
    """Audit transfer of H&E alpha to unconditional cell-anchor geometry."""

    patient = F.normalize(
        torch.as_tensor(patient_aligned_embeddings, dtype=torch.float32).detach().cpu(),
        dim=1,
    )
    cells = F.normalize(
        torch.as_tensor(cell_embeddings, dtype=torch.float32).detach().cpu(),
        dim=1,
    )
    base_proxy = np.asarray(base_proxy_cell_indices, dtype=np.int64)
    query_indices = np.asarray(query_indices, dtype=np.int64)
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    alpha = np.asarray(alpha, dtype=np.float64)
    if neighbor_indices.shape != alpha.shape:
        raise ValueError("neighbor indices and alpha shapes differ")

    l2_errors = np.full(len(query_indices), np.nan, dtype=np.float64)
    relative_errors = np.full(len(query_indices), np.nan, dtype=np.float64)
    for row, query_idx in enumerate(query_indices):
        proxy_indices = base_proxy[neighbor_indices[row]]
        if np.any(proxy_indices < 0):
            continue
        metrics = compute_reconstruction_metrics(
            patient[int(query_idx)],
            cells[proxy_indices],
            alpha[row],
        )
        l2_errors[row] = metrics.l2_error
        relative_errors[row] = metrics.relative_l2_error
    return l2_errors, relative_errors


# Dataset


# Model loading helpers


def resume_stage1_model_if_needed(model, config, device):
    checkpoint_path = config.get("resume_stage1_ckpt", None)

    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        raise FileNotFoundError("[resume stage1] no checkpoint specified")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"[resume stage1] checkpoint not found: {checkpoint_path}"
        )

    print("\nLoad Stage1 checkpoint")
    print(f"[resume stage1] ckpt_path = {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")

    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"[resume stage1] loaded from: {checkpoint_path}")
    print(f"[resume stage1] missing keys = {len(missing)}")
    print(f"[resume stage1] unexpected keys = {len(unexpected)}")

    if "epoch" in checkpoint:
        print(f"[resume stage1] checkpoint epoch = {checkpoint['epoch']}")

    if "metrics" in checkpoint:
        print(f"[resume stage1] metrics = {checkpoint['metrics']}")

    for k in missing[:30]:
        print("  MISSING:", k)

    for k in unexpected[:30]:
        print("  UNEXPECTED:", k)

    model = model.to(device)
    return model, checkpoint


def resume_stage2_model_if_needed(model, config, device):
    checkpoint_path = config.get("resume_stage2_ckpt", None)

    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        raise FileNotFoundError("[resume stage2] no checkpoint specified")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"[resume stage2] checkpoint not found: {checkpoint_path}"
        )

    print("\nLoad Stage2 checkpoint")
    print(f"[resume stage2] ckpt_path = {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"[resume stage2] loaded from: {checkpoint_path}")
    print(f"[resume stage2] missing keys = {len(missing)}")
    print(f"[resume stage2] unexpected keys = {len(unexpected)}")

    if "epoch" in checkpoint:
        print(f"[resume stage2] checkpoint epoch = {checkpoint['epoch']}")

    if "metrics" in checkpoint:
        print(f"[resume stage2] metrics = {checkpoint['metrics']}")

    for k in missing[:30]:
        print("  MISSING:", k)

    for k in unexpected[:30]:
        print("  UNEXPECTED:", k)

    model = model.to(device)
    return model, checkpoint


def configure_stage3_parameters(model):
    """Enable context-fusion and perturbation parameters for Stage 3 training.

    Molecular and image alignment remain fixed. The trainable perturbation
    components are the query, key and value projections and both decoders.
    """
    # 1. Freeze everything first
    for parameter in model.parameters():
        parameter.requires_grad = False

    # 2. Train patient text/image fusion module
    if (
        hasattr(model, "patient_context_fusion")
        and model.patient_context_fusion is not None
    ):
        for parameter in model.patient_context_fusion.parameters():
            parameter.requires_grad = True
    else:
        raise RuntimeError(
            "model has no patient_context_fusion. Cannot train Stage3 text fusion."
        )

    # 3. Train only UniCure Stage3 prediction heads
    if hasattr(model, "unicure_model") and model.unicure_model is not None:
        trainable_unicure_prefixes = (
            "query_map.",
            "key_map.",
            "value_map.",
            "fusion_decoder.",
            "decoder.",
        )

        for name, parameter in model.unicure_model.named_parameters():
            if name.startswith(trainable_unicure_prefixes):
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False

    # 4. Explicitly keep Stage1 frozen
    if hasattr(model, "tcga_align_head") and model.tcga_align_head is not None:
        for parameter in model.tcga_align_head.parameters():
            parameter.requires_grad = False

    # 5. Explicitly keep Stage2 frozen
    if hasattr(model, "img_align_head") and model.img_align_head is not None:
        for parameter in model.img_align_head.parameters():
            parameter.requires_grad = False

    # 6. Print trainable check
    print("\nTrainable parameter check")

    trainable = []
    frozen = []

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable.append(name)
        else:
            frozen.append(name)

    print(f"trainable tensors = {len(trainable)}")

    for name in trainable[:200]:
        print("  TRAIN:", name)

    if len(trainable) > 200:
        print(f"  ... {len(trainable) - 200} more")

    print(f"frozen tensors = {len(frozen)}")

    trainable_param_n = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen_param_n = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )

    print(f"trainable parameters = {trainable_param_n:,}")
    print(f"frozen parameters = {frozen_param_n:,}")

    # 7. Validate frozen modules
    forbidden_prefixes = (
        "tcga_align_head.",
        "img_align_head.",
        "cancer_embedding.",
        "unicure_model.uce_lora.",
    )

    bad_trainable = []

    for name, parameter in model.named_parameters():
        if parameter.requires_grad and name.startswith(forbidden_prefixes):
            bad_trainable.append(name)

    if len(bad_trainable) > 0:
        print("[ERROR] unexpected trainable parameters:")
        for name in bad_trainable[:100]:
            print("  UNEXPECTED:", name)

        raise RuntimeError(
            "Stage 1/2 alignment and the UCE backbone must remain frozen."
        )

    print("[OK] Stage3 trainable modules are correct.")


@torch.no_grad()
def build_aligned_patient_embeddings(
    model,
    patient_gene,
    device,
    batch_size=512,
    patient_indices=None,
):
    model.eval()
    outs = []

    if patient_indices is None:
        selected_gene = patient_gene
    else:
        patient_indices = np.asarray(patient_indices, dtype=np.int64)
        if np.any(patient_indices < 0) or np.any(
            patient_indices >= patient_gene.size(0)
        ):
            raise IndexError("patient_indices contains an out-of-range row")
        selected_gene = patient_gene[torch.as_tensor(patient_indices, dtype=torch.long)]

    with torch.no_grad():
        for i in range(0, selected_gene.size(0), batch_size):
            gene_batch = selected_gene[i : i + batch_size].to(device)
            aligned = model.align_tcga_to_cell(gene_batch)
            outs.append(aligned.detach().cpu())

    if not outs:
        raise ValueError("cannot align an empty patient selection")
    return torch.cat(outs, dim=0)


# Patient text embedding


def load_patient_text_embeddings(config, patient_ids):
    text_emb_path = config["patient_text_embedding_path"]
    text_index_path = config["patient_text_embedding_index_path"]

    if not os.path.exists(text_emb_path):
        raise FileNotFoundError(f"patient text embedding not found: {text_emb_path}")

    if not os.path.exists(text_index_path):
        raise FileNotFoundError(f"patient text index not found: {text_index_path}")

    patient_text_np = np.load(text_emb_path)
    text_index_df = pd.read_csv(text_index_path)

    print("[patient_text] embedding shape:", patient_text_np.shape)
    print("[patient_text] index shape:", text_index_df.shape)

    if patient_text_np.shape[1] != config["patient_text_dim"]:
        raise ValueError(
            f"patient text dim mismatch: got {patient_text_np.shape[1]}, "
            f"expected {config['patient_text_dim']}"
        )

    if "patient_id" not in text_index_df.columns:
        raise KeyError("patient text index csv must contain patient_id column")

    loaded_patient_ids = [str(x) for x in patient_ids]
    text_patient_ids = text_index_df["patient_id"].astype(str).tolist()

    if len(text_patient_ids) != patient_text_np.shape[0]:
        raise ValueError("text index rows != embedding rows")

    if text_patient_ids == loaded_patient_ids:
        print("[patient_text] order matches patient_ids")
    else:
        print("[patient_text] order does not match. Reordering by patient_id...")

        id_to_row = {patient_id: i for i, patient_id in enumerate(text_patient_ids)}
        missing = [
            patient_id
            for patient_id in loaded_patient_ids
            if patient_id not in id_to_row
        ]

        if missing:
            raise ValueError(
                f"missing patient_ids in text embedding index. First missing: {missing[:20]}"
            )

        reorder_idx = [id_to_row[patient_id] for patient_id in loaded_patient_ids]
        patient_text_np = patient_text_np[reorder_idx]

        print("[patient_text] reordered successfully")

    if patient_text_np.shape[0] != len(patient_ids):
        raise ValueError(
            f"patient_text rows {patient_text_np.shape[0]} != patient_ids {len(patient_ids)}"
        )

    if len(set(text_patient_ids)) != len(text_patient_ids):
        raise ValueError("Duplicate patient IDs in context embedding index")

    if np.isnan(patient_text_np).any():
        raise ValueError("patient_text has NaN")

    if np.isinf(patient_text_np).any():
        raise ValueError("patient_text has Inf")

    return torch.tensor(patient_text_np, dtype=torch.float32)


# Build resources


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


def build_training_resources(config):
    print("\nBuild training resources for Stage3")

    stage1_cancer_encoder = load_stage1_cancer_encoder(config)
    allowed_stage1_labels = set(stage1_cancer_encoder.keys())

    # Reliable Stage3 tumor cell labels
    clean_tumor_labels = sorted(
        [
            c
            for c in allowed_stage1_labels
            if c not in {"OTHER", "UNKNOWN", "Unknown", "UNMATCHED"}
        ]
    )

    print("\nClean tumor labels for Stage3 cell candidates")
    print(clean_tumor_labels)
    print(f"[clean tumor label n] {len(clean_tumor_labels)}")

    has_unmatched = "UNMATCHED" in allowed_stage1_labels
    if not has_unmatched:
        raise ValueError(
            "Stage1 encoder has no UNMATCHED label. "
            "Cannot keep all patients safely without fallback."
        )

    # Load patients

    patient_img_full, patient_gene_full, patient_ids_full, patient_cancers_raw_full = (
        load_patient_embeddings(
            features_output_dir=config["features_output_dir"],
            txt_file_path=config["txt_file_path"],
            patient_csv_path=config["patient_csv_path"],
            patient_uce_path=config["patient_uce_path"],
            id_mode="case",
            dedup_case=True,
        )
    )

    patient_clin_full = torch.zeros((len(patient_ids_full), 0), dtype=torch.float32)

    patient_text_full = load_patient_text_embeddings(config, patient_ids_full)

    patient_cancers_stage1_raw_full = [
        normalize_patient_cancer_stage1(x) for x in patient_cancers_raw_full
    ]

    patient_cancers_neighbor_full = [
        normalize_patient_cancer_for_neighbor(x) for x in patient_cancers_raw_full
    ]

    print("\nFull TCGA cancer counts")
    print("[Stage1 raw normalized labels]")
    print(pd.Series(patient_cancers_stage1_raw_full).value_counts().sort_index())
    print("[HE neighbor project labels]")
    print(pd.Series(patient_cancers_neighbor_full).value_counts().sort_index())

    patient_img = patient_img_full
    patient_gene = patient_gene_full
    patient_clin = patient_clin_full
    patient_text = patient_text_full
    patient_ids = list(patient_ids_full)
    patient_cancers_raw = list(patient_cancers_raw_full)
    patient_cancers_stage1_raw = list(patient_cancers_stage1_raw_full)
    patient_cancers_neighbor = list(patient_cancers_neighbor_full)

    patient_cancers_stage1 = []
    mapped_to_unmatched = []

    for c in patient_cancers_stage1_raw_full:
        if c in allowed_stage1_labels:
            patient_cancers_stage1.append(c)
        else:
            patient_cancers_stage1.append("UNMATCHED")
            mapped_to_unmatched.append(c)

    patient_cancer_idx_stage1 = np.array(
        encode_cancer_labels(patient_cancers_stage1, stage1_cancer_encoder),
        dtype=np.int64,
    )

    print("\nStage3 keeps all patients")
    print(f"[patient_n] {len(patient_ids)}")

    print("\n[Stage3 mapped cancer counts used for encoder]")
    for k, v in sorted(
        Counter(patient_cancers_stage1).items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {k}: {v}")

    if len(mapped_to_unmatched) > 0:
        print("\n[Cancers mapped to UNMATCHED]")
        for k, v in sorted(
            Counter(mapped_to_unmatched).items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {k}: {v}")

    # Load cells

    cell_emb_df_raw = pd.read_parquet(config["cell_emb_path"])
    cell_emb_df_raw.index = cell_emb_df_raw.index.astype(str)

    n_total = len(cell_emb_df_raw)
    n_unique = cell_emb_df_raw.index.nunique()
    n_dup = n_total - n_unique

    print("\nCell embedding info")
    print(f"[cell_emb] total rows = {n_total}")
    print(f"[cell_emb] unique cell_ids = {n_unique}")
    print(f"[cell_emb] duplicate rows = {n_dup}")

    cell_emb_df_for_mmd_all = cell_emb_df_raw.copy()

    if n_dup > 0:
        cell_emb_df_mean_all = cell_emb_df_raw.groupby(level=0).mean()
    else:
        cell_emb_df_mean_all = cell_emb_df_raw.copy()

    cell_ids_mmd_all = list(cell_emb_df_for_mmd_all.index)
    cell_ids_mean_all = list(cell_emb_df_mean_all.index)

    cell_cancer_map_mmd = load_cell_cancer_map(
        config["cell_cancer_path"], cell_ids_mmd_all
    )
    cell_cancer_map_mean = load_cell_cancer_map(
        config["cell_cancer_path"], cell_ids_mean_all
    )

    cell_cancers_mmd_std_full = [
        normalize_cell_cancer(cell_cancer_map_mmd[cid]) for cid in cell_ids_mmd_all
    ]

    cell_cancers_mean_std_full = [
        normalize_cell_cancer(cell_cancer_map_mean[cid]) for cid in cell_ids_mean_all
    ]

    print("\nRaw cell cancer counts")
    print("[cell MMD rows]")
    print(pd.Series(cell_cancers_mmd_std_full).value_counts().sort_index())
    print("[cell unique lines]")
    print(pd.Series(cell_cancers_mean_std_full).value_counts().sort_index())

    # Stage3 candidate pool: clean tumor only
    keep_cell_mmd_mask = np.array(
        [c in clean_tumor_labels for c in cell_cancers_mmd_std_full],
        dtype=bool,
    )

    keep_cell_mean_mask = np.array(
        [c in clean_tumor_labels for c in cell_cancers_mean_std_full],
        dtype=bool,
    )

    cell_emb_df_for_mmd = cell_emb_df_for_mmd_all.iloc[keep_cell_mmd_mask].copy()
    cell_emb_df_mean = cell_emb_df_mean_all.iloc[keep_cell_mean_mask].copy()

    cell_cancers_mmd_std = [
        c for c, keep in zip(cell_cancers_mmd_std_full, keep_cell_mmd_mask) if keep
    ]

    cell_cancers_mean_std = [
        c for c, keep in zip(cell_cancers_mean_std_full, keep_cell_mean_mask) if keep
    ]

    excluded_cell_mmd_counter = Counter(
        [
            c
            for c, keep in zip(cell_cancers_mmd_std_full, keep_cell_mmd_mask)
            if not keep
        ]
    )
    excluded_cell_mean_counter = Counter(
        [
            c
            for c, keep in zip(cell_cancers_mean_std_full, keep_cell_mean_mask)
            if not keep
        ]
    )

    print("\nClean filtered cell cancer counts")
    print("[cell MMD rows: kept]")
    print(pd.Series(cell_cancers_mmd_std).value_counts().sort_index())
    print("[cell unique lines: kept]")
    print(pd.Series(cell_cancers_mean_std).value_counts().sort_index())

    print("\n[cell MMD rows: excluded]")
    print(dict(excluded_cell_mmd_counter))
    print("[cell unique lines: excluded]")
    print(dict(excluded_cell_mean_counter))

    cell_cancer_idx_mmd = encode_cancer_labels(
        cell_cancers_mmd_std, stage1_cancer_encoder
    )
    cell_cancer_idx_mean = encode_cancer_labels(
        cell_cancers_mean_std, stage1_cancer_encoder
    )

    cell_cancer_idx_tensor_mmd = torch.tensor(cell_cancer_idx_mmd, dtype=torch.long)
    cell_cancer_idx_tensor_mean = torch.tensor(cell_cancer_idx_mean, dtype=torch.long)

    cell_emb_tensor_mmd = torch.tensor(cell_emb_df_for_mmd.values, dtype=torch.float32)
    cell_emb_tensor_mean = torch.tensor(cell_emb_df_mean.values, dtype=torch.float32)

    clean_global_cell_indices = np.arange(len(cell_emb_df_mean), dtype=np.int64)

    # Build cell centroids

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

    # Main split

    train_idx, val_idx, main_test_idx, split_payload = load_main_split_train_val_test(
        patient_ids=list(patient_ids),
        split_path=config["main_split_path"],
    )

    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    main_test_idx = np.asarray(main_test_idx, dtype=int)

    # Build Stage1 datasets

    stage1_train_dataset = build_stage1_alignment_dataset(
        patient_img,
        patient_gene,
        patient_clin,
        patient_cancer_idx_stage1,
        train_idx,
    )

    stage1_val_dataset = build_stage1_alignment_dataset(
        patient_img,
        patient_gene,
        patient_clin,
        patient_cancer_idx_stage1,
        val_idx,
    )

    # patient-patient neighbors in image embedding space
    # Same TCGA project only

    print("\nPatient-patient HE neighbors")
    print("[neighbor rule] all splits: same TCGA project, exclude same patient ID")

    train_img = patient_img[train_idx]
    val_img = patient_img[val_idx]

    patient_neighbor_project_arr = np.asarray(patient_cancers_neighbor, dtype=object)

    train_project = patient_neighbor_project_arr[train_idx]
    val_project = patient_neighbor_project_arr[val_idx]

    (
        train_patient_topk_idx_local,
        train_patient_topk_w,
        train_patient_topk_cos,
        train_patient_topk_diag,
    ) = compute_topk_neighbors_same_project_patient(
        query_emb=train_img,
        query_project=train_project,
        query_patient_ids=np.asarray(patient_ids, dtype=object)[train_idx],
        candidate_emb=train_img,
        candidate_project=train_project,
        candidate_patient_ids=np.asarray(patient_ids, dtype=object)[train_idx],
        k=config["topk_patient"],
        trace_scale=config["lbpt_trace_regularization"],
        absolute_epsilon=config["lbpt_absolute_epsilon"],
    )
    train_patient_topk_idx_global = train_idx[train_patient_topk_idx_local]
    assert not np.any(train_patient_topk_idx_global == train_idx[:, None]), (
        "train top-k contains the query patient's own global row"
    )

    (
        val_patient_topk_idx_local,
        val_patient_topk_w,
        val_patient_topk_cos,
        val_patient_topk_diag,
    ) = compute_topk_neighbors_same_project_patient(
        query_emb=val_img,
        query_project=val_project,
        query_patient_ids=np.asarray(patient_ids, dtype=object)[val_idx],
        candidate_emb=train_img,
        candidate_project=train_project,
        candidate_patient_ids=np.asarray(patient_ids, dtype=object)[train_idx],
        k=config["topk_patient"],
        trace_scale=config["lbpt_trace_regularization"],
        absolute_epsilon=config["lbpt_absolute_epsilon"],
    )
    val_patient_topk_idx_global = train_idx[val_patient_topk_idx_local]

    # Build main-test neighbors after validating the manifest and checkpoint.
    main_test_patient_topk_idx_global = None
    main_test_patient_topk_w = None
    main_test_patient_topk_cos = None
    main_test_patient_topk_diag = None

    print("\nPatient neighbor summary")
    print(
        f"[patient neighbors] train->train idx shape    = {train_patient_topk_idx_global.shape}"
    )
    print(
        f"[patient neighbors] val->train idx shape      = {val_patient_topk_idx_global.shape}"
    )
    if main_test_patient_topk_idx_global is not None:
        print(
            "[patient neighbors] main_test->train idx shape = "
            f"{main_test_patient_topk_idx_global.shape}"
        )
    else:
        print("[patient neighbors] main_test deferred to evaluation")

    check_patient_neighbor_same_project_ratio(
        query_indices=train_idx,
        neighbor_indices_global=train_patient_topk_idx_global,
        patient_projects=patient_cancers_neighbor,
        name="train",
    )

    check_patient_neighbor_same_project_ratio(
        query_indices=val_idx,
        neighbor_indices_global=val_patient_topk_idx_global,
        patient_projects=patient_cancers_neighbor,
        name="val",
    )

    # Save audits

    os.makedirs(config["save_dir"], exist_ok=True)

    with open(
        os.path.join(config["save_dir"], "stage1_cancer_encoder_used_by_stage3.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(stage1_cancer_encoder, f, ensure_ascii=False, indent=2)

    split_name_arr = np.array([""] * len(patient_ids), dtype=object)
    split_name_arr[train_idx] = "main_train"
    split_name_arr[val_idx] = "main_val"
    split_name_arr[main_test_idx] = "main_test"

    pd.DataFrame(
        {
            "patient_id": patient_ids,
            "cancer_raw": patient_cancers_raw,
            "cancer_neighbor_project": patient_cancers_neighbor,
            "cancer_stage1_raw": patient_cancers_stage1_raw,
            "cancer_stage1": patient_cancers_stage1,
            "cancer_idx": patient_cancer_idx_stage1,
            "split": split_name_arr,
        }
    ).to_csv(
        os.path.join(config["save_dir"], "stage3_patients.csv"),
        index=False,
    )

    pd.DataFrame(
        {
            "cell_id": list(cell_emb_df_mean.index),
            "cell_cancer_stage1": cell_cancers_mean_std,
            "cell_cancer_idx": cell_cancer_idx_mean,
            "included_in_clean_global": True,
        }
    ).to_csv(
        os.path.join(config["save_dir"], "stage3_clean_cell_candidates_unique.csv"),
        index=False,
    )

    cell_summary_rows = []
    for c in sorted(set(cell_cancers_mean_std_full)):
        cell_summary_rows.append(
            {
                "cell_cancer_label": c,
                "unique_cell_n_raw": Counter(cell_cancers_mean_std_full).get(c, 0),
                "mmd_rows_n_raw": Counter(cell_cancers_mmd_std_full).get(c, 0),
                "kept_for_stage3_clean_global": c in clean_tumor_labels,
            }
        )

    pd.DataFrame(cell_summary_rows).to_csv(
        os.path.join(config["save_dir"], "stage3_cell_candidate_summary.csv"),
        index=False,
    )

    with open(
        os.path.join(config["save_dir"], "stage3_patient_mapping_summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "raw_patient_n": len(patient_cancers_stage1_raw_full),
                "kept_patient_n": len(patient_ids),
                "neighbor_project_counts": dict(Counter(patient_cancers_neighbor)),
                "stage1_raw_cancer_counts": dict(Counter(patient_cancers_stage1_raw)),
                "stage3_mapped_cancer_counts": dict(Counter(patient_cancers_stage1)),
                "mapped_to_unmatched": dict(Counter(mapped_to_unmatched)),
                "clean_tumor_labels": clean_tumor_labels,
                "train_n": int(len(train_idx)),
                "val_n": int(len(val_idx)),
                "main_test_n": int(len(main_test_idx)),
                "main_split_path": config["main_split_path"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "stage1_train_dataset": stage1_train_dataset,
        "stage1_val_dataset": stage1_val_dataset,
        "cell_emb_tensor_mmd": cell_emb_tensor_mmd,
        "cell_cancer_idx_tensor_mmd": cell_cancer_idx_tensor_mmd,
        "cell_emb_tensor_mean": cell_emb_tensor_mean,
        "cell_cancer_idx_tensor_mean": cell_cancer_idx_tensor_mean,
        "cell_centroids": cell_centroids,
        "clean_global_cell_indices": clean_global_cell_indices,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "main_test_idx": main_test_idx,
        "split_payload": split_payload,
        "stage1_cancer_encoder": stage1_cancer_encoder,
        "unmatched_cancer_idx": stage1_cancer_encoder.get("UNMATCHED", None),
        "matchable_cancer_indices": [
            stage1_cancer_encoder[c]
            for c in clean_tumor_labels
            if c in stage1_cancer_encoder
        ],
        "patient_cancers_neighbor": patient_cancers_neighbor,
        "patient_cancers_stage1_raw": patient_cancers_stage1_raw,
        "patient_cancers_stage1": patient_cancers_stage1,
        "patient_cancers_raw": patient_cancers_raw,
        "patient_gene": patient_gene,
        "patient_ids": patient_ids,
        "patient_clin": patient_clin,
        "patient_img": patient_img,
        "patient_text": patient_text,
        "patient_cancer_idx_stage1": patient_cancer_idx_stage1,
        "cell_emb_df_mean": cell_emb_df_mean,
        "cell_cancers_mean_std": cell_cancers_mean_std,
        "train_patient_topk_idx": train_patient_topk_idx_global,
        "train_patient_topk_w": train_patient_topk_w,
        "train_patient_topk_cos": train_patient_topk_cos,
        "train_patient_topk_diag": train_patient_topk_diag,
        "val_patient_topk_idx": val_patient_topk_idx_global,
        "val_patient_topk_w": val_patient_topk_w,
        "val_patient_topk_cos": val_patient_topk_cos,
        "val_patient_topk_diag": val_patient_topk_diag,
        "main_test_patient_topk_idx": main_test_patient_topk_idx_global,
        "main_test_patient_topk_w": main_test_patient_topk_w,
        "main_test_patient_topk_cos": main_test_patient_topk_cos,
        "main_test_patient_topk_diag": main_test_patient_topk_diag,
    }


def build_locked_main_test_patient_neighbors(resources, config):
    """Build main-test neighbors after validating the method configuration."""

    if resources.get("main_test_patient_topk_idx") is not None:
        raise RuntimeError("main_test patient neighbors were already constructed")
    train_idx = np.asarray(resources["train_idx"], dtype=np.int64)
    main_test_idx = np.asarray(resources["main_test_idx"], dtype=np.int64)
    patient_img = resources["patient_img"]
    patient_ids = np.asarray(resources["patient_ids"], dtype=object)
    projects = np.asarray(resources["patient_cancers_neighbor"], dtype=object)

    (
        neighbor_idx_local,
        neighbor_alpha,
        neighbor_cosines,
        neighbor_diagnostics,
    ) = compute_topk_neighbors_same_project_patient(
        query_emb=patient_img[main_test_idx],
        query_project=projects[main_test_idx],
        query_patient_ids=patient_ids[main_test_idx],
        candidate_emb=patient_img[train_idx],
        candidate_project=projects[train_idx],
        candidate_patient_ids=patient_ids[train_idx],
        k=config["topk_patient"],
        trace_scale=config["lbpt_trace_regularization"],
        absolute_epsilon=config["lbpt_absolute_epsilon"],
    )
    neighbor_idx_global = train_idx[neighbor_idx_local]
    check_patient_neighbor_same_project_ratio(
        query_indices=main_test_idx,
        neighbor_indices_global=neighbor_idx_global,
        patient_projects=resources["patient_cancers_neighbor"],
        name="main_test",
    )
    resources["main_test_patient_topk_idx"] = neighbor_idx_global
    resources["main_test_patient_topk_w"] = neighbor_alpha
    resources["main_test_patient_topk_cos"] = neighbor_cosines
    resources["main_test_patient_topk_diag"] = neighbor_diagnostics
    print(
        f"[patient neighbors] main_test->train idx shape = {neighbor_idx_global.shape}"
    )


# Resume Stage3


def load_stage3_history_for_resume(save_dir, start_epoch):
    history_path = os.path.join(save_dir, "stage3_text_fusion_unicure_history.csv")

    history_keys = [
        "epoch",
        "train_loss",
        "train_mse",
        "train_cos",
        "train_img_align",
        "train_img_cos",
        "train_pearson",
        "val_loss",
        "val_mse",
        "val_cos",
        "val_img_align",
        "val_img_cos",
        "val_pearson",
    ]

    if not os.path.exists(history_path):
        print(f"[resume stage3] history csv not found: {history_path}")
        return None

    history_table = pd.read_csv(history_path)

    if "epoch" in history_table.columns and start_epoch is not None:
        history_table = history_table[history_table["epoch"] <= int(start_epoch)].copy()

    history = {}

    for key in history_keys:
        history[key] = (
            history_table[key].tolist() if key in history_table.columns else []
        )

    print(f"[resume stage3] loaded history csv: {history_path}")
    print(f"[resume stage3] restored history rows = {len(history['epoch'])}")

    return history


def load_stage3_checkpoint_for_resume(model, config, device):
    ckpt_type = str(config.get("resume_stage3_ckpt_type", "last")).strip().lower()

    if ckpt_type not in {"last", "best"}:
        raise ValueError(
            f"resume_stage3_ckpt_type must be 'last' or 'best', got: {ckpt_type}"
        )

    ckpt_path_key = f"resume_stage3_{ckpt_type}_ckpt"
    checkpoint_path = config.get(ckpt_path_key, None)

    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        raise FileNotFoundError("[resume stage3] no checkpoint specified")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"[resume stage3] checkpoint not found: {checkpoint_path}"
        )

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "[resume stage3] checkpoint must contain model_state_dict"
        )
    expected_method_sha = str(config.get("method_definition_sha256", "")).strip()
    checkpoint_method_sha = str(
        checkpoint.get("config", {}).get("method_definition_sha256", "")
    ).strip()
    if not expected_method_sha:
        raise RuntimeError("[resume stage3] method_definition_sha256 is required")
    if checkpoint_method_sha != expected_method_sha:
        raise RuntimeError(
            "[resume stage3] checkpoint method configuration mismatch"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    start_epoch = int(checkpoint.get("epoch", 0))
    best_val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", None))
    best_epoch = checkpoint.get("best_epoch", start_epoch)
    patience_counter = int(checkpoint.get("patience_counter", 0))
    optimizer_state_dict = checkpoint.get("optimizer_state_dict", None)
    scaler_state_dict = checkpoint.get("scaler_state_dict", None)

    print(f"[resume stage3] ckpt_type = {ckpt_type}")
    print(f"[resume stage3] loaded checkpoint from: {checkpoint_path}")
    print(f"[resume stage3] checkpoint epoch = {start_epoch}")
    print("[resume stage3] strict model-state validation passed")

    history = load_stage3_history_for_resume(config["save_dir"], start_epoch)

    model = model.to(device)

    return model, {
        "start_epoch": start_epoch,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "patience_counter": patience_counter,
        "optimizer_state_dict": optimizer_state_dict,
        "scaler_state_dict": scaler_state_dict,
        "history": history,
    }


# Stage3 training


def fit_perturbation_model(
    model,
    train_dataset,
    val_dataset,
    config,
    device,
    start_epoch=0,
    init_best_val_loss=None,
    init_best_epoch=None,
    init_patience_counter=0,
    optimizer_state_dict=None,
    scaler_state_dict=None,
    init_history=None,
):
    """Optimize context fusion and perturbation prediction on training and validation records."""
    print("\nStage 3: contextual perturbation prediction")

    num_workers = int(config.get("stage3_num_workers", 8))
    prefetch_factor = int(config.get("stage3_prefetch_factor", 4))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["stage3_batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["stage3_batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    model.to(device)

    configure_stage3_parameters(model)

    optimizer = make_optimizer(
        model,
        lr=config["stage3_lr"],
        weight_decay=1e-4,
    )

    use_amp = bool(config.get("stage3_use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

        print("[Stage3 resume] optimizer_state_dict loaded")

    if scaler_state_dict is not None and use_amp:
        scaler.load_state_dict(scaler_state_dict)
        print("[Stage3 resume] scaler_state_dict loaded")

    best_val_loss = (
        float("inf") if init_best_val_loss is None else float(init_best_val_loss)
    )
    best_epoch = int(init_best_epoch) if init_best_epoch is not None else -1
    patience_counter = int(init_patience_counter or 0)

    history_keys = [
        "epoch",
        "train_loss",
        "train_mse",
        "train_cos",
        "train_img_align",
        "train_img_cos",
        "train_pearson",
        "val_loss",
        "val_mse",
        "val_cos",
        "val_img_align",
        "val_img_cos",
        "val_pearson",
    ]

    history = {key: [] for key in history_keys}

    if init_history is not None:
        for key in history_keys:
            history[key] = list(init_history.get(key, []))

    patience = config.get("stage3_early_stop_patience", 20)
    min_delta = config.get("stage3_min_delta", 1e-4)

    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    history_path = os.path.join(save_dir, "stage3_text_fusion_unicure_history.csv")
    last_ckpt_path = os.path.join(save_dir, "stage3_text_fusion_unicure_last.pt")
    best_ckpt_path = os.path.join(save_dir, "stage3_text_fusion_unicure_best.pt")

    print(f"[Stage3] use_amp = {use_amp}")
    print(f"[Stage3] batch_size = {config['stage3_batch_size']}")
    print(f"[Stage3] num_workers = {num_workers}")
    print(f"[Stage3] prefetch_factor = {prefetch_factor}")
    print(f"[Stage3] start_epoch = {start_epoch}")
    print(f"[Stage3] best_val_loss = {best_val_loss}")
    print(f"[Stage3] best_epoch = {best_epoch}")
    print(f"[Stage3] patience_counter = {patience_counter}/{patience}")

    if start_epoch >= config["stage3_epochs"]:
        print("[Stage3] start_epoch >= stage3_epochs, skip training.")
        return history

    for epoch in range(start_epoch, config["stage3_epochs"]):
        model.train()

        # Frozen Stage1 must stay eval
        if hasattr(model, "tcga_align_head") and model.tcga_align_head is not None:
            model.tcga_align_head.eval()

        # Frozen Stage2 must stay eval
        if hasattr(model, "img_align_head") and model.img_align_head is not None:
            model.img_align_head.eval()

        # UniCure head/decoder train, but UCE backbone eval
        if hasattr(model, "unicure_model") and model.unicure_model is not None:
            model.unicure_model.train()

            if (
                hasattr(model.unicure_model, "uce_lora")
                and model.unicure_model.uce_lora is not None
            ):
                model.unicure_model.uce_lora.eval()

        # Stage3 fusion module train
        if (
            hasattr(model, "patient_context_fusion")
            and model.patient_context_fusion is not None
        ):
            model.patient_context_fusion.train()

        train_loss_sum = 0.0
        train_mse_sum = 0.0
        train_cos_sum = 0.0
        train_img_align_sum = 0.0
        train_img_cos_sum = 0.0
        train_pearson_sum = 0.0
        n_train = 0

        loop = tqdm(
            train_loader,
            desc=f"Stage3 TextFusion Train {epoch + 1}",
            dynamic_ncols=True,
        )

        for batch in loop:
            (
                image_embeddings,
                gene_embeddings,
                clin,
                context_embeddings,
                cancer_idx,
                drug_embeddings,
                delta_target,
                control_target,
            ) = batch

            image_embeddings = image_embeddings.to(device, non_blocking=True)
            gene_embeddings = gene_embeddings.to(device, non_blocking=True)
            context_embeddings = context_embeddings.to(device, non_blocking=True)
            drug_embeddings = drug_embeddings.to(device, non_blocking=True)
            delta_target = delta_target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                losses = model.stage23_text_fusion_losses(
                    img=image_embeddings,
                    gene=gene_embeddings,
                    text_emb=context_embeddings,
                    drug_emb=drug_embeddings,
                    delta_target=delta_target,
                    lambda_mse=1.0,
                    lambda_cos=0.5,
                    lambda_img_align=0.1,
                    lambda_img_cos=0.05,
                )

                loss = losses["loss"]
                pred_delta = losses["pred_delta"]
                loss_mse = losses["loss_mse"]
                loss_cos = losses["loss_cos"]
                loss_img_align = losses["loss_img_align"]
                loss_img_cos = losses["loss_img_cos"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if config.get("compute_train_pearson", False):
                batch_pearson = compute_batch_pearson(pred_delta, delta_target)
            else:
                batch_pearson = 0.0

            train_loss_sum += float(loss.item())
            train_mse_sum += float(loss_mse.item())
            train_cos_sum += float(loss_cos.item())
            train_img_align_sum += float(loss_img_align.item())
            train_img_cos_sum += float(loss_img_cos.item())
            train_pearson_sum += float(batch_pearson)
            n_train += 1

        avg_train_loss = train_loss_sum / max(n_train, 1)
        avg_train_mse = train_mse_sum / max(n_train, 1)
        avg_train_cos = train_cos_sum / max(n_train, 1)
        avg_train_img_align = train_img_align_sum / max(n_train, 1)
        avg_train_img_cos = train_img_cos_sum / max(n_train, 1)
        avg_train_pearson = train_pearson_sum / max(n_train, 1)

        model.eval()

        if model.unicure_model is not None:
            model.unicure_model.eval()

        val_loss_sum = 0.0
        val_mse_sum = 0.0
        val_cos_sum = 0.0
        val_img_align_sum = 0.0
        val_img_cos_sum = 0.0
        val_pearson_sum = 0.0
        n_val = 0

        with torch.no_grad():
            loop_val = tqdm(
                val_loader,
                desc=f"Stage3 TextFusion Val {epoch + 1}",
                dynamic_ncols=True,
            )

            for batch in loop_val:
                (
                    image_embeddings,
                    gene_embeddings,
                    clin,
                    context_embeddings,
                    cancer_idx,
                    drug_embeddings,
                    delta_target,
                    control_target,
                ) = batch

                image_embeddings = image_embeddings.to(device, non_blocking=True)
                gene_embeddings = gene_embeddings.to(device, non_blocking=True)
                context_embeddings = context_embeddings.to(device, non_blocking=True)
                drug_embeddings = drug_embeddings.to(device, non_blocking=True)
                delta_target = delta_target.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    losses = model.stage23_text_fusion_losses(
                        img=image_embeddings,
                        gene=gene_embeddings,
                        text_emb=context_embeddings,
                        drug_emb=drug_embeddings,
                        delta_target=delta_target,
                        lambda_mse=1.0,
                        lambda_cos=0.5,
                        lambda_img_align=0.1,
                        lambda_img_cos=0.05,
                    )

                    loss = losses["loss"]
                    pred_delta = losses["pred_delta"]
                    loss_mse = losses["loss_mse"]
                    loss_cos = losses["loss_cos"]
                    loss_img_align = losses["loss_img_align"]
                    loss_img_cos = losses["loss_img_cos"]

                if config.get("compute_val_pearson", False):
                    batch_pearson = compute_batch_pearson(pred_delta, delta_target)
                else:
                    batch_pearson = 0.0

                val_loss_sum += float(loss.item())
                val_mse_sum += float(loss_mse.item())
                val_cos_sum += float(loss_cos.item())
                val_img_align_sum += float(loss_img_align.item())
                val_img_cos_sum += float(loss_img_cos.item())
                val_pearson_sum += float(batch_pearson)
                n_val += 1

        avg_val_loss = val_loss_sum / max(n_val, 1)
        avg_val_mse = val_mse_sum / max(n_val, 1)
        avg_val_cos = val_cos_sum / max(n_val, 1)
        avg_val_img_align = val_img_align_sum / max(n_val, 1)
        avg_val_img_cos = val_img_cos_sum / max(n_val, 1)
        avg_val_pearson = val_pearson_sum / max(n_val, 1)

        print(
            f"[Stage3 TextFusion Epoch {epoch + 1}] "
            f"train_loss={avg_train_loss:.6f} "
            f"train_mse={avg_train_mse:.6f} "
            f"train_cos={avg_train_cos:.6f} "
            f"train_img_align={avg_train_img_align:.6f} "
            f"train_img_cos={avg_train_img_cos:.6f} "
            f"train_pcc={avg_train_pearson:.6f} | "
            f"val_loss={avg_val_loss:.6f} "
            f"val_mse={avg_val_mse:.6f} "
            f"val_cos={avg_val_cos:.6f} "
            f"val_img_align={avg_val_img_align:.6f} "
            f"val_img_cos={avg_val_img_cos:.6f} "
            f"val_pcc={avg_val_pearson:.6f}"
        )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["train_mse"].append(avg_train_mse)
        history["train_cos"].append(avg_train_cos)
        history["train_img_align"].append(avg_train_img_align)
        history["train_img_cos"].append(avg_train_img_cos)
        history["train_pearson"].append(avg_train_pearson)
        history["val_loss"].append(avg_val_loss)
        history["val_mse"].append(avg_val_mse)
        history["val_cos"].append(avg_val_cos)
        history["val_img_align"].append(avg_val_img_align)
        history["val_img_cos"].append(avg_val_img_cos)
        history["val_pearson"].append(avg_val_pearson)

        pd.DataFrame(history).to_csv(history_path, index=False)

        improved = (best_val_loss - avg_val_loss) > min_delta

        if improved:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if use_amp else None,
                    "epoch": epoch + 1,
                    "val_loss": avg_val_loss,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "patience_counter": patience_counter,
                    "config": config,
                },
                best_ckpt_path,
            )

            print(
                f"[Stage3 TextFusion] new best val_loss = "
                f"{best_val_loss:.6f} at epoch {best_epoch}"
            )
        else:
            patience_counter += 1

            print(
                f"[Stage3 TextFusion] no improvement. "
                f"best_val_loss={best_val_loss:.6f} at epoch {best_epoch}. "
                f"patience {patience_counter}/{patience}"
            )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "patience_counter": patience_counter,
                "config": config,
            },
            last_ckpt_path,
        )

        if patience_counter >= patience:
            print(f"[Stage3 TextFusion] Early stopping at epoch {epoch + 1}")
            break

    return history


def plot_stage3_text_fusion_history(save_dir):
    hist_path = os.path.join(save_dir, "stage3_text_fusion_unicure_history.csv")

    if not os.path.exists(hist_path):
        print(f"[plot] history not found: {hist_path}")
        return

    history_table = pd.read_csv(hist_path)

    plot_specs = [
        (
            "train_loss",
            "val_loss",
            "Stage3 TextFusion Total Loss",
            "Loss",
            "stage3_text_fusion_total_loss.png",
        ),
        (
            "train_mse",
            "val_mse",
            "Stage3 TextFusion Delta MSE",
            "MSE",
            "stage3_text_fusion_delta_mse.png",
        ),
        (
            "train_cos",
            "val_cos",
            "Stage3 TextFusion Cosine Loss",
            "Cosine Loss",
            "stage3_text_fusion_cos_loss.png",
        ),
        (
            "train_pearson",
            "val_pearson",
            "Stage3 TextFusion Pearson",
            "Pearson",
            "stage3_text_fusion_pearson.png",
        ),
        (
            "train_img_align",
            "val_img_align",
            "Stage3 TextFusion Image Align Loss",
            "MSE Loss",
            "stage3_text_fusion_img_align_loss.png",
        ),
        (
            "train_img_cos",
            "val_img_cos",
            "Stage3 TextFusion Image Cosine Loss",
            "Cosine Loss",
            "stage3_text_fusion_img_cos_loss.png",
        ),
    ]

    for train_col, val_col, title, ylabel, filename in plot_specs:
        if (
            train_col not in history_table.columns
            or val_col not in history_table.columns
        ):
            continue

        plt.figure(figsize=(8, 5))
        plt.plot(history_table["epoch"], history_table[train_col], label=train_col)
        plt.plot(history_table["epoch"], history_table[val_col], label=val_col)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=200)
        plt.close()


# H&E-LBPT condition-specific integration


def _json_sha256(payload):
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_values_sha256(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _rounded_array_sha256(values, decimals=6):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.floating):
        array = np.round(array.astype(np.float64), decimals=decimals)
    return hash_numpy_array(array)


def _atomic_json_dump(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _he_lbpt_method_definition(config, resources, condition_store):
    train_val_idx = np.concatenate(
        [resources["train_idx"], resources["val_idx"]]
    ).astype(np.int64)
    artifact_manifest = source_file_manifest(
        [
            config["resume_stage1_ckpt"],
            config["resume_stage2_ckpt"],
            config["main_split_path"],
            config["txt_file_path"],
            config["patient_csv_path"],
            config["patient_uce_path"],
            config["patient_text_embedding_path"],
            config["patient_text_embedding_index_path"],
            config["cell_emb_path"],
            config["cell_cancer_path"],
        ]
    )
    code_dir = Path(__file__).resolve().parent
    implementation_manifest = source_file_manifest(sorted(code_dir.glob("*.py")))
    payload = {
        "method_version": str(config["pseudo_target_cache_version"]),
        "initialization": "random_perturbation_network",
        "model_spec": config["model_spec"],
        "perturb_hidden_dim": int(config["perturb_hidden_dim"]),
        "seed": int(config["seed"]),
        "patient_neighbor_rule": str(config["patient_neighbor_rule"]),
        "patient_neighbor_weight_mode": str(config["patient_neighbor_weight_mode"]),
        "topk_patient": int(config["topk_patient"]),
        "topk_cell": int(config["topk_cell"]),
        "lbpt_trace_regularization": float(config["lbpt_trace_regularization"]),
        "lbpt_absolute_epsilon": float(config["lbpt_absolute_epsilon"]),
        "self_cell_weight": float(config["self_cell_weight"]),
        "condition_universe_mode": str(config["condition_universe_mode"]),
        "proxy_assignment_mode": str(config["proxy_assignment_mode"]),
        "fixed_proxy_missing_policy": str(config["fixed_proxy_missing_policy"]),
        "zero_available_weight_policy": str(config["zero_available_weight_policy"]),
        "strict_same_cancer_cell_matching": bool(
            config["strict_same_cancer_cell_matching"]
        ),
        "allow_cross_cancer_cell_fallback": bool(
            config["allow_cross_cancer_cell_fallback"]
        ),
        "condition_store_schema_version": int(config["condition_store_schema_version"]),
        "record_cache_schema_version": int(config["record_cache_schema_version"]),
        "stage3_output_dim": int(config["stage3_output_dim"]),
        "use_patient_text": bool(config["use_patient_text"]),
        "patient_text_dim": int(config["patient_text_dim"]),
        "stage3_batch_size": int(config["stage3_batch_size"]),
        "stage3_lr": float(config["stage3_lr"]),
        "stage3_epochs": int(config["stage3_epochs"]),
        "stage3_early_stop_patience": int(config["stage3_early_stop_patience"]),
        "stage3_min_delta": float(config["stage3_min_delta"]),
        "train_val_patient_ids_sha256": _text_values_sha256(
            np.asarray(resources["patient_ids"], dtype=object)[train_val_idx]
        ),
        "train_val_patient_img_sha256": _rounded_array_sha256(
            resources["patient_img"][train_val_idx]
        ),
        "train_val_patient_gene_sha256": _rounded_array_sha256(
            resources["patient_gene"][train_val_idx]
        ),
        "train_val_patient_text_sha256": _rounded_array_sha256(
            resources["patient_text"][train_val_idx]
        ),
        "cell_embedding_sha256": _rounded_array_sha256(
            resources["cell_emb_tensor_mean"]
        ),
        "condition_store_fingerprint": str(condition_store.store_fingerprint),
        "condition_source_manifest": condition_store.source_manifest,
        "artifact_manifest": artifact_manifest,
        "implementation_manifest": implementation_manifest,
    }
    return payload, _json_sha256(payload)


def _he_lbpt_record_metadata(
    *,
    config,
    split_name,
    query_indices,
    patient_ids,
    neighbor_indices,
    alpha,
    base_proxy_cell_indices,
    base_proxy_cosines,
    active_aligned_embeddings,
    cell_embeddings,
    condition_store,
    condition_index,
    method_definition_sha256,
):
    query_indices = np.asarray(query_indices, dtype=np.int64)
    participant_indices = np.unique(
        np.concatenate([query_indices, np.asarray(neighbor_indices).reshape(-1)])
    )
    return {
        "record_schema_version": int(config["record_cache_schema_version"]),
        "method_version": str(config["pseudo_target_cache_version"]),
        "initialization": "random_perturbation_network",
        "model_spec": config["model_spec"],
        "perturb_hidden_dim": int(config["perturb_hidden_dim"]),
        "method_definition_sha256": str(method_definition_sha256),
        "split": str(split_name),
        "query_n": int(len(query_indices)),
        "query_indices_sha256": hash_numpy_array(query_indices, np.int64),
        "query_patient_ids_sha256": _text_values_sha256(
            np.asarray(patient_ids, dtype=object)[query_indices]
        ),
        "neighbor_indices_sha256": hash_numpy_array(neighbor_indices, np.int64),
        "alpha_sha256": _rounded_array_sha256(alpha, decimals=9),
        "participant_base_proxy_sha256": hash_numpy_array(
            np.asarray(base_proxy_cell_indices, dtype=np.int32)[participant_indices],
            np.int32,
        ),
        "participant_base_proxy_cosine_sha256": _rounded_array_sha256(
            np.asarray(base_proxy_cosines, dtype=np.float32)[participant_indices],
            decimals=7,
        ),
        "participant_aligned_embedding_sha256": _rounded_array_sha256(
            active_aligned_embeddings[participant_indices], decimals=6
        ),
        "cell_aligned_embedding_sha256": _rounded_array_sha256(
            cell_embeddings, decimals=6
        ),
        "condition_store_fingerprint": str(condition_store.store_fingerprint),
        "condition_index_sha256": str(condition_index.definition_sha256()),
        "condition_universe_mode": str(config["condition_universe_mode"]),
        "proxy_assignment_mode": str(config["proxy_assignment_mode"]),
        "fixed_proxy_missing_policy": str(config["fixed_proxy_missing_policy"]),
        "zero_available_weight_policy": str(config["zero_available_weight_policy"]),
        "self_cell_weight": float(config["self_cell_weight"]),
        "topk_patient": int(config["topk_patient"]),
        "lbpt_trace_regularization": float(config["lbpt_trace_regularization"]),
        "lbpt_absolute_epsilon": float(config["lbpt_absolute_epsilon"]),
    }


def _save_condition_dataset_audit(dataset, save_dir, split_name, max_rows):
    audit_indices = deterministic_audit_indices(len(dataset.records), max_rows)
    rows = [dataset.audit_record(int(index)) for index in audit_indices]
    audit_path = os.path.join(
        save_dir,
        f"stage3_{split_name}_condition_proxy_audit.csv",
    )
    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(audit_path, index=False)
    if len(audit_df) == 0:
        raise RuntimeError(f"{split_name}: condition proxy audit is empty")
    if not audit_df["delta_is_finite"].all():
        raise AssertionError(f"{split_name}: non-finite pseudo delta found")
    if not np.allclose(
        audit_df["final_weight_sum"].to_numpy(dtype=float),
        1.0,
        atol=1e-6,
        rtol=0.0,
    ):
        raise AssertionError(f"{split_name}: final proxy weights do not sum to 1")
    if (audit_df["candidate_n"] < 1).any():
        raise AssertionError(f"{split_name}: empty condition candidate pool found")
    if not audit_df["selected_is_fixed_proxy"].all():
        raise AssertionError(f"{split_name}: a non-dictionary proxy was selected")
    if (audit_df["available_role_n"] < 1).any():
        raise AssertionError(f"{split_name}: record has no available fixed proxy")
    summary = {
        "audit_path": audit_path,
        "audited_record_n": int(len(audit_df)),
        "candidate_n_min": int(audit_df["candidate_n"].min()),
        "candidate_n_median": float(audit_df["candidate_n"].median()),
        "candidate_n_max": int(audit_df["candidate_n"].max()),
        "proxy_collision_record_n": int(audit_df["proxy_collision"].sum()),
        "single_proxy_record_n": int(audit_df["single_proxy"].sum()),
        "zero_mass_fallback_record_n": int(audit_df["zero_mass_fallback"].sum()),
        "available_role_n_min": int(audit_df["available_role_n"].min()),
        "available_role_n_median": float(audit_df["available_role_n"].median()),
        "available_role_n_max": int(audit_df["available_role_n"].max()),
        "available_weight_mass_min": float(audit_df["available_weight_mass"].min()),
        "available_weight_mass_median": float(
            audit_df["available_weight_mass"].median()
        ),
        "unique_proxy_n_min": int(audit_df["unique_proxy_n"].min()),
        "unique_proxy_n_max": int(audit_df["unique_proxy_n"].max()),
        "delta_l2_norm_min": float(audit_df["delta_l2_norm"].min()),
        "delta_l2_norm_median": float(audit_df["delta_l2_norm"].median()),
        "delta_l2_norm_max": float(audit_df["delta_l2_norm"].max()),
    }
    return summary


def _load_locked_preflight_manifest(config, expected_method_sha256):
    manifest_path = (
        Path(config["save_dir"]) / "preflight_manifest_he_lbpt_fixed_gate_sweep_v1.json"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Preflight manifest not found: "
            f"{manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "passed":
        raise RuntimeError("preflight manifest does not have status='passed'")
    if manifest.get("method_definition_sha256") != expected_method_sha256:
        raise RuntimeError(
            "Method configuration differs from the preflight manifest"
        )
    if manifest.get("main_test_touched") is not False:
        raise RuntimeError("Preflight requires main_test_touched=False")
    return manifest


def prepare_he_lbpt_condition_datasets(model, resources, config, device):
    if int(config["condition_store_schema_version"]) != STORE_SCHEMA_VERSION:
        raise ValueError("configured condition-store schema does not match the code")
    if int(config["record_cache_schema_version"]) != RECORD_SCHEMA_VERSION:
        raise ValueError("configured record-cache schema does not match the code")
    if config.get("allow_cross_cancer_cell_fallback", False):
        raise ValueError("Cross-cancer fallback is incompatible with fixed same-cancer proxies")
    if not config.get("strict_same_cancer_cell_matching", False):
        raise ValueError("fixed-proxy H&E-LBPT v2 requires strict same-cancer proxies")
    if not config.get("fixed_proxy_pseudo_targets", False):
        raise ValueError("fixed-proxy pseudo targets must be enabled")
    if config.get("condition_universe_mode") != "fixed_proxy_union":
        raise ValueError(
            "Expected condition_universe_mode='fixed_proxy_union'"
        )
    if config.get("proxy_assignment_mode") != (
        "fixed_same_cancer_target_plus_neighbors_v2"
    ):
        raise ValueError("unexpected fixed-proxy assignment mode")
    if config.get("fixed_proxy_missing_policy") != ("available_roles_renormalize"):
        raise ValueError("unexpected fixed-proxy missing-condition policy")
    if config.get("zero_available_weight_policy") != ("uniform_available_roles"):
        raise ValueError("unexpected zero-weight policy")
    if config.get("final_test_mode", False) and config.get("run_stage3", False):
        raise ValueError("Set run_stage3=False for final-test evaluation")
    if config.get("preflight_only", False) and config.get("final_test_mode", False):
        raise ValueError("preflight and final-test modes are mutually exclusive")

    condition_store_path = os.path.join(
        config["save_dir"],
        "condition_store_he_lbpt_fixed_gate_sweep_v1.pt",
    )
    condition_store = load_or_build_condition_store(
        cache_path=condition_store_path,
        cell_ids=list(resources["cell_emb_df_mean"].index),
        cell_cancers=resources["cell_cancers_mean_std"],
        treated_path=config["cell_drug_dose_expr_parquet"],
        control_path=config["cell_control_parquet"],
        drug_embedding_path=config["drug_embedding_path"],
        force_rebuild=bool(config.get("force_rebuild_condition_store", False)),
    )
    config["gene_names"] = list(condition_store.gene_names)
    if condition_store.output_dim != int(config["stage3_output_dim"]):
        raise ValueError(
            "condition expression output dimension differs from Stage3: "
            f"{condition_store.output_dim} vs {config['stage3_output_dim']}"
        )
    if tuple(condition_store.cell_ids) != tuple(
        map(str, resources["cell_emb_df_mean"].index)
    ):
        raise AssertionError("condition store cell order differs from cell embeddings")

    method_definition, method_sha256 = _he_lbpt_method_definition(
        config,
        resources,
        condition_store,
    )
    locked_manifest = None
    if config.get("require_preflight_manifest", True) and not config.get(
        "preflight_only", False
    ):
        locked_manifest = _load_locked_preflight_manifest(config, method_sha256)

    if config.get("final_test_mode", False):
        best_checkpoint_path = config["resume_stage3_best_ckpt"]
        if not os.path.exists(best_checkpoint_path):
            raise FileNotFoundError(
                f"Best Stage 3 checkpoint not found: {best_checkpoint_path}"
            )
        best_checkpoint = load_checkpoint(best_checkpoint_path, map_location="cpu")
        if not isinstance(best_checkpoint, dict):
            raise RuntimeError("best Stage3 checkpoint is not a structured checkpoint")
        checkpoint_method_sha = best_checkpoint.get("config", {}).get(
            "method_definition_sha256"
        )
        if checkpoint_method_sha != method_sha256:
            raise RuntimeError(
                "Best Stage 3 checkpoint method configuration mismatch"
            )
        del best_checkpoint
        build_locked_main_test_patient_neighbors(resources, config)
        split_names = ["main_test"]
        active_indices = np.unique(
            np.concatenate([resources["train_idx"], resources["main_test_idx"]])
        )
    else:
        split_names = ["main_train", "main_val"]
        active_indices = np.unique(
            np.concatenate([resources["train_idx"], resources["val_idx"]])
        )

    print("\nBuild active patient aligned embeddings")
    print("[active split patients]", len(active_indices))
    aligned_active = build_aligned_patient_embeddings(
        model=model,
        patient_gene=resources["patient_gene"],
        device=device,
        batch_size=512,
        patient_indices=active_indices,
    )
    patient_aligned = torch.zeros(
        (len(resources["patient_ids"]), aligned_active.shape[1]),
        dtype=torch.float32,
    )
    patient_aligned[:, 0] = 1.0
    patient_aligned[torch.as_tensor(active_indices, dtype=torch.long)] = aligned_active
    print("[active aligned patient emb] shape =", aligned_active.shape)

    condition_index = condition_index_from_lincs_store(condition_store)
    base_proxy_cell_indices, base_proxy_cosines = strict_same_cancer_base_proxy(
        patient_embeddings=patient_aligned,
        patient_cancers=resources["patient_cancers_stage1"],
        cell_embeddings=resources["cell_emb_tensor_mean"],
        store=condition_store,
        active_patient_indices=active_indices,
    )
    drug_embedding_df = pd.read_parquet(config["drug_embedding_path"])
    drug_embedding_df.index = drug_embedding_df.index.astype(str)
    datasets = {}
    split_summaries = {}
    patient_audit_summaries = {}
    condition_audit_summaries = {}

    split_spec = {
        "main_train": (
            resources["train_idx"],
            resources["train_patient_topk_idx"],
            resources["train_patient_topk_w"],
            resources["train_patient_topk_cos"],
            resources["train_patient_topk_diag"],
            config.get("stage3_max_train_records"),
        ),
        "main_val": (
            resources["val_idx"],
            resources["val_patient_topk_idx"],
            resources["val_patient_topk_w"],
            resources["val_patient_topk_cos"],
            resources["val_patient_topk_diag"],
            config.get("stage3_max_val_records"),
        ),
        "main_test": (
            resources["main_test_idx"],
            resources["main_test_patient_topk_idx"],
            resources["main_test_patient_topk_w"],
            resources["main_test_patient_topk_cos"],
            resources["main_test_patient_topk_diag"],
            config.get("stage3_max_test_records"),
        ),
    }

    for split_name in split_names:
        (
            query_indices,
            neighbor_indices,
            alpha,
            neighbor_cosines,
            lbpt_diagnostics,
            record_limit,
        ) = split_spec[split_name]
        if neighbor_indices is None:
            raise RuntimeError(f"{split_name}: patient neighbors were not built")
        record_metadata = _he_lbpt_record_metadata(
            config=config,
            split_name=split_name,
            query_indices=query_indices,
            patient_ids=resources["patient_ids"],
            neighbor_indices=neighbor_indices,
            alpha=alpha,
            base_proxy_cell_indices=base_proxy_cell_indices,
            base_proxy_cosines=base_proxy_cosines,
            active_aligned_embeddings=patient_aligned,
            cell_embeddings=resources["cell_emb_tensor_mean"],
            condition_store=condition_store,
            condition_index=condition_index,
            method_definition_sha256=method_sha256,
        )
        records, record_summary = load_or_build_split_records(
            cache_path=os.path.join(
                config["save_dir"],
                f"condition_records_{split_name}_he_lbpt_fixed_gate_sweep_v1.pt",
            ),
            expected_metadata=record_metadata,
            build_kwargs={
                "query_patient_indices": query_indices,
                "patient_neighbor_indices": neighbor_indices,
                "patient_cancers": resources["patient_cancers_stage1"],
                "base_proxy_cell_indices": base_proxy_cell_indices,
                "store": condition_store,
                "universe_mode": config["condition_universe_mode"],
                "patient_alpha": alpha,
                "self_cell_weight": config["self_cell_weight"],
            },
            force_rebuild=bool(config.get("force_rebuild_record_caches", False)),
        )
        dataset = ConditionSpecificPatientDrugDoseDataset(
            patient_img=resources["patient_img"],
            patient_gene=resources["patient_gene"],
            patient_ids=resources["patient_ids"],
            patient_cancer_idx=resources["patient_cancer_idx_stage1"],
            patient_cancers=resources["patient_cancers_stage1"],
            patient_clin=resources["patient_clin"],
            patient_text=resources["patient_text"],
            patient_text_dim=config["patient_text_dim"],
            query_patient_indices=query_indices,
            patient_neighbor_indices=neighbor_indices,
            patient_alpha=alpha,
            base_proxy_cell_indices=base_proxy_cell_indices,
            base_proxy_cosines=base_proxy_cosines,
            store=condition_store,
            records=records,
            drug_embedding_path=config["drug_embedding_path"],
            drug_embedding_df=drug_embedding_df,
            self_cell_weight=config["self_cell_weight"],
            drug_target_dim=528,
            record_limit=record_limit,
        )
        if len(dataset) == 0:
            raise RuntimeError(f"{split_name}: condition-specific dataset is empty")
        datasets[split_name] = dataset

        gene_l2, gene_relative = compute_patient_space_reconstruction_errors(
            patient_aligned,
            query_indices,
            neighbor_indices,
            alpha,
            normalize=True,
        )
        cell_l2, cell_relative = compute_cell_proxy_reconstruction_errors(
            patient_aligned,
            resources["cell_emb_tensor_mean"],
            base_proxy_cell_indices,
            query_indices,
            neighbor_indices,
            alpha,
        )
        patient_audit_summaries[split_name] = save_patient_neighbor_audit(
            save_dir=config["save_dir"],
            split_tag=split_name,
            query_indices=query_indices,
            neighbor_indices_global=neighbor_indices,
            neighbor_weights=alpha,
            neighbor_cosines=neighbor_cosines,
            lbpt_diagnostics=lbpt_diagnostics,
            patient_ids=resources["patient_ids"],
            patient_projects=resources["patient_cancers_neighbor"],
            gene_reconstruction_l2=gene_l2,
            gene_reconstruction_relative_l2=gene_relative,
            cell_reconstruction_l2=cell_l2,
            cell_reconstruction_relative_l2=cell_relative,
        )
        condition_audit_summaries[split_name] = _save_condition_dataset_audit(
            dataset,
            config["save_dir"],
            split_name,
            config["condition_audit_max_rows"],
        )
        split_summaries[split_name] = {
            "query_n": int(len(query_indices)),
            "full_record_n": int(len(records)),
            "runtime_record_n": int(len(dataset)),
            "record_summary": record_summary,
            "record_metadata_sha256": _json_sha256(record_metadata),
            "base_proxy_missing_n": int(
                np.sum(base_proxy_cell_indices[query_indices] < 0)
            ),
        }

    summary_payload = {
        "method_definition_sha256": method_sha256,
        "condition_store": condition_store.summary(),
        "condition_index_sha256": condition_index.definition_sha256(),
        "proxy_assignment_mode": str(config["proxy_assignment_mode"]),
        "active_patient_n": int(len(active_indices)),
        "active_base_proxy_missing_n": int(
            np.sum(base_proxy_cell_indices[active_indices] < 0)
        ),
        "base_proxy_cosine_finite_n": int(
            np.isfinite(base_proxy_cosines[active_indices]).sum()
        ),
        "splits": split_summaries,
        "patient_neighbor_audit": patient_audit_summaries,
        "condition_proxy_audit": condition_audit_summaries,
        "main_test_touched": bool(config.get("final_test_mode", False)),
    }
    summary_filename = (
        "he_lbpt_fixed_gate_sweep_v1_main_test_audit_summary.json"
        if config.get("final_test_mode", False)
        else "he_lbpt_fixed_gate_sweep_v1_audit_summary.json"
    )
    _atomic_json_dump(
        summary_payload,
        os.path.join(config["save_dir"], summary_filename),
    )

    if config.get("preflight_only", False):
        manifest = {
            "schema_version": 1,
            "method_version": str(config["pseudo_target_cache_version"]),
            "initialization": "random_perturbation_network",
            "model_spec": config["model_spec"],
            "perturb_hidden_dim": int(config["perturb_hidden_dim"]),
            "status": "passed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method_definition": method_definition,
            "method_definition_sha256": method_sha256,
            "main_test_touched": False,
            "audited_splits": split_names,
            "split_summaries": split_summaries,
            "condition_store_fingerprint": condition_store.store_fingerprint,
            "condition_index_sha256": condition_index.definition_sha256(),
            "proxy_assignment_mode": str(config["proxy_assignment_mode"]),
        }
        _atomic_json_dump(
            manifest,
            os.path.join(
                config["save_dir"],
                "preflight_manifest_he_lbpt_fixed_gate_sweep_v1.json",
            ),
        )
    elif locked_manifest is not None and not config.get("final_test_mode", False):
        locked_splits = locked_manifest.get("split_summaries", {})
        for split_name in ("main_train", "main_val"):
            expected_record_sha = (
                locked_splits.get(split_name, {})
                .get("record_summary", {})
                .get("records_sha256")
            )
            current_record_sha = (
                split_summaries.get(split_name, {})
                .get("record_summary", {})
                .get("records_sha256")
            )
            if expected_record_sha != current_record_sha:
                raise RuntimeError(
                    f"{split_name}: record checksum differs from preflight"
                )
            expected_metadata_sha = locked_splits.get(split_name, {}).get(
                "record_metadata_sha256"
            )
            current_metadata_sha = split_summaries.get(split_name, {}).get(
                "record_metadata_sha256"
            )
            if expected_metadata_sha != current_metadata_sha:
                raise RuntimeError(
                    f"{split_name}: pseudo-target metadata differs from preflight"
                )

    return {
        "datasets": datasets,
        "summary": summary_payload,
        "method_definition_sha256": method_sha256,
    }


def run_he_lbpt_stage3_flow(model, resources, config, device):
    prepared = prepare_he_lbpt_condition_datasets(
        model=model,
        resources=resources,
        config=config,
        device=device,
    )
    if config.get("preflight_only", False):
        print("[preflight] target preparation complete; manifest saved")
        return prepared["summary"]
    if config.get("final_test_mode", False):
        print(
            "[final-test-cache] main_test target cache and summary saved"
        )
        return prepared["summary"]
    if not config.get("run_stage3", False):
        print("[Stage3] dataset preparation complete")
        return prepared["summary"]

    config["method_definition_sha256"] = prepared["method_definition_sha256"]
    _atomic_json_dump(
        config,
        os.path.join(
            config["save_dir"],
            "config_he_lbpt_fixed_gate_sweep_v1_train_locked.json",
        ),
    )

    train_dataset = prepared["datasets"]["main_train"]
    val_dataset = prepared["datasets"]["main_val"]
    unicure_model = PerturbationPredictor(
        output_size=config["stage3_output_dim"],
        hidden_dim=config["perturb_hidden_dim"],
    ).to(device)
    model.unicure_model = unicure_model
    print(
        "[Stage3] perturbation network: random initialization"
    )

    start_epoch = 0
    resume_state = {
        "best_val_loss": None,
        "best_epoch": None,
        "patience_counter": 0,
        "optimizer_state_dict": None,
        "scaler_state_dict": None,
        "history": None,
    }
    if config.get("resume_stage3_training", False):
        model, resume_state = load_stage3_checkpoint_for_resume(
            model=model,
            config=config,
            device=device,
        )
        start_epoch = int(resume_state.get("start_epoch", 0))

    history = fit_perturbation_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device,
        start_epoch=start_epoch,
        init_best_val_loss=resume_state.get("best_val_loss"),
        init_best_epoch=resume_state.get("best_epoch"),
        init_patience_counter=resume_state.get("patience_counter", 0),
        optimizer_state_dict=resume_state.get("optimizer_state_dict"),
        scaler_state_dict=resume_state.get("scaler_state_dict"),
        init_history=resume_state.get("history"),
    )
    plot_stage3_text_fusion_history(config["save_dir"])
    return history


# Main


def train_patient_model_text_fusion(config=None):
    """Run Stage 3 preparation or training with precomputed patient and drug features."""
    if config is None:
        raise ValueError("Use python -m histopert with an explicit configuration")

    print("\nText Fusion Stage3 main split 75/10/15")
    print("save_dir =", config["save_dir"])
    print("device =", config["device"])
    print("stage1_dir =", config["stage1_dir"])
    print("stage2_dir =", config["stage2_dir"])
    print("resume_stage1_ckpt =", config["resume_stage1_ckpt"])
    print("resume_stage2_ckpt =", config["resume_stage2_ckpt"])
    print("main_split_path =", config["main_split_path"])
    print("patient_csv_path =", config["patient_csv_path"])
    print("patient_uce_path =", config["patient_uce_path"])
    print("topk_patient =", config["topk_patient"])
    print("topk_cell =", config["topk_cell"])
    print("self_cell_weight =", config["self_cell_weight"])
    print("load_old_stage3_partial =", config["load_old_stage3_partial"])

    os.makedirs(config["save_dir"], exist_ok=True)

    execution_mode = str(config.get("execution_mode", "direct"))
    with open(
        os.path.join(
            config["save_dir"],
            f"config_he_lbpt_fixed_gate_sweep_v1_{execution_mode}.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    set_seed(config["seed"])

    device = torch.device(config["device"])

    resources = build_training_resources(config)

    print("\nResource check")
    print("patient_img shape:", resources["patient_img"].shape)
    print("patient_gene shape:", resources["patient_gene"].shape)
    print("patient_text shape:", resources["patient_text"].shape)
    print("patient_clin shape:", resources["patient_clin"].shape)
    print("patients:", len(resources["patient_ids"]))
    print("train patients:", len(resources["train_idx"]))
    print("val patients:", len(resources["val_idx"]))
    print("main_test patients:", len(resources["main_test_idx"]))

    stage1_train_dataset = resources["stage1_train_dataset"]
    cell_emb_tensor_mmd = resources["cell_emb_tensor_mmd"]
    stage1_cancer_encoder = resources["stage1_cancer_encoder"]

    sample = stage1_train_dataset[0]

    img_dim = sample[0].numel()
    gene_dim = sample[1].numel()
    clin_dim = sample[2].numel()
    cell_dim = cell_emb_tensor_mmd.shape[1]

    print("\nModel dim check")
    print("img_dim:", img_dim)
    print("gene_dim:", gene_dim)
    print("clin_dim:", clin_dim)
    print("cell_dim:", cell_dim)
    print("patient_text_dim:", config["patient_text_dim"])
    print("num_cancers:", len(stage1_cancer_encoder))

    if cell_dim != 1280:
        raise ValueError(f"Expected cell_dim=1280, got {cell_dim}")

    if resources["patient_text"].shape[1] != config["patient_text_dim"]:
        raise ValueError("patient_text dim mismatch")

    model = PatientContextModel(
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
        use_clinical=False,
        use_patient_text=config.get("use_patient_text", True),
        patient_text_dim=config.get("patient_text_dim", 1024),
        unicure_model=None,
    ).to(device)

    config["model_spec"] = {
        "img_dim": img_dim,
        "gene_dim": gene_dim,
        "cell_dim": cell_dim,
        "num_cancers": len(stage1_cancer_encoder),
        "hidden_dim": config["hidden_dim"],
        "cancer_emb_dim": config["cancer_emb_dim"],
        "dropout": config["dropout"],
        "patient_text_dim": config["patient_text_dim"],
        "output_dim": config["stage3_output_dim"],
        "perturb_hidden_dim": config["perturb_hidden_dim"],
    }
    model, _ = resume_stage1_model_if_needed(model, config, device)
    model, _ = resume_stage2_model_if_needed(model, config, device)

    if config.get("fixed_proxy_pseudo_targets", False):
        return run_he_lbpt_stage3_flow(
            model=model,
            resources=resources,
            config=config,
            device=device,
        )

    raise ValueError("Expected fixed-proxy G05-GF targets")


# Public aliases preserve compatibility with existing integrations.
freeze_stage1_stage2_train_fusion_and_stage3 = configure_stage3_parameters
train_stage3_text_fusion_fast = fit_perturbation_model
