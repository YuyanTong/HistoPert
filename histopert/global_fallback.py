"""Cancer-aware cell proxies with global fallback for unmatched patients."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import stage3 as base
from .condition_specific_store import build_fixed_proxy_mixture

RESULT_DIR = None  # Set to this release's output folder by the CLI.
_RAW_FALLBACK_AUDIT = None

_ORIGINAL_METHOD_DEFINITION = base._he_lbpt_method_definition


class GlobalFallbackConditionSpecificDataset(
    base.ConditionSpecificPatientDrugDoseDataset
):
    """Build fixed-proxy targets with cancer-scoped global fallback."""

    def __init__(
        self,
        *,
        patient_img,
        patient_gene,
        patient_ids,
        patient_cancer_idx,
        patient_cancers,
        patient_clin,
        patient_text,
        patient_text_dim,
        query_patient_indices,
        patient_neighbor_indices,
        patient_alpha,
        base_proxy_cell_indices,
        base_proxy_cosines,
        store,
        records,
        drug_embedding_path,
        drug_embedding_df=None,
        self_cell_weight=0.3,
        drug_target_dim=528,
        record_limit=None,
    ):
        self.patient_img = patient_img
        self.patient_gene = patient_gene
        self.patient_ids = tuple(map(str, patient_ids))
        self.patient_cancer_idx = np.asarray(patient_cancer_idx, dtype=np.int64)
        self.patient_cancers = np.asarray(patient_cancers, dtype=object)
        self.patient_clin = patient_clin
        self.patient_text = patient_text
        self.patient_text_dim = int(patient_text_dim)
        self.query_patient_indices = np.asarray(query_patient_indices, dtype=np.int64)
        self.patient_neighbor_indices = np.asarray(
            patient_neighbor_indices, dtype=np.int64
        )
        self.patient_alpha = np.asarray(patient_alpha, dtype=np.float64)
        base_proxy_cell_indices = np.asarray(base_proxy_cell_indices, dtype=np.int32)
        base_proxy_cosines = np.asarray(base_proxy_cosines, dtype=np.float32)
        self.store = store
        self.records = np.asarray(records, dtype=np.int32)
        self.self_cell_weight = float(self_cell_weight)
        self.drug_target_dim = int(drug_target_dim)
        self.record_limit = None if record_limit is None else int(record_limit)

        if not 0.0 <= self.self_cell_weight <= 1.0:
            raise ValueError("self_cell_weight must be in [0, 1]")
        if self.patient_neighbor_indices.shape != self.patient_alpha.shape:
            raise ValueError("patient neighbor and alpha shapes differ")
        if self.patient_neighbor_indices.shape[0] != len(self.query_patient_indices):
            raise ValueError("query and patient-neighbor row count mismatch")
        if self.records.ndim != 2 or self.records.shape[1] != 2:
            raise ValueError("records must have shape (N, 2)")
        if len(self.records) and (
            self.records[:, 0].min() < 0
            or self.records[:, 0].max() >= len(self.query_patient_indices)
            or self.records[:, 1].min() < 0
            or self.records[:, 1].max() >= self.store.condition_count
        ):
            raise ValueError("records contain out-of-range indices")
        alpha_sums = self.patient_alpha.sum(axis=1)
        if np.any(self.patient_alpha < -1e-7) or not np.allclose(
            alpha_sums, 1.0, atol=1e-6, rtol=0.0
        ):
            raise ValueError("patient alpha violates the probability simplex")
        if base_proxy_cell_indices.shape != (len(self.patient_ids),):
            raise ValueError("base proxy indices must align with all patients")
        if base_proxy_cosines.shape != (len(self.patient_ids),):
            raise ValueError("base proxy cosines must align with all patients")

        participants = np.concatenate(
            [self.query_patient_indices[:, None], self.patient_neighbor_indices],
            axis=1,
        )
        self.fixed_proxy_cell_indices = base_proxy_cell_indices[participants]
        self.fixed_proxy_cosines = base_proxy_cosines[participants]
        used_query_positions = (
            np.unique(self.records[:, 0])
            if len(self.records)
            else np.empty((0,), dtype=np.int32)
        )
        used_fixed_cells = self.fixed_proxy_cell_indices[used_query_positions]
        if np.any(used_fixed_cells < 0) or np.any(
            used_fixed_cells >= len(self.store.cell_ids)
        ):
            raise ValueError("a recorded target or neighbor lacks a fixed proxy")
        fixed_proxy_cancers = np.asarray(self.store.cell_cancers, dtype=object)[
            used_fixed_cells
        ]
        query_cancers = self.patient_cancers[
            self.query_patient_indices[used_query_positions]
        ]
        cross_cancer = fixed_proxy_cancers != query_cancers[:, None]
        allowed = query_cancers[:, None] == "UNMATCHED"
        if np.any(cross_cancer & ~allowed):
            bad = np.argwhere(cross_cancer & ~allowed)[0]
            raise ValueError(
                "fixed proxy dictionary crosses a non-UNMATCHED cancer label: "
                f"query_cancer={query_cancers[bad[0]]!r}, "
                f"proxy_cancer={fixed_proxy_cancers[tuple(bad)]!r}"
            )
        if not np.isfinite(self.fixed_proxy_cosines[used_query_positions]).all():
            raise ValueError("fixed proxy cosine contains NaN or infinity")

        self.fixed_proxy_weights = np.concatenate(
            [
                np.full(
                    (len(self.query_patient_indices), 1),
                    self.self_cell_weight,
                    dtype=np.float64,
                ),
                (1.0 - self.self_cell_weight) * self.patient_alpha,
            ],
            axis=1,
        )
        if not np.allclose(
            self.fixed_proxy_weights.sum(axis=1), 1.0, atol=1e-6, rtol=0.0
        ):
            raise ValueError("fixed proxy intended weights do not sum to one")

        if drug_embedding_df is None:
            self.drug_emb_df = pd.read_parquet(drug_embedding_path)
            self.drug_emb_df.index = self.drug_emb_df.index.astype(str)
        else:
            self.drug_emb_df = drug_embedding_df
        self.drug_dose_emb_cache = {}

    def _match_record(self, qpos: int, condition_id: int) -> dict:
        patient_index = int(self.query_patient_indices[qpos])
        neighbors = self.patient_neighbor_indices[qpos]
        participants = np.concatenate([[patient_index], neighbors])
        cancer, drug, dose = self.store.condition_key(condition_id)
        participant_cancers = self.patient_cancers[participants]
        if not np.all(participant_cancers == participant_cancers[0]):
            raise AssertionError("record participants cross Stage1 cancer labels")
        if participant_cancers[0] != "UNMATCHED" and not np.all(
            participant_cancers == cancer
        ):
            raise AssertionError(
                "non-UNMATCHED participants do not match condition cancer"
            )

        candidate_slice = self.store.candidate_slice(condition_id)
        candidate_cells = self.store.candidate_cell_indices[candidate_slice]
        if len(candidate_cells) == 0:
            raise RuntimeError("condition record has no eligible candidates")
        candidate_rows = np.arange(
            int(candidate_slice.start), int(candidate_slice.stop), dtype=np.int64
        )
        mixture = build_fixed_proxy_mixture(
            fixed_cell_indices=self.fixed_proxy_cell_indices[qpos],
            fixed_weights=self.fixed_proxy_weights[qpos],
            candidate_cell_indices=candidate_cells,
            candidate_treated_row_indices=candidate_rows,
            condition_id=condition_id,
        )
        selected_cells = mixture.available_cell_indices
        selected_rows = mixture.available_treated_row_indices
        merged_rows = torch.as_tensor(mixture.treated_row_indices, dtype=torch.long)
        merged_cells = torch.as_tensor(mixture.cell_indices, dtype=torch.long)
        treated = self.store.treated_expr[merged_rows]
        control = self.store.control_expr[merged_cells]
        if not torch.isfinite(treated).all() or not torch.isfinite(control).all():
            raise RuntimeError("selected condition proxy lacks finite treated/control")
        expression_delta = treated - control
        weights = torch.tensor(mixture.weights, dtype=torch.float32)
        delta_target = (expression_delta * weights[:, None]).sum(dim=0)
        control_target = (control * weights[:, None]).sum(dim=0)
        if not torch.isfinite(delta_target).all():
            raise RuntimeError("pseudo delta contains NaN or infinity")

        return {
            "patient_idx": patient_index,
            "neighbor_indices": neighbors,
            "condition_id": int(condition_id),
            "cancer": cancer,
            "drug": drug,
            "dose": float(dose),
            "candidate_cells": candidate_cells,
            "fixed_cells": mixture.fixed_cell_indices,
            "fixed_cosines": self.fixed_proxy_cosines[qpos],
            "fixed_weights": mixture.fixed_weights,
            "available_mask": mixture.available_mask,
            "selected_cells": selected_cells.astype(np.int32),
            "available_weights_before_normalization": (
                mixture.available_weights_before_normalization
            ),
            "available_weights": mixture.available_weights,
            "available_weight_mass": float(mixture.available_weight_mass),
            "zero_mass_fallback": bool(mixture.zero_mass_fallback),
            "merged_cells": mixture.cell_indices,
            "merged_rows": mixture.treated_row_indices,
            "merged_weights": mixture.weights,
            "collision_n": mixture.collision_n,
            "delta_target": delta_target,
            "control_target": control_target,
            "selected_rows": selected_rows.astype(np.int64),
        }


def same_cancer_else_clean_global_base_proxy(
    *,
    patient_embeddings: torch.Tensor,
    patient_cancers,
    cell_embeddings: torch.Tensor,
    store,
    active_patient_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign same-cancer proxies, using the global tumor pool for UNMATCHED patients."""

    patient_embeddings = torch.nn.functional.normalize(
        patient_embeddings.detach().cpu().float(), dim=1
    )
    cell_embeddings = torch.nn.functional.normalize(
        cell_embeddings.detach().cpu().float(), dim=1
    )
    patient_cancers = np.asarray(patient_cancers, dtype=object)
    if len(patient_cancers) != len(patient_embeddings):
        raise ValueError("patient cancer metadata length mismatch")
    if len(cell_embeddings) != len(store.cell_ids):
        raise ValueError("cell embedding order does not match condition store")

    has_conditions = np.diff(store.cell_condition_offsets) > 0
    cell_cancers = np.asarray(store.cell_cancers, dtype=object)
    indices = np.full((len(patient_embeddings),), -1, dtype=np.int32)
    similarities = np.full((len(patient_embeddings),), np.nan, dtype=np.float32)
    active = np.ones(len(patient_embeddings), dtype=bool)
    if active_patient_indices is not None:
        active[:] = False
        active_indices = np.asarray(active_patient_indices, dtype=np.int64)
        if np.any(active_indices < 0) or np.any(active_indices >= len(active)):
            raise IndexError("active patient index is out of range")
        active[active_indices] = True

    global_candidates = np.arange(len(cell_embeddings), dtype=np.int64)
    audit_rows = []
    for cancer in sorted(set(map(str, patient_cancers))):
        query_indices = np.flatnonzero((patient_cancers == cancer) & active)
        if len(query_indices) == 0:
            continue
        same_cancer_indices = np.flatnonzero((cell_cancers == cancer) & has_conditions)
        if len(same_cancer_indices):
            candidates = same_cancer_indices
            mode = "same_cancer"
        else:
            candidates = global_candidates
            mode = "clean_global_fallback"
        similarities_by_cell = (
            patient_embeddings[query_indices] @ cell_embeddings[candidates].T
        )
        positions = torch.argmax(similarities_by_cell, dim=1)
        rows = np.arange(len(query_indices))
        selected = candidates[positions.numpy()]
        selected_scores = (
            similarities_by_cell[rows, positions].numpy().astype(np.float32)
        )
        indices[query_indices] = selected
        similarities[query_indices] = selected_scores
        audit_rows.extend(
            {
                "patient_index": int(patient_index),
                "patient_cancer": str(cancer),
                "selection_mode": mode,
                "selected_cell_index": int(cell_index),
                "selected_cell_id": str(store.cell_ids[int(cell_index)]),
                "selected_cell_cancer": str(store.cell_cancers[int(cell_index)]),
                "selected_cell_has_conditions": bool(has_conditions[int(cell_index)]),
                "cosine": float(cosine),
            }
            for patient_index, cell_index, cosine in zip(
                query_indices, selected, selected_scores
            )
        )

    audit = pd.DataFrame(audit_rows).sort_values("patient_index")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(_RAW_FALLBACK_AUDIT, index=False)
    fallback = audit[audit["selection_mode"].eq("clean_global_fallback")]
    if len(fallback) and set(fallback["patient_cancer"]) != {"UNMATCHED"}:
        raise AssertionError(
            "Global fallback requires UNMATCHED patients: "
            f"{sorted(set(fallback['patient_cancer']))}"
        )
    print(
        "[G05 global fallback] active patients="
        f"{len(audit)}, fallback patients={len(fallback)}, "
        f"fallback cancers={sorted(set(fallback['patient_cancer']))}"
    )
    return indices, similarities


def method_definition_with_global_fallback(config, resources, condition_store):
    payload, _old_sha = _ORIGINAL_METHOD_DEFINITION(config, resources, condition_store)
    payload["global_fallback_extension"] = {
        "enabled": True,
        "scope": "only_when_no_eligible_same_cancer_cell_exists",
        "expected_fallback_stage1_label": "UNMATCHED",
        "candidate_library": "all_clean_global_stage3_cells",
        "condition_level_fallback": False,
        "wrapper_manifest": base.source_file_manifest([Path(__file__).resolve()]),
    }
    return payload, base._json_sha256(payload)


def install_extension() -> None:
    base.strict_same_cancer_base_proxy = same_cancer_else_clean_global_base_proxy
    base.ConditionSpecificPatientDrugDoseDataset = (
        GlobalFallbackConditionSpecificDataset
    )
    base._he_lbpt_method_definition = method_definition_with_global_fallback


def finalize_fallback_audit() -> None:
    if not _RAW_FALLBACK_AUDIT.is_file():
        return
    patient_path = RESULT_DIR / "stage3_patients.csv"
    if not patient_path.is_file():
        raise FileNotFoundError(patient_path)
    audit = pd.read_csv(_RAW_FALLBACK_AUDIT)
    patients = pd.read_csv(patient_path).reset_index(names="patient_index")
    audit = audit.merge(
        patients[["patient_index", "patient_id", "cancer_raw", "split"]],
        on="patient_index",
        how="left",
        validate="one_to_one",
    )
    audit.to_csv(RESULT_DIR / "base_proxy_global_fallback_audit.csv", index=False)
    summary = (
        audit.groupby(["split", "selection_mode"], dropna=False)
        .size()
        .rename("patient_n")
        .reset_index()
    )
    summary.to_csv(RESULT_DIR / "base_proxy_global_fallback_summary.csv", index=False)
    print(summary.to_string(index=False))
