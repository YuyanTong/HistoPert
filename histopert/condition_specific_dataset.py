"""LINCS expression store and fixed-proxy targets for Stage 3."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .condition_specific_store import (
    ConditionSpecificIndex,
    build_fixed_proxy_mixture,
)

STORE_SCHEMA_VERSION = 2
RECORD_SCHEMA_VERSION = 3
CONTROL_NAMES = ("control", "CTRL", "Control")


def _torch_load(path: os.PathLike[str] | str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sha256_bytes(*chunks: bytes) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def hash_numpy_array(value: Any, dtype=None) -> str:
    array = np.asarray(value, dtype=dtype)
    contiguous = np.ascontiguousarray(array)
    return _sha256_bytes(
        str(contiguous.dtype).encode("utf-8"),
        str(contiguous.shape).encode("utf-8"),
        contiguous.tobytes(),
    )


def _update_digest_array(digest, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
    else:
        array = np.ascontiguousarray(np.asarray(value))
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))


def _update_digest_strings(digest, values: Sequence[str]) -> None:
    encoded = json.dumps(
        list(map(str, values)),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little", signed=False))
    digest.update(encoded)


def source_file_manifest(paths: Sequence[os.PathLike[str] | str]) -> dict:
    files = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        files.append(
            {
                "path": str(path),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": digest.hexdigest(),
            }
        )
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"files": files, "signature": hashlib.sha256(encoded).hexdigest()}


def _atomic_torch_save(payload: dict, path: os.PathLike[str] | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass
class ConditionSpecificLINCSStore:
    """Compact eligible LINCS expression and condition-candidate index."""

    cell_ids: tuple[str, ...]
    cell_cancers: tuple[str, ...]
    gene_names: tuple[str, ...]
    cancer_names: tuple[str, ...]
    drug_names: tuple[str, ...]
    cell_cancer_idx: np.ndarray
    condition_cancer_idx: np.ndarray
    condition_drug_idx: np.ndarray
    condition_doses: np.ndarray
    condition_candidate_offsets: np.ndarray
    candidate_cell_indices: np.ndarray
    treated_expr: torch.Tensor
    control_expr: torch.Tensor
    control_repeat_counts: np.ndarray
    treated_repeat_counts: np.ndarray
    cell_condition_offsets: np.ndarray
    cell_condition_ids: np.ndarray
    cancer_condition_offsets: np.ndarray
    cancer_condition_ids: np.ndarray
    source_manifest: dict
    store_fingerprint: str

    @property
    def output_dim(self) -> int:
        return int(self.treated_expr.shape[1])

    @property
    def condition_count(self) -> int:
        return int(len(self.condition_doses))

    @property
    def eligible_row_count(self) -> int:
        return int(len(self.candidate_cell_indices))

    def candidate_slice(self, condition_id: int) -> slice:
        condition_id = int(condition_id)
        if not 0 <= condition_id < self.condition_count:
            raise IndexError(f"condition_id out of range: {condition_id}")
        return slice(
            int(self.condition_candidate_offsets[condition_id]),
            int(self.condition_candidate_offsets[condition_id + 1]),
        )

    def condition_for_cell(self, cell_idx: int) -> np.ndarray:
        cell_idx = int(cell_idx)
        if not 0 <= cell_idx < len(self.cell_ids):
            raise IndexError(f"cell_idx out of range: {cell_idx}")
        start = int(self.cell_condition_offsets[cell_idx])
        end = int(self.cell_condition_offsets[cell_idx + 1])
        return self.cell_condition_ids[start:end]

    def conditions_for_cancer_idx(self, cancer_idx: int) -> np.ndarray:
        cancer_idx = int(cancer_idx)
        if not 0 <= cancer_idx < len(self.cancer_names):
            return np.empty((0,), dtype=np.int32)
        start = int(self.cancer_condition_offsets[cancer_idx])
        end = int(self.cancer_condition_offsets[cancer_idx + 1])
        return self.cancer_condition_ids[start:end]

    def condition_key(self, condition_id: int) -> tuple[str, str, float]:
        condition_id = int(condition_id)
        cancer = self.cancer_names[int(self.condition_cancer_idx[condition_id])]
        drug = self.drug_names[int(self.condition_drug_idx[condition_id])]
        dose = float(self.condition_doses[condition_id])
        return cancer, drug, dose

    def summary(self) -> dict:
        candidate_n = np.diff(self.condition_candidate_offsets)
        condition_counts = np.bincount(
            self.condition_cancer_idx.astype(np.int64),
            minlength=len(self.cancer_names),
        )
        cell_counts = np.bincount(
            self.cell_cancer_idx.astype(np.int64),
            minlength=len(self.cancer_names),
        )
        return {
            "schema_version": STORE_SCHEMA_VERSION,
            "store_fingerprint": self.store_fingerprint,
            "cell_n": int(len(self.cell_ids)),
            "condition_n": self.condition_count,
            "eligible_cell_condition_rows": self.eligible_row_count,
            "output_dim": self.output_dim,
            "candidate_n_min": int(candidate_n.min()),
            "candidate_n_median": float(np.median(candidate_n)),
            "candidate_n_max": int(candidate_n.max()),
            "conditions_by_cancer": {
                cancer: int(condition_counts[index])
                for index, cancer in enumerate(self.cancer_names)
            },
            "cells_by_cancer": {
                cancer: int(cell_counts[index])
                for index, cancer in enumerate(self.cancer_names)
            },
            "source_manifest": self.source_manifest,
        }

    def to_payload(self) -> dict:
        return {
            "schema_version": STORE_SCHEMA_VERSION,
            "cell_ids": list(self.cell_ids),
            "cell_cancers": list(self.cell_cancers),
            "gene_names": list(self.gene_names),
            "cancer_names": list(self.cancer_names),
            "drug_names": list(self.drug_names),
            "cell_cancer_idx": self.cell_cancer_idx,
            "condition_cancer_idx": self.condition_cancer_idx,
            "condition_drug_idx": self.condition_drug_idx,
            "condition_doses": self.condition_doses,
            "condition_candidate_offsets": self.condition_candidate_offsets,
            "candidate_cell_indices": self.candidate_cell_indices,
            "treated_expr": self.treated_expr,
            "control_expr": self.control_expr,
            "control_repeat_counts": self.control_repeat_counts,
            "treated_repeat_counts": self.treated_repeat_counts,
            "cell_condition_offsets": self.cell_condition_offsets,
            "cell_condition_ids": self.cell_condition_ids,
            "cancer_condition_offsets": self.cancer_condition_offsets,
            "cancer_condition_ids": self.cancer_condition_ids,
            "source_manifest": self.source_manifest,
            "store_fingerprint": self.store_fingerprint,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "ConditionSpecificLINCSStore":
        if int(payload.get("schema_version", -1)) != STORE_SCHEMA_VERSION:
            raise RuntimeError("condition store schema mismatch")
        store = cls(
            cell_ids=tuple(map(str, payload["cell_ids"])),
            cell_cancers=tuple(map(str, payload["cell_cancers"])),
            gene_names=tuple(map(str, payload["gene_names"])),
            cancer_names=tuple(map(str, payload["cancer_names"])),
            drug_names=tuple(map(str, payload["drug_names"])),
            cell_cancer_idx=np.asarray(payload["cell_cancer_idx"], dtype=np.int16),
            condition_cancer_idx=np.asarray(
                payload["condition_cancer_idx"], dtype=np.int16
            ),
            condition_drug_idx=np.asarray(
                payload["condition_drug_idx"], dtype=np.int32
            ),
            condition_doses=np.asarray(payload["condition_doses"], dtype=np.float64),
            condition_candidate_offsets=np.asarray(
                payload["condition_candidate_offsets"], dtype=np.int64
            ),
            candidate_cell_indices=np.asarray(
                payload["candidate_cell_indices"], dtype=np.int32
            ),
            treated_expr=torch.as_tensor(payload["treated_expr"], dtype=torch.float32),
            control_expr=torch.as_tensor(payload["control_expr"], dtype=torch.float32),
            control_repeat_counts=np.asarray(
                payload["control_repeat_counts"], dtype=np.int32
            ),
            treated_repeat_counts=np.asarray(
                payload["treated_repeat_counts"], dtype=np.int32
            ),
            cell_condition_offsets=np.asarray(
                payload["cell_condition_offsets"], dtype=np.int64
            ),
            cell_condition_ids=np.asarray(
                payload["cell_condition_ids"], dtype=np.int32
            ),
            cancer_condition_offsets=np.asarray(
                payload["cancer_condition_offsets"], dtype=np.int64
            ),
            cancer_condition_ids=np.asarray(
                payload["cancer_condition_ids"], dtype=np.int32
            ),
            source_manifest=dict(payload["source_manifest"]),
            store_fingerprint=str(payload["store_fingerprint"]),
        )
        store.validate()
        actual_fingerprint = _condition_store_fingerprint(store)
        if store.store_fingerprint != actual_fingerprint:
            raise RuntimeError(
                "condition store content fingerprint mismatch; rebuild the cache"
            )
        return store

    def validate(self) -> None:
        cell_n = len(self.cell_ids)
        condition_n = self.condition_count
        row_n = self.eligible_row_count
        if len(set(self.cell_ids)) != cell_n:
            raise ValueError("condition store cell_ids are not unique")
        if len(self.cell_cancers) != cell_n or len(self.cell_cancer_idx) != cell_n:
            raise ValueError("condition store cell metadata length mismatch")
        if self.control_expr.shape != (cell_n, self.output_dim):
            raise ValueError("condition store control tensor shape mismatch")
        if self.treated_expr.shape != (row_n, self.output_dim):
            raise ValueError("condition store treated tensor shape mismatch")
        if len(self.condition_candidate_offsets) != condition_n + 1:
            raise ValueError("condition offsets length mismatch")
        if int(self.condition_candidate_offsets[0]) != 0:
            raise ValueError("condition offsets must start at zero")
        if int(self.condition_candidate_offsets[-1]) != row_n:
            raise ValueError("condition offsets do not cover candidate rows")
        if np.any(np.diff(self.condition_candidate_offsets) <= 0):
            raise ValueError("every condition must contain at least one candidate")
        if np.any(self.candidate_cell_indices < 0) or np.any(
            self.candidate_cell_indices >= cell_n
        ):
            raise ValueError("condition store contains invalid cell indices")
        for condition_id in range(condition_n):
            candidate_slice = self.candidate_slice(condition_id)
            cells = self.candidate_cell_indices[candidate_slice]
            if len(cells) > 1 and np.any(cells[1:] <= cells[:-1]):
                raise ValueError(
                    "candidate cells must be strictly increasing within each condition"
                )
        if not torch.isfinite(self.treated_expr).all():
            raise ValueError("treated expression contains NaN or infinity")
        unique_candidate_cells = np.unique(self.candidate_cell_indices)
        candidate_controls = self.control_expr[
            torch.as_tensor(unique_candidate_cells, dtype=torch.long)
        ]
        if not torch.isfinite(candidate_controls).all():
            raise ValueError("eligible candidate control contains NaN or infinity")
        if np.any(self.condition_cancer_idx < 0) or np.any(
            self.condition_cancer_idx >= len(self.cancer_names)
        ):
            raise ValueError("invalid condition cancer indices")
        repeated_condition_ids = np.repeat(
            np.arange(condition_n, dtype=np.int32),
            np.diff(self.condition_candidate_offsets),
        )
        if not np.array_equal(
            self.cell_cancer_idx[self.candidate_cell_indices],
            self.condition_cancer_idx[repeated_condition_ids],
        ):
            raise ValueError("condition candidate crosses cancer labels")


def _build_offsets_for_group_ids(
    group_ids: np.ndarray,
    values: np.ndarray,
    group_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((values, group_ids))
    sorted_groups = group_ids[order]
    sorted_values = values[order]
    counts = np.bincount(sorted_groups, minlength=group_count)
    offsets = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])
    return offsets.astype(np.int64), sorted_values.astype(np.int32)


def condition_index_from_lincs_store(
    store: ConditionSpecificLINCSStore,
) -> ConditionSpecificIndex:
    """Expose expression rows aligned with candidate cell indices."""

    set_maps: dict[int, dict[tuple[int, ...], int]] = {}
    set_lists: dict[int, list[np.ndarray]] = {}
    condition_set_local = np.empty(store.condition_count, dtype=np.int32)
    conditions_by_cancer: dict[int, list[int]] = {}
    cell_to_conditions: list[list[int]] = [[] for _ in range(len(store.cell_ids))]

    for condition_id in range(store.condition_count):
        candidate_slice = store.candidate_slice(condition_id)
        cells = store.candidate_cell_indices[candidate_slice]
        cancer = int(store.condition_cancer_idx[condition_id])
        key = tuple(int(x) for x in cells)
        local_map = set_maps.setdefault(cancer, {})
        local_sets = set_lists.setdefault(cancer, [])
        set_local = local_map.get(key)
        if set_local is None:
            set_local = len(local_sets)
            local_map[key] = set_local
            local_sets.append(np.asarray(cells, dtype=np.int32))
        condition_set_local[condition_id] = set_local
        conditions_by_cancer.setdefault(cancer, []).append(condition_id)
        for cell_idx in cells:
            cell_to_conditions[int(cell_idx)].append(condition_id)

    index = ConditionSpecificIndex(
        cell_ids=np.asarray(store.cell_ids, dtype=object),
        cell_cancer_idx=np.asarray(store.cell_cancer_idx, dtype=np.int32),
        drug_names=tuple(store.drug_names),
        condition_cancer_idx=np.asarray(store.condition_cancer_idx, dtype=np.int32),
        condition_drug_idx=np.asarray(store.condition_drug_idx, dtype=np.int32),
        condition_dose=np.asarray(store.condition_doses, dtype=np.float64),
        condition_offsets=np.asarray(store.condition_candidate_offsets, dtype=np.int64),
        condition_cells=np.asarray(store.candidate_cell_indices, dtype=np.int32),
        condition_treated_rows=np.arange(store.eligible_row_count, dtype=np.int64),
        condition_set_local=condition_set_local,
        candidate_sets_by_cancer={
            cancer: tuple(values) for cancer, values in set_lists.items()
        },
        conditions_by_cancer={
            cancer: np.asarray(values, dtype=np.int32)
            for cancer, values in conditions_by_cancer.items()
        },
        cell_to_conditions=tuple(
            np.asarray(sorted(set(values)), dtype=np.int32)
            for values in cell_to_conditions
        ),
    )
    return index


def _condition_store_fingerprint(store: ConditionSpecificLINCSStore) -> str:
    """Hash every field that can affect record scope or pseudo targets."""

    digest = hashlib.sha256(
        f"condition-specific-lincs-store-v{STORE_SCHEMA_VERSION}\0".encode("ascii")
    )
    digest.update(
        json.dumps(
            store.source_manifest,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for values in (
        store.cell_ids,
        store.cell_cancers,
        store.gene_names,
        store.cancer_names,
        store.drug_names,
    ):
        _update_digest_strings(digest, values)
    for values in (
        store.cell_cancer_idx,
        store.condition_cancer_idx,
        store.condition_drug_idx,
        store.condition_doses,
        store.condition_candidate_offsets,
        store.candidate_cell_indices,
        store.treated_expr,
        store.control_expr,
        store.control_repeat_counts,
        store.treated_repeat_counts,
        store.cell_condition_offsets,
        store.cell_condition_ids,
        store.cancer_condition_offsets,
        store.cancer_condition_ids,
    ):
        _update_digest_array(digest, values)
    return digest.hexdigest()


def build_condition_specific_lincs_store(
    *,
    cell_ids: Sequence[str],
    cell_cancers: Sequence[str],
    treated_path: os.PathLike[str] | str,
    control_path: os.PathLike[str] | str,
    drug_embedding_path: os.PathLike[str] | str,
    source_manifest: dict | None = None,
) -> ConditionSpecificLINCSStore:
    """Build a deterministic strict-same-cancer condition store from parquet."""

    cell_ids = tuple(map(str, cell_ids))
    cell_cancers = tuple(map(str, cell_cancers))
    if len(cell_ids) != len(cell_cancers):
        raise ValueError("cell_ids and cell_cancers length mismatch")
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("cell_ids must be unique before store construction")

    if source_manifest is None:
        source_manifest = source_file_manifest(
            [treated_path, control_path, drug_embedding_path]
        )
    cell_to_idx = {cell_id: index for index, cell_id in enumerate(cell_ids)}
    cancer_names = tuple(sorted(set(cell_cancers)))
    cancer_to_idx = {cancer: index for index, cancer in enumerate(cancer_names)}
    cell_cancer_idx = np.asarray(
        [cancer_to_idx[cancer] for cancer in cell_cancers], dtype=np.int16
    )

    control_df = pd.read_parquet(control_path)
    control_df.columns = [str(column).strip() for column in control_df.columns]
    if "new_cid" not in control_df.columns:
        raise ValueError("control parquet is missing new_cid")
    control_df["new_cid"] = control_df["new_cid"].astype(str).str.strip()

    treated_df = pd.read_parquet(treated_path)
    if not isinstance(treated_df.index, pd.MultiIndex):
        raise ValueError("treated parquet must use a MultiIndex")
    treated_df.index = treated_df.index.set_names(
        ["new_cid", "cmap_name", "pert_idose"]
    )
    treated_df.columns = [str(column).strip() for column in treated_df.columns]
    gene_names = tuple(map(str, treated_df.columns))
    missing_control_genes = [
        gene for gene in gene_names if gene not in control_df.columns
    ]
    if missing_control_genes:
        raise ValueError(
            "control parquet is missing treated genes: "
            + ", ".join(missing_control_genes[:10])
        )

    control_grouped = control_df.groupby("new_cid", sort=True)
    control_repeat_series = control_grouped.size()
    control_mean = control_grouped[list(gene_names)].mean()
    cell_n = len(cell_ids)
    output_dim = len(gene_names)
    control_expr = np.full((cell_n, output_dim), np.nan, dtype=np.float32)
    control_repeat_counts = np.zeros((cell_n,), dtype=np.int32)
    for cell_id, cell_idx in cell_to_idx.items():
        if cell_id in control_mean.index:
            control_expr[cell_idx] = control_mean.loc[cell_id].to_numpy(
                dtype=np.float32, copy=True
            )
            control_repeat_counts[cell_idx] = int(control_repeat_series.loc[cell_id])
    del control_grouped, control_mean, control_df

    drug_index = pd.read_parquet(drug_embedding_path, columns=[]).index
    eligible_drugs = set(map(str, drug_index))

    if treated_df.index.has_duplicates:
        treated_repeat_series = treated_df.groupby(level=[0, 1, 2], sort=True).size()
        treated_df = treated_df.groupby(level=[0, 1, 2], sort=True).mean().sort_index()
    else:
        treated_repeat_series = pd.Series(
            np.ones((len(treated_df),), dtype=np.int32), index=treated_df.index
        )

    meta = treated_df.index.to_frame(index=False)
    meta["new_cid"] = meta["new_cid"].astype(str).str.strip()
    meta["cmap_name"] = meta["cmap_name"].astype(str)
    meta["pert_idose"] = meta["pert_idose"].astype(float)
    meta["source_row"] = np.arange(len(meta), dtype=np.int64)
    meta["cell_idx"] = meta["new_cid"].map(cell_to_idx)
    valid_cell = meta["cell_idx"].notna()
    meta = meta.loc[valid_cell].copy()
    meta["cell_idx"] = meta["cell_idx"].astype(np.int32)
    has_control = control_repeat_counts[meta["cell_idx"].to_numpy()] > 0
    has_drug = meta["cmap_name"].isin(eligible_drugs).to_numpy()
    not_control = ~meta["cmap_name"].isin(CONTROL_NAMES).to_numpy()
    meta = meta.loc[has_control & has_drug & not_control].copy()
    if len(meta) == 0:
        raise RuntimeError("no eligible treated cell-drug-dose rows remain")
    meta["cancer"] = [cell_cancers[index] for index in meta["cell_idx"]]
    meta["repeat_count"] = [
        int(
            treated_repeat_series.loc[
                (row.new_cid, row.cmap_name, float(row.pert_idose))
            ]
        )
        for row in meta.itertuples(index=False)
    ]
    meta = meta.sort_values(
        ["cancer", "cmap_name", "pert_idose", "cell_idx"], kind="mergesort"
    ).reset_index(drop=True)

    treated_expr = torch.tensor(
        treated_df.iloc[meta["source_row"].to_numpy()][list(gene_names)].to_numpy(
            dtype=np.float32, copy=True
        ),
        dtype=torch.float32,
    )
    treated_repeat_counts = meta["repeat_count"].to_numpy(dtype=np.int32)
    del treated_df, treated_repeat_series

    drug_names = tuple(sorted(meta["cmap_name"].unique().tolist()))
    drug_to_idx = {drug: index for index, drug in enumerate(drug_names)}

    condition_cancer_idx = []
    condition_drug_idx = []
    condition_doses = []
    condition_offsets = [0]
    candidate_cells = []
    for (cancer, drug, dose), group in meta.groupby(
        ["cancer", "cmap_name", "pert_idose"], sort=True, observed=True
    ):
        cells = group["cell_idx"].to_numpy(dtype=np.int32)
        if len(np.unique(cells)) != len(cells):
            raise ValueError("duplicate cell within an aggregated condition")
        condition_cancer_idx.append(cancer_to_idx[str(cancer)])
        condition_drug_idx.append(drug_to_idx[str(drug)])
        condition_doses.append(float(dose))
        candidate_cells.append(cells)
        condition_offsets.append(condition_offsets[-1] + len(cells))

    condition_cancer_idx = np.asarray(condition_cancer_idx, dtype=np.int16)
    condition_drug_idx = np.asarray(condition_drug_idx, dtype=np.int32)
    condition_doses = np.asarray(condition_doses, dtype=np.float64)
    condition_offsets = np.asarray(condition_offsets, dtype=np.int64)
    candidate_cell_indices = np.concatenate(candidate_cells).astype(np.int32)
    if not np.array_equal(candidate_cell_indices, meta["cell_idx"].to_numpy()):
        raise AssertionError(
            "candidate order no longer matches treated expression rows"
        )

    condition_ids_for_rows = np.repeat(
        np.arange(len(condition_doses), dtype=np.int32),
        np.diff(condition_offsets),
    )
    cell_condition_offsets, cell_condition_ids = _build_offsets_for_group_ids(
        candidate_cell_indices,
        condition_ids_for_rows,
        len(cell_ids),
    )
    cancer_condition_offsets, cancer_condition_ids = _build_offsets_for_group_ids(
        condition_cancer_idx.astype(np.int64),
        np.arange(len(condition_doses), dtype=np.int32),
        len(cancer_names),
    )

    store = ConditionSpecificLINCSStore(
        cell_ids=cell_ids,
        cell_cancers=cell_cancers,
        gene_names=gene_names,
        cancer_names=cancer_names,
        drug_names=drug_names,
        cell_cancer_idx=cell_cancer_idx,
        condition_cancer_idx=condition_cancer_idx,
        condition_drug_idx=condition_drug_idx,
        condition_doses=condition_doses,
        condition_candidate_offsets=condition_offsets,
        candidate_cell_indices=candidate_cell_indices,
        treated_expr=treated_expr,
        control_expr=torch.tensor(control_expr, dtype=torch.float32),
        control_repeat_counts=control_repeat_counts,
        treated_repeat_counts=treated_repeat_counts,
        cell_condition_offsets=cell_condition_offsets,
        cell_condition_ids=cell_condition_ids,
        cancer_condition_offsets=cancer_condition_offsets,
        cancer_condition_ids=cancer_condition_ids,
        source_manifest=source_manifest,
        store_fingerprint="",
    )
    store.validate()
    store.store_fingerprint = _condition_store_fingerprint(store)
    return store


def load_or_build_condition_store(
    *,
    cache_path: os.PathLike[str] | str,
    cell_ids: Sequence[str],
    cell_cancers: Sequence[str],
    treated_path: os.PathLike[str] | str,
    control_path: os.PathLike[str] | str,
    drug_embedding_path: os.PathLike[str] | str,
    force_rebuild: bool = False,
) -> ConditionSpecificLINCSStore:
    source_manifest = source_file_manifest(
        [treated_path, control_path, drug_embedding_path]
    )
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_rebuild:
        payload = _torch_load(cache_path)
        if payload.get("source_manifest") != source_manifest:
            raise RuntimeError(
                "condition store source manifest changed; use a new cache version "
                "or explicitly force a rebuild"
            )
        store = ConditionSpecificLINCSStore.from_payload(payload)
        if store.cell_ids != tuple(map(str, cell_ids)):
            raise RuntimeError("condition store cell ID order mismatch")
        if store.cell_cancers != tuple(map(str, cell_cancers)):
            raise RuntimeError("condition store cell cancer labels mismatch")
        return store

    store = build_condition_specific_lincs_store(
        cell_ids=cell_ids,
        cell_cancers=cell_cancers,
        treated_path=treated_path,
        control_path=control_path,
        drug_embedding_path=drug_embedding_path,
        source_manifest=source_manifest,
    )
    _atomic_torch_save(store.to_payload(), cache_path)
    return store


def strict_same_cancer_base_proxy(
    *,
    patient_embeddings: torch.Tensor,
    patient_cancers: Sequence[str],
    cell_embeddings: torch.Tensor,
    store: ConditionSpecificLINCSStore,
    active_patient_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-1 same-cancer anchors used only to define the condition universe."""

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
    indices = np.full((len(patient_embeddings),), -1, dtype=np.int32)
    similarities = np.full((len(patient_embeddings),), np.nan, dtype=np.float32)
    cell_cancers = np.asarray(store.cell_cancers, dtype=object)
    active = np.ones(len(patient_embeddings), dtype=bool)
    if active_patient_indices is not None:
        active[:] = False
        active_indices = np.asarray(active_patient_indices, dtype=np.int64)
        if np.any(active_indices < 0) or np.any(active_indices >= len(active)):
            raise IndexError("active patient index is out of range")
        active[active_indices] = True
    for cancer in sorted(set(map(str, patient_cancers))):
        query = np.flatnonzero((patient_cancers == cancer) & active)
        if len(query) == 0:
            continue
        candidates = np.flatnonzero((cell_cancers == cancer) & has_conditions)
        if len(candidates) == 0:
            continue
        scores = patient_embeddings[query] @ cell_embeddings[candidates].T
        positions = torch.argmax(scores, dim=1)
        rows = np.arange(len(query))
        indices[query] = candidates[positions.numpy()]
        similarities[query] = scores[rows, positions].numpy().astype(np.float32)
    return indices, similarities


def build_condition_records(
    *,
    query_patient_indices: np.ndarray,
    patient_neighbor_indices: np.ndarray,
    patient_cancers: Sequence[str],
    base_proxy_cell_indices: np.ndarray,
    store: ConditionSpecificLINCSStore,
    universe_mode: str = "fixed_proxy_union",
    patient_alpha: np.ndarray | None = None,
    self_cell_weight: float = 0.3,
) -> tuple[np.ndarray, dict]:
    """Build deterministic compact split records without condition Cartesian blowup."""

    query_patient_indices = np.asarray(query_patient_indices, dtype=np.int64)
    patient_neighbor_indices = np.asarray(patient_neighbor_indices, dtype=np.int64)
    base_proxy_cell_indices = np.asarray(base_proxy_cell_indices, dtype=np.int32)
    patient_cancers = np.asarray(patient_cancers, dtype=object)
    if patient_neighbor_indices.shape[0] != len(query_patient_indices):
        raise ValueError("query and patient-neighbor row count mismatch")
    if universe_mode not in {
        "fixed_proxy_union",
        "anchor_union",
        "all_same_cancer",
    }:
        raise ValueError(f"unsupported condition universe mode: {universe_mode}")
    if universe_mode == "fixed_proxy_union":
        if patient_alpha is None:
            raise ValueError("fixed_proxy_union requires patient_alpha")
        patient_alpha = np.asarray(patient_alpha, dtype=np.float64)
        if patient_alpha.shape != patient_neighbor_indices.shape:
            raise ValueError("patient_alpha and neighbor shapes differ")
        if not 0.0 <= float(self_cell_weight) <= 1.0:
            raise ValueError("self_cell_weight must be in [0, 1]")
        if np.any(patient_alpha < -1e-7) or not np.allclose(
            patient_alpha.sum(axis=1), 1.0, atol=1e-6, rtol=0.0
        ):
            raise ValueError("patient_alpha violates the probability simplex")

    cancer_to_idx = {cancer: index for index, cancer in enumerate(store.cancer_names)}
    condition_rows = []
    skipped_no_same_cancer = 0
    zero_weight_mass_record_n = 0
    singleton_proxy_record_n = 0
    target_anchor_available_record_n = 0
    available_weight_mass_sum = 0.0
    available_weight_mass_min = float("inf")
    available_weight_mass_max = 0.0
    for qpos, patient_idx in enumerate(query_patient_indices):
        participants = np.concatenate(
            [[int(patient_idx)], patient_neighbor_indices[qpos].astype(np.int64)]
        )
        participant_cancers = patient_cancers[participants]
        if not np.all(participant_cancers == patient_cancers[patient_idx]):
            raise AssertionError("patient neighbors cross Stage1 cancer labels")

        if universe_mode in {"fixed_proxy_union", "anchor_union"}:
            anchors = base_proxy_cell_indices[participants]
            if np.any(anchors < 0):
                skipped_no_same_cancer += 1
                continue
            condition_parts = [store.condition_for_cell(cell) for cell in anchors]
            condition_ids = np.unique(np.concatenate(condition_parts))
            if universe_mode == "fixed_proxy_union" and len(condition_ids):
                role_weights = np.concatenate(
                    [
                        np.asarray([self_cell_weight], dtype=np.float64),
                        (1.0 - self_cell_weight) * patient_alpha[qpos],
                    ]
                )
                available_mass = np.zeros(len(condition_ids), dtype=np.float64)
                unique_proxy_n = np.zeros(len(condition_ids), dtype=np.int16)
                for cell, weight in zip(anchors, role_weights):
                    covered = store.condition_for_cell(int(cell))
                    positions = np.searchsorted(condition_ids, covered)
                    available_mass[positions] += float(weight)
                for cell in np.unique(anchors):
                    covered = store.condition_for_cell(int(cell))
                    positions = np.searchsorted(condition_ids, covered)
                    unique_proxy_n[positions] += 1
                target_conditions = store.condition_for_cell(int(anchors[0]))
                target_anchor_available_record_n += int(len(target_conditions))
                zero_weight_mass_record_n += int(np.sum(available_mass <= 0.0))
                singleton_proxy_record_n += int(np.sum(unique_proxy_n == 1))
                available_weight_mass_sum += float(available_mass.sum())
                available_weight_mass_min = min(
                    available_weight_mass_min, float(available_mass.min())
                )
                available_weight_mass_max = max(
                    available_weight_mass_max, float(available_mass.max())
                )
        else:
            cancer_idx = cancer_to_idx.get(str(patient_cancers[patient_idx]), -1)
            condition_ids = store.conditions_for_cancer_idx(cancer_idx)
            if len(condition_ids) == 0:
                skipped_no_same_cancer += 1
                continue

        if len(condition_ids):
            rows = np.empty((len(condition_ids), 2), dtype=np.int32)
            rows[:, 0] = int(qpos)
            rows[:, 1] = condition_ids
            condition_rows.append(rows)

    records = (
        np.concatenate(condition_rows, axis=0)
        if condition_rows
        else np.empty((0, 2), dtype=np.int32)
    )
    summary = {
        "query_n": int(len(query_patient_indices)),
        "record_n": int(len(records)),
        "condition_universe_mode": str(universe_mode),
        "queries_skipped_no_same_cancer_proxy": int(skipped_no_same_cancer),
        "queries_with_records": int(
            len(np.unique(records[:, 0])) if len(records) else 0
        ),
        "zero_available_weight_mass_record_n": int(zero_weight_mass_record_n),
        "single_unique_proxy_record_n": int(singleton_proxy_record_n),
        "target_anchor_available_record_n": int(target_anchor_available_record_n),
        "available_weight_mass_min": (
            float(available_weight_mass_min)
            if universe_mode == "fixed_proxy_union" and len(records)
            else None
        ),
        "available_weight_mass_mean": (
            float(available_weight_mass_sum / len(records))
            if universe_mode == "fixed_proxy_union" and len(records)
            else None
        ),
        "available_weight_mass_max": (
            float(available_weight_mass_max)
            if universe_mode == "fixed_proxy_union" and len(records)
            else None
        ),
        "records_sha256": hash_numpy_array(records, dtype=np.int32),
    }
    return records, summary


def load_or_build_split_records(
    *,
    cache_path: os.PathLike[str] | str,
    expected_metadata: dict,
    build_kwargs: dict,
    force_rebuild: bool = False,
) -> tuple[np.ndarray, dict]:
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_rebuild:
        payload = _torch_load(cache_path)
        if int(payload.get("schema_version", -1)) != RECORD_SCHEMA_VERSION:
            raise RuntimeError("split record cache schema mismatch")
        if payload.get("metadata") != expected_metadata:
            raise RuntimeError("split record cache metadata mismatch")
        records = np.asarray(payload["records"], dtype=np.int32)
        summary = dict(payload["summary"])
        if summary.get("records_sha256") != hash_numpy_array(records, np.int32):
            raise RuntimeError("split record cache checksum mismatch")
        return records, summary

    records, summary = build_condition_records(**build_kwargs)
    _atomic_torch_save(
        {
            "schema_version": RECORD_SCHEMA_VERSION,
            "metadata": expected_metadata,
            "records": records,
            "summary": summary,
        },
        cache_path,
    )
    return records, summary


class ConditionSpecificPatientDrugDoseDataset(Dataset):
    """Stage3 dataset using an immutable target-plus-neighbor proxy dictionary."""

    def __init__(
        self,
        *,
        patient_img: torch.Tensor,
        patient_gene: torch.Tensor,
        patient_ids: Sequence[str],
        patient_cancer_idx: np.ndarray,
        patient_cancers: Sequence[str],
        patient_clin: torch.Tensor | None,
        patient_text: torch.Tensor | None,
        patient_text_dim: int,
        query_patient_indices: np.ndarray,
        patient_neighbor_indices: np.ndarray,
        patient_alpha: np.ndarray,
        base_proxy_cell_indices: np.ndarray,
        base_proxy_cosines: np.ndarray,
        store: ConditionSpecificLINCSStore,
        records: np.ndarray,
        drug_embedding_path: os.PathLike[str] | str,
        drug_embedding_df: pd.DataFrame | None = None,
        self_cell_weight: float = 0.3,
        drug_target_dim: int = 528,
        record_limit: int | None = None,
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
            raise ValueError(
                "a recorded target or H&E neighbor lacks a fixed same-cancer proxy"
            )
        fixed_proxy_cancers = np.asarray(self.store.cell_cancers, dtype=object)[
            used_fixed_cells
        ]
        query_cancers = self.patient_cancers[
            self.query_patient_indices[used_query_positions]
        ]
        if not np.all(fixed_proxy_cancers == query_cancers[:, None]):
            raise ValueError("fixed proxy dictionary crosses cancer labels")
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
        self.drug_dose_emb_cache: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        if self.record_limit is None:
            return int(len(self.records))
        return int(min(len(self.records), self.record_limit))

    def _drug_embedding(self, condition_id: int) -> torch.Tensor:
        condition_id = int(condition_id)
        cached = self.drug_dose_emb_cache.get(condition_id)
        if cached is not None:
            return cached
        _, drug, dose = self.store.condition_key(condition_id)
        if drug not in self.drug_emb_df.index:
            raise KeyError(f"drug={drug!r} is absent from the drug embedding parquet")
        base = self.drug_emb_df.loc[drug].to_numpy(dtype=np.float32).reshape(-1)
        dose_value = np.float32(np.log10(float(dose) + 1.0))
        values = np.concatenate([base, np.asarray([dose_value], dtype=np.float32)])
        if len(values) < self.drug_target_dim:
            values = np.pad(
                values,
                (0, self.drug_target_dim - len(values)),
                constant_values=float(dose_value),
            )
        elif len(values) > self.drug_target_dim:
            values = values[: self.drug_target_dim]
        tensor = torch.tensor(values, dtype=torch.float32)
        self.drug_dose_emb_cache[condition_id] = tensor
        return tensor

    def _match_record(self, qpos: int, condition_id: int) -> dict:
        patient_idx = int(self.query_patient_indices[qpos])
        neighbors = self.patient_neighbor_indices[qpos]
        participants = np.concatenate([[patient_idx], neighbors])
        cancer, drug, dose = self.store.condition_key(condition_id)
        if not np.all(self.patient_cancers[participants] == cancer):
            raise AssertionError("record participants do not match condition cancer")

        candidate_slice = self.store.candidate_slice(condition_id)
        candidate_cells = self.store.candidate_cell_indices[candidate_slice]
        if len(candidate_cells) == 0:
            raise RuntimeError("condition record has no eligible candidates")

        candidate_rows = np.arange(
            int(candidate_slice.start),
            int(candidate_slice.stop),
            dtype=np.int64,
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
        delta = treated - control

        weights = torch.tensor(mixture.weights, dtype=torch.float32)
        delta_target = (delta * weights[:, None]).sum(dim=0)
        control_target = (control * weights[:, None]).sum(dim=0)
        if not torch.isfinite(delta_target).all():
            raise RuntimeError("pseudo delta contains NaN or infinity")

        return {
            "patient_idx": patient_idx,
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

    def __getitem__(self, index: int):
        qpos, condition_id = self.records[int(index)]
        matched = self._match_record(int(qpos), int(condition_id))
        patient_idx = matched["patient_idx"]
        img = self.patient_img[patient_idx]
        gene = self.patient_gene[patient_idx]
        clin = (
            self.patient_clin[patient_idx]
            if self.patient_clin is not None
            else torch.zeros(1, dtype=torch.float32)
        )
        text_embedding = (
            self.patient_text[patient_idx]
            if self.patient_text is not None
            else torch.zeros(self.patient_text_dim, dtype=torch.float32)
        )
        cancer_idx = torch.tensor(
            self.patient_cancer_idx[patient_idx], dtype=torch.long
        )
        drug_embedding = self._drug_embedding(matched["condition_id"])
        return (
            img,
            gene,
            clin,
            text_embedding,
            cancer_idx,
            drug_embedding,
            matched["delta_target"],
            matched["control_target"],
        )

    def audit_record(self, index: int) -> dict:
        qpos, condition_id = self.records[int(index)]
        matched = self._match_record(int(qpos), int(condition_id))
        merged_cells = np.asarray(matched["merged_cells"], dtype=np.int32)
        merged_weights = np.asarray(matched["merged_weights"], dtype=np.float64)
        selected_rows = matched["selected_rows"]
        participant_ids = [self.patient_ids[int(matched["patient_idx"])]] + [
            self.patient_ids[int(x)] for x in matched["neighbor_indices"]
        ]
        return {
            "record_index": int(index),
            "query_position": int(qpos),
            "patient_idx": int(matched["patient_idx"]),
            "patient_id": self.patient_ids[int(matched["patient_idx"])],
            "cancer": matched["cancer"],
            "drug": matched["drug"],
            "dose": float(matched["dose"]),
            "candidate_n": int(len(matched["candidate_cells"])),
            "candidate_cell_ids": ";".join(
                self.store.cell_ids[int(cell)] for cell in matched["candidate_cells"]
            ),
            "participant_patient_ids": ";".join(participant_ids),
            "proxy_roles": ";".join(
                ["target_anchor"]
                + [f"neighbor_{i + 1}" for i in range(len(matched["neighbor_indices"]))]
            ),
            "fixed_proxy_cell_ids": ";".join(
                self.store.cell_ids[int(cell)] for cell in matched["fixed_cells"]
            ),
            "fixed_proxy_base_cosines": ";".join(
                f"{float(value):.9f}" for value in matched["fixed_cosines"]
            ),
            "nominal_role_weights": ";".join(
                f"{float(value):.9f}" for value in matched["fixed_weights"]
            ),
            "condition_available_mask": ";".join(
                "1" if bool(value) else "0" for value in matched["available_mask"]
            ),
            "available_role_n": int(np.sum(matched["available_mask"])),
            "target_anchor_available": bool(matched["available_mask"][0]),
            "available_weight_mass": float(matched["available_weight_mass"]),
            "dropped_weight_mass": float(1.0 - matched["available_weight_mass"]),
            "zero_mass_fallback": bool(matched["zero_mass_fallback"]),
            "selected_cell_ids": ";".join(
                self.store.cell_ids[int(cell)] for cell in matched["selected_cells"]
            ),
            "available_weights_before_normalization": ";".join(
                f"{float(value):.9f}"
                for value in matched["available_weights_before_normalization"]
            ),
            "available_normalized_role_weights": ";".join(
                f"{float(value):.9f}" for value in matched["available_weights"]
            ),
            "merged_cell_ids": ";".join(
                self.store.cell_ids[int(cell)] for cell in merged_cells
            ),
            "merged_weights": ";".join(
                f"{float(value):.9f}" for value in merged_weights
            ),
            "unique_proxy_n": int(len(merged_cells)),
            "single_proxy": bool(len(merged_cells) == 1),
            "selected_is_fixed_proxy": bool(
                set(map(int, merged_cells)).issubset(
                    set(map(int, matched["fixed_cells"]))
                )
            ),
            "proxy_collision_n": int(matched["collision_n"]),
            "proxy_collision": bool(matched["collision_n"] > 0),
            "final_weight_sum": float(merged_weights.sum()),
            "treated_repeat_counts": ";".join(
                str(int(value))
                for value in self.store.treated_repeat_counts[selected_rows]
            ),
            "control_repeat_counts": ";".join(
                str(int(value))
                for value in self.store.control_repeat_counts[matched["selected_cells"]]
            ),
            "delta_l2_norm": float(torch.linalg.vector_norm(matched["delta_target"])),
            "delta_is_finite": bool(torch.isfinite(matched["delta_target"]).all()),
        }


def deterministic_audit_indices(record_count: int, max_rows: int) -> np.ndarray:
    record_count = int(record_count)
    max_rows = int(max_rows)
    if record_count <= 0 or max_rows <= 0:
        return np.empty((0,), dtype=np.int64)
    if record_count <= max_rows:
        return np.arange(record_count, dtype=np.int64)
    return np.unique(np.linspace(0, record_count - 1, num=max_rows, dtype=np.int64))
