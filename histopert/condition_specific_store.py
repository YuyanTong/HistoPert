"""LINCS condition indices, patient-condition records and fixed-proxy mixtures."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Collection, Mapping, Sequence

import numpy as np

ArrayLike = Sequence | np.ndarray


def _update_hash_array(digest, value: ArrayLike) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())


def _as_numpy(x, *, dtype=None) -> np.ndarray:
    """Convert NumPy/CPU torch-like input without importing torch."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=dtype)


def _normalize_rows(x: ArrayLike, eps: float = 1e-12) -> np.ndarray:
    x = _as_numpy(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected a 2D embedding matrix, got shape={x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("embedding matrix contains NaN or Inf")
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm <= eps):
        bad = int(np.flatnonzero(norm[:, 0] <= eps)[0])
        raise ValueError(f"embedding row {bad} has near-zero L2 norm")
    return x / norm


def _canonical_cell_id(x) -> str:
    return str(x).strip()


def _canonical_drug(x) -> str:
    return str(x).strip()


def _canonical_dose(x) -> float:
    dose = float(x)
    if not np.isfinite(dose):
        raise ValueError(f"dose must be finite, got {x!r}")
    # Preserve exact float64 doses for expression-table and record lookups.
    return dose


@dataclass(frozen=True)
class ProxyMatch:
    patient_idx: int
    condition_id: int
    cell_idx: int
    treated_row_idx: int
    cosine: float
    candidate_n: int


@dataclass(frozen=True)
class ProxyMixture:
    """Pre- and post-merge proxy assignments for one pseudo target."""

    condition_id: int
    raw_patient_indices: np.ndarray
    raw_cell_indices: np.ndarray
    raw_treated_row_indices: np.ndarray
    raw_cosines: np.ndarray
    raw_weights: np.ndarray
    cell_indices: np.ndarray
    treated_row_indices: np.ndarray
    weights: np.ndarray

    @property
    def collision_n(self) -> int:
        return int(len(self.raw_cell_indices) - len(self.cell_indices))


@dataclass(frozen=True)
class FixedProxyMixture:
    """Availability-filtered mixture over one patient's fixed proxy dictionary."""

    condition_id: int
    fixed_cell_indices: np.ndarray
    fixed_weights: np.ndarray
    available_mask: np.ndarray
    available_cell_indices: np.ndarray
    available_treated_row_indices: np.ndarray
    available_weights_before_normalization: np.ndarray
    available_weights: np.ndarray
    available_weight_mass: float
    zero_mass_fallback: bool
    cell_indices: np.ndarray
    treated_row_indices: np.ndarray
    weights: np.ndarray

    @property
    def collision_n(self) -> int:
        return int(len(self.available_cell_indices) - len(self.cell_indices))

    @property
    def missing_n(self) -> int:
        return int(len(self.fixed_cell_indices) - int(self.available_mask.sum()))


@dataclass(frozen=True)
class CompactConditionRecords:
    """Two int32 arrays replace millions of Python record tuples."""

    query_positions: np.ndarray
    condition_ids: np.ndarray

    def __post_init__(self):
        q = np.asarray(self.query_positions)
        c = np.asarray(self.condition_ids)
        if q.dtype != np.int32 or c.dtype != np.int32:
            raise TypeError("compact record arrays must both use int32")
        if q.ndim != 1 or c.ndim != 1 or len(q) != len(c):
            raise ValueError("compact record arrays must be aligned 1D arrays")

    def __len__(self) -> int:
        return int(len(self.query_positions))

    def __getitem__(self, idx: int) -> tuple[int, int]:
        return int(self.query_positions[idx]), int(self.condition_ids[idx])

    def definition_sha256(self) -> str:
        digest = hashlib.sha256(b"compact-condition-records-v1\0")
        _update_hash_array(digest, self.query_positions)
        _update_hash_array(digest, self.condition_ids)
        return digest.hexdigest()

    def to_payload(self) -> dict:
        return {
            "record_schema_version": 1,
            "query_positions": self.query_positions,
            "condition_ids": self.condition_ids,
            "definition_sha256": self.definition_sha256(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping) -> "CompactConditionRecords":
        if int(payload.get("record_schema_version", -1)) != 1:
            raise ValueError("unsupported compact record schema")
        out = cls(
            query_positions=np.asarray(payload["query_positions"], dtype=np.int32),
            condition_ids=np.asarray(payload["condition_ids"], dtype=np.int32),
        )
        expected = payload.get("definition_sha256")
        if not expected or str(expected) != out.definition_sha256():
            raise ValueError("compact record payload checksum mismatch")
        return out


class ConditionSpecificIndex:
    """CSR index of cells for each ``(cancer_idx, drug_name, float64_dose)``.

    Candidate cells are sorted by cell index and aligned with treated rows.
    """

    schema_version = 1

    def __init__(
        self,
        *,
        cell_ids: np.ndarray,
        cell_cancer_idx: np.ndarray,
        drug_names: tuple[str, ...],
        condition_cancer_idx: np.ndarray,
        condition_drug_idx: np.ndarray,
        condition_dose: np.ndarray,
        condition_offsets: np.ndarray,
        condition_cells: np.ndarray,
        condition_treated_rows: np.ndarray,
        condition_set_local: np.ndarray,
        candidate_sets_by_cancer: dict[int, tuple[np.ndarray, ...]],
        conditions_by_cancer: dict[int, np.ndarray],
        cell_to_conditions: tuple[np.ndarray, ...],
    ):
        self.cell_ids = np.asarray(cell_ids, dtype=object)
        self.cell_cancer_idx = np.asarray(cell_cancer_idx, dtype=np.int32)
        self.drug_names = tuple(drug_names)
        self.condition_cancer_idx = np.asarray(condition_cancer_idx, dtype=np.int32)
        self.condition_drug_idx = np.asarray(condition_drug_idx, dtype=np.int32)
        self.condition_dose = np.asarray(condition_dose, dtype=np.float64)
        self.condition_offsets = np.asarray(condition_offsets, dtype=np.int64)
        self.condition_cells = np.asarray(condition_cells, dtype=np.int32)
        self.condition_treated_rows = np.asarray(condition_treated_rows, dtype=np.int64)
        self.condition_set_local = np.asarray(condition_set_local, dtype=np.int32)
        self.candidate_sets_by_cancer = {
            int(c): tuple(np.asarray(x, dtype=np.int32) for x in sets)
            for c, sets in candidate_sets_by_cancer.items()
        }
        self.conditions_by_cancer = {
            int(c): np.asarray(x, dtype=np.int32)
            for c, x in conditions_by_cancer.items()
        }
        self.cell_to_conditions = tuple(
            np.asarray(x, dtype=np.int32) for x in cell_to_conditions
        )

        self._validate()
        self._drug_to_idx = {x: i for i, x in enumerate(self.drug_names)}
        self._condition_key_to_id = {
            (
                int(self.condition_cancer_idx[i]),
                self.drug_names[int(self.condition_drug_idx[i])],
                float(self.condition_dose[i]),
            ): i
            for i in range(self.n_conditions)
        }
        if len(self._condition_key_to_id) != self.n_conditions:
            raise ValueError("condition keys must be unique")

    @property
    def n_cells(self) -> int:
        return int(len(self.cell_ids))

    @property
    def n_conditions(self) -> int:
        return int(len(self.condition_cancer_idx))

    @property
    def n_candidate_references(self) -> int:
        return int(len(self.condition_cells))

    def _validate(self):
        if len(set(map(str, self.cell_ids))) != self.n_cells:
            raise ValueError("cell_ids must be unique")
        if len(self.cell_cancer_idx) != self.n_cells:
            raise ValueError("cell_cancer_idx length mismatch")
        if self.n_cells == 0:
            raise ValueError("condition index must contain at least one cell")
        if np.any(self.cell_cancer_idx < 0):
            raise ValueError("clean-cell cancer indices must be nonnegative")
        if len(set(self.drug_names)) != len(self.drug_names):
            raise ValueError("drug_names must be unique")
        if any(not str(drug).strip() for drug in self.drug_names):
            raise ValueError("drug_names must be nonempty")
        n = self.n_conditions
        if n == 0:
            raise ValueError("condition index must contain at least one condition")
        for name, x in (
            ("condition_drug_idx", self.condition_drug_idx),
            ("condition_dose", self.condition_dose),
            ("condition_set_local", self.condition_set_local),
        ):
            if len(x) != n:
                raise ValueError(f"{name} length mismatch")
        if np.any(self.condition_drug_idx < 0) or np.any(
            self.condition_drug_idx >= len(self.drug_names)
        ):
            raise ValueError("condition drug index out of range")
        if not np.isfinite(self.condition_dose).all():
            raise ValueError("condition dose contains NaN or Inf")
        if np.any(self.condition_dose < 0):
            raise ValueError("condition dose must be nonnegative")
        if np.any(self.condition_cancer_idx < 0):
            raise ValueError("condition cancer indices must be nonnegative")
        if np.any(self.condition_set_local < 0):
            raise ValueError("condition candidate-set index must be nonnegative")
        if len(self.condition_offsets) != n + 1:
            raise ValueError("condition_offsets must have n_conditions + 1 entries")
        if self.condition_offsets[0] != 0:
            raise ValueError("condition_offsets must start at zero")
        if np.any(np.diff(self.condition_offsets) <= 0):
            raise ValueError("every condition must have at least one candidate")
        if int(self.condition_offsets[-1]) != self.n_candidate_references:
            raise ValueError("condition_offsets terminal value mismatch")
        if len(self.condition_treated_rows) != self.n_candidate_references:
            raise ValueError("treated-row CSR length mismatch")
        if np.any(self.condition_treated_rows < 0):
            raise ValueError("treated row index must be nonnegative")
        if len(np.unique(self.condition_treated_rows)) != len(
            self.condition_treated_rows
        ):
            raise ValueError("each eligible treated row must occur exactly once")
        if np.any(self.condition_cells < 0) or np.any(
            self.condition_cells >= self.n_cells
        ):
            raise ValueError("condition cell index out of range")
        if len(self.cell_to_conditions) != self.n_cells:
            raise ValueError("cell_to_conditions length mismatch")

        expected_conditions_by_cancer: dict[int, list[int]] = {}
        expected_cell_conditions: list[list[int]] = [[] for _ in range(self.n_cells)]
        for cond in range(n):
            cells, _ = self.condition_candidates(cond)
            cancer = int(self.condition_cancer_idx[cond])
            if np.any(np.diff(cells.astype(np.int64)) <= 0):
                raise ValueError(
                    f"condition {cond} candidate cells must be unique and sorted"
                )
            if not np.all(self.cell_cancer_idx[cells] == cancer):
                raise ValueError(f"condition {cond} contains a cross-cancer candidate")
            set_local = int(self.condition_set_local[cond])
            if cancer not in self.candidate_sets_by_cancer or set_local >= len(
                self.candidate_sets_by_cancer[cancer]
            ):
                raise ValueError(f"condition {cond} candidate-set index is invalid")
            expected = self.candidate_sets_by_cancer[cancer][set_local]
            if not np.array_equal(cells, expected):
                raise ValueError(
                    f"condition {cond} candidate-set mapping is inconsistent"
                )
            expected_conditions_by_cancer.setdefault(cancer, []).append(cond)
            for cell in cells:
                expected_cell_conditions[int(cell)].append(cond)

        if set(self.conditions_by_cancer) != set(expected_conditions_by_cancer):
            raise ValueError("conditions_by_cancer keys are inconsistent")
        for cancer, values in expected_conditions_by_cancer.items():
            expected = np.asarray(values, dtype=np.int32)
            if not np.array_equal(self.conditions_by_cancer[cancer], expected):
                raise ValueError(f"conditions_by_cancer[{cancer}] is inconsistent")
        for cell, values in enumerate(expected_cell_conditions):
            expected = np.asarray(values, dtype=np.int32)
            if not np.array_equal(self.cell_to_conditions[cell], expected):
                raise ValueError(f"cell_to_conditions[{cell}] is inconsistent")

        # Candidate-set IDs must be dense within each cancer. Otherwise a
        # malformed cache could allocate an incorrect matcher matrix.
        for cancer, sets in self.candidate_sets_by_cancer.items():
            for set_local, cells in enumerate(sets):
                if cells.ndim != 1 or len(cells) == 0:
                    raise ValueError(
                        f"candidate set {cancer}/{set_local} must be nonempty and 1D"
                    )
            used = self.condition_set_local[self.condition_cancer_idx == cancer]
            if len(sets) == 0 or not np.array_equal(
                np.unique(used), np.arange(len(sets), dtype=np.int32)
            ):
                raise ValueError(f"candidate-set IDs for cancer {cancer} are not dense")

    @classmethod
    def build(
        cls,
        *,
        cell_ids: Sequence,
        cell_cancer_idx: ArrayLike,
        treated_cell_ids: Sequence,
        treated_drugs: Sequence,
        treated_doses: ArrayLike,
        control_cell_ids: Collection,
        available_drugs: Collection | None = None,
        treated_row_indices: ArrayLike | None = None,
    ) -> "ConditionSpecificIndex":
        """Index expression metadata with unique cell/drug/dose treated rows.

        ``cell_ids`` follows the mean-aggregated cell embedding table.
        """
        cell_ids_arr = np.asarray(
            [_canonical_cell_id(x) for x in cell_ids], dtype=object
        )
        cancer_arr = np.asarray(cell_cancer_idx, dtype=np.int32)
        if len(cell_ids_arr) != len(cancer_arr):
            raise ValueError("cell_ids and cell_cancer_idx length mismatch")
        if len(set(cell_ids_arr.tolist())) != len(cell_ids_arr):
            raise ValueError("clean cell_ids must be unique")

        n_treated = len(treated_cell_ids)
        if len(treated_drugs) != n_treated or len(treated_doses) != n_treated:
            raise ValueError("treated metadata arrays must have equal length")
        if treated_row_indices is None:
            treated_rows = np.arange(n_treated, dtype=np.int64)
        else:
            treated_rows = np.asarray(treated_row_indices, dtype=np.int64)
            if len(treated_rows) != n_treated:
                raise ValueError("treated_row_indices length mismatch")

        cell_to_idx = {x: i for i, x in enumerate(cell_ids_arr.tolist())}
        control_set = {_canonical_cell_id(x) for x in control_cell_ids}
        drug_set = (
            None
            if available_drugs is None
            else {_canonical_drug(x) for x in available_drugs}
        )

        # Each value maps a candidate clean-cell index to one treated row.
        groups: dict[tuple[int, str, float], dict[int, int]] = {}
        for cid_raw, drug_raw, dose_raw, row_idx in zip(
            treated_cell_ids,
            treated_drugs,
            treated_doses,
            treated_rows,
        ):
            cid = _canonical_cell_id(cid_raw)
            cell_idx = cell_to_idx.get(cid)
            if cell_idx is None or cid not in control_set:
                continue
            drug = _canonical_drug(drug_raw)
            if drug_set is not None and drug not in drug_set:
                continue
            dose = _canonical_dose(dose_raw)
            cancer = int(cancer_arr[cell_idx])
            key = (cancer, drug, dose)
            by_cell = groups.setdefault(key, {})
            if cell_idx in by_cell:
                raise ValueError(
                    "treated rows must be mean-aggregated and unique at "
                    f"new_cid+drug+dose; duplicate key={key + (cid,)}"
                )
            by_cell[cell_idx] = int(row_idx)

        if not groups:
            raise ValueError("no eligible same-cancer drug-dose conditions were found")

        condition_keys = sorted(groups, key=lambda x: (x[0], x[1], x[2]))
        drug_names = tuple(sorted({x[1] for x in condition_keys}))
        drug_to_idx = {x: i for i, x in enumerate(drug_names)}

        condition_cancer = []
        condition_drug = []
        condition_dose = []
        condition_offsets = [0]
        flat_cells = []
        flat_rows = []
        condition_set_local = []
        set_maps: dict[int, dict[tuple[int, ...], int]] = {}
        set_lists: dict[int, list[np.ndarray]] = {}
        conditions_by_cancer_lists: dict[int, list[int]] = {}
        cell_condition_lists: list[list[int]] = [[] for _ in range(len(cell_ids_arr))]

        for condition_id, key in enumerate(condition_keys):
            cancer, drug, dose = key
            by_cell = groups[key]
            cells_tuple = tuple(sorted(by_cell))
            rows = [by_cell[c] for c in cells_tuple]

            local_map = set_maps.setdefault(cancer, {})
            local_sets = set_lists.setdefault(cancer, [])
            set_local = local_map.get(cells_tuple)
            if set_local is None:
                set_local = len(local_sets)
                local_map[cells_tuple] = set_local
                local_sets.append(np.asarray(cells_tuple, dtype=np.int32))

            condition_cancer.append(cancer)
            condition_drug.append(drug_to_idx[drug])
            condition_dose.append(dose)
            flat_cells.extend(cells_tuple)
            flat_rows.extend(rows)
            condition_offsets.append(len(flat_cells))
            condition_set_local.append(set_local)
            conditions_by_cancer_lists.setdefault(cancer, []).append(condition_id)
            for cell_idx in cells_tuple:
                cell_condition_lists[cell_idx].append(condition_id)

        return cls(
            cell_ids=cell_ids_arr,
            cell_cancer_idx=cancer_arr,
            drug_names=drug_names,
            condition_cancer_idx=np.asarray(condition_cancer, dtype=np.int32),
            condition_drug_idx=np.asarray(condition_drug, dtype=np.int32),
            condition_dose=np.asarray(condition_dose, dtype=np.float64),
            condition_offsets=np.asarray(condition_offsets, dtype=np.int64),
            condition_cells=np.asarray(flat_cells, dtype=np.int32),
            condition_treated_rows=np.asarray(flat_rows, dtype=np.int64),
            condition_set_local=np.asarray(condition_set_local, dtype=np.int32),
            candidate_sets_by_cancer={c: tuple(x) for c, x in set_lists.items()},
            conditions_by_cancer={
                c: np.asarray(x, dtype=np.int32)
                for c, x in conditions_by_cancer_lists.items()
            },
            cell_to_conditions=tuple(
                np.asarray(sorted(set(x)), dtype=np.int32) for x in cell_condition_lists
            ),
        )

    def to_payload(self) -> dict:
        """Serialize index metadata with treated-row references and a checksum."""
        return {
            "condition_index_schema_version": self.schema_version,
            "definition_sha256": self.definition_sha256(),
            "cell_ids": tuple(map(str, self.cell_ids)),
            "cell_cancer_idx": self.cell_cancer_idx,
            "drug_names": self.drug_names,
            "condition_cancer_idx": self.condition_cancer_idx,
            "condition_drug_idx": self.condition_drug_idx,
            "condition_dose": self.condition_dose,
            "condition_offsets": self.condition_offsets,
            "condition_cells": self.condition_cells,
            "condition_treated_rows": self.condition_treated_rows,
            "condition_set_local": self.condition_set_local,
            "candidate_sets_by_cancer": {
                str(cancer): tuple(sets)
                for cancer, sets in self.candidate_sets_by_cancer.items()
            },
            "conditions_by_cancer": {
                str(cancer): values
                for cancer, values in self.conditions_by_cancer.items()
            },
            "cell_to_conditions": self.cell_to_conditions,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping,
        *,
        expected_definition_sha256: str | None = None,
    ) -> "ConditionSpecificIndex":
        if int(payload.get("condition_index_schema_version", -1)) != cls.schema_version:
            raise ValueError("unsupported condition index cache schema")
        required = (
            "definition_sha256",
            "cell_ids",
            "cell_cancer_idx",
            "drug_names",
            "condition_cancer_idx",
            "condition_drug_idx",
            "condition_dose",
            "condition_offsets",
            "condition_cells",
            "condition_treated_rows",
            "condition_set_local",
            "candidate_sets_by_cancer",
            "conditions_by_cancer",
            "cell_to_conditions",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(
                "condition index payload is missing fields: " + ", ".join(missing)
            )
        out = cls(
            cell_ids=np.asarray(payload["cell_ids"], dtype=object),
            cell_cancer_idx=np.asarray(payload["cell_cancer_idx"], dtype=np.int32),
            drug_names=tuple(map(str, payload["drug_names"])),
            condition_cancer_idx=np.asarray(
                payload["condition_cancer_idx"], dtype=np.int32
            ),
            condition_drug_idx=np.asarray(
                payload["condition_drug_idx"], dtype=np.int32
            ),
            condition_dose=np.asarray(payload["condition_dose"], dtype=np.float64),
            condition_offsets=np.asarray(payload["condition_offsets"], dtype=np.int64),
            condition_cells=np.asarray(payload["condition_cells"], dtype=np.int32),
            condition_treated_rows=np.asarray(
                payload["condition_treated_rows"], dtype=np.int64
            ),
            condition_set_local=np.asarray(
                payload["condition_set_local"], dtype=np.int32
            ),
            candidate_sets_by_cancer={
                int(cancer): tuple(np.asarray(x, dtype=np.int32) for x in sets)
                for cancer, sets in payload["candidate_sets_by_cancer"].items()
            },
            conditions_by_cancer={
                int(cancer): np.asarray(values, dtype=np.int32)
                for cancer, values in payload["conditions_by_cancer"].items()
            },
            cell_to_conditions=tuple(
                np.asarray(values, dtype=np.int32)
                for values in payload["cell_to_conditions"]
            ),
        )
        stored = str(payload["definition_sha256"])
        actual = out.definition_sha256()
        if stored != actual:
            raise ValueError("condition index payload checksum mismatch")
        if (
            expected_definition_sha256 is not None
            and str(expected_definition_sha256) != actual
        ):
            raise ValueError("condition index does not match the expected definition")
        return out

    def _check_condition_id(self, condition_id: int) -> int:
        condition_id = int(condition_id)
        if condition_id < 0 or condition_id >= self.n_conditions:
            raise IndexError(f"condition_id out of range: {condition_id}")
        return condition_id

    def condition_candidates(self, condition_id: int) -> tuple[np.ndarray, np.ndarray]:
        condition_id = self._check_condition_id(condition_id)
        start = int(self.condition_offsets[condition_id])
        stop = int(self.condition_offsets[condition_id + 1])
        return (
            self.condition_cells[start:stop],
            self.condition_treated_rows[start:stop],
        )

    def condition_key(self, condition_id: int) -> tuple[int, str, float]:
        condition_id = self._check_condition_id(condition_id)
        return (
            int(self.condition_cancer_idx[condition_id]),
            self.drug_names[int(self.condition_drug_idx[condition_id])],
            float(self.condition_dose[condition_id]),
        )

    def find_condition(self, cancer_idx: int, drug: str, dose: float) -> int | None:
        return self._condition_key_to_id.get(
            (int(cancer_idx), _canonical_drug(drug), _canonical_dose(dose))
        )

    def definition_sha256(self) -> str:
        digest = hashlib.sha256(
            f"condition-index-v{self.schema_version}\0".encode("ascii")
        )
        for cid, cancer in zip(self.cell_ids, self.cell_cancer_idx):
            digest.update(str(cid).encode("utf-8"))
            digest.update(b"\0")
            digest.update(np.int32(cancer).tobytes())
        for drug in self.drug_names:
            digest.update(drug.encode("utf-8"))
            digest.update(b"\0")
        for arr in (
            self.condition_cancer_idx,
            self.condition_drug_idx,
            self.condition_dose,
            self.condition_offsets,
            self.condition_cells,
            self.condition_treated_rows,
            self.condition_set_local,
        ):
            _update_hash_array(digest, arr)
        return digest.hexdigest()


def build_anchor_union_records(
    *,
    index: ConditionSpecificIndex,
    query_patient_indices: ArrayLike,
    patient_cancer_idx: ArrayLike,
    anchor_cell_indices: Sequence[ArrayLike],
    max_records: int | None = None,
) -> tuple[CompactConditionRecords, dict]:
    """Enumerate drug-dose conditions measured by each patient's anchor cells."""
    query = np.asarray(query_patient_indices, dtype=np.int64)
    cancers = np.asarray(patient_cancer_idx, dtype=np.int32)
    if query.ndim != 1 or cancers.ndim != 1:
        raise ValueError("query_patient_indices and patient_cancer_idx must be 1D")
    if len(anchor_cell_indices) != len(query):
        raise ValueError("anchor_cell_indices must align with query patients")

    q_chunks = []
    c_chunks = []
    empty_patient_n = 0
    cross_cancer_anchor_n = 0

    for qpos, pidx in enumerate(query):
        if pidx < 0 or pidx >= len(cancers):
            raise IndexError(f"query patient index out of range: {pidx}")
        cancer = int(cancers[pidx])
        arrays = []
        anchor_row = np.asarray(anchor_cell_indices[qpos], dtype=np.int64).reshape(-1)
        for cell_idx_raw in anchor_row:
            cell_idx = int(cell_idx_raw)
            if cell_idx < 0 or cell_idx >= index.n_cells:
                raise IndexError(f"anchor cell index out of range: {cell_idx}")
            if int(index.cell_cancer_idx[cell_idx]) != cancer:
                cross_cancer_anchor_n += 1
                continue
            cond = index.cell_to_conditions[cell_idx]
            if len(cond):
                arrays.append(cond)

        if not arrays:
            empty_patient_n += 1
            continue
        condition_ids = np.unique(np.concatenate(arrays)).astype(np.int32, copy=False)
        same = index.condition_cancer_idx[condition_ids] == cancer
        condition_ids = condition_ids[same]
        if len(condition_ids) == 0:
            empty_patient_n += 1
            continue
        q_chunks.append(np.full(len(condition_ids), qpos, dtype=np.int32))
        c_chunks.append(condition_ids)

    if q_chunks:
        query_positions = np.concatenate(q_chunks)
        condition_ids = np.concatenate(c_chunks)
    else:
        query_positions = np.empty(0, dtype=np.int32)
        condition_ids = np.empty(0, dtype=np.int32)

    total_before_limit = int(len(query_positions))
    if max_records is not None:
        max_records = int(max_records)
        if max_records < 0:
            raise ValueError("max_records must be nonnegative")
        query_positions = query_positions[:max_records]
        condition_ids = condition_ids[:max_records]

    records = CompactConditionRecords(query_positions, condition_ids)
    audit = {
        "record_scope": "anchor_union",
        "query_n": int(len(query)),
        "record_n_before_limit": total_before_limit,
        "record_n": int(len(records)),
        "patients_without_eligible_anchor_condition": int(empty_patient_n),
        "cross_cancer_anchors_ignored": int(cross_cancer_anchor_n),
    }
    return records, audit


def _matcher_source_sha256(
    index: ConditionSpecificIndex,
    patient_emb: np.ndarray,
    patient_cancer_idx: np.ndarray,
    cell_emb: np.ndarray,
    active_patient_indices: np.ndarray,
) -> str:
    digest = hashlib.sha256(b"condition-specific-matcher-source-v2\0")
    digest.update(index.definition_sha256().encode("ascii"))
    _update_hash_array(
        digest,
        np.asarray([len(patient_emb), patient_emb.shape[1]], dtype=np.int64),
    )
    _update_hash_array(digest, active_patient_indices)
    _update_hash_array(digest, patient_emb[active_patient_indices])
    _update_hash_array(digest, patient_cancer_idx[active_patient_indices])
    _update_hash_array(digest, cell_emb)
    return digest.hexdigest()


def _canonical_active_patient_indices(
    active_patient_indices: ArrayLike | None,
    n_patients: int,
) -> np.ndarray:
    """Return a deterministic, unique int32 subset of global patient rows."""
    if active_patient_indices is None:
        return np.arange(n_patients, dtype=np.int32)
    active = _as_numpy(active_patient_indices, dtype=np.int64)
    if active.ndim != 1:
        raise ValueError("active_patient_indices must be 1D")
    if np.any(active < 0) or np.any(active >= n_patients):
        raise IndexError("active patient index is out of range")
    if len(np.unique(active)) != len(active):
        raise ValueError("active_patient_indices must not contain duplicates")
    return np.sort(active).astype(np.int32, copy=False)


class ConditionSpecificMatcher:
    """Precompute top-1 proxies per patient and unique candidate-cell set."""

    schema_version = 2

    def __init__(
        self,
        *,
        index: ConditionSpecificIndex,
        patient_aligned_embeddings: ArrayLike,
        patient_cancer_idx: ArrayLike,
        cell_aligned_embeddings: ArrayLike,
        keep_cosine: bool = True,
        active_patient_indices: ArrayLike | None = None,
    ):
        self.index = index
        patient_emb = _normalize_rows(patient_aligned_embeddings)
        cell_emb = _normalize_rows(cell_aligned_embeddings)
        self.patient_cancer_idx = np.asarray(patient_cancer_idx, dtype=np.int32)
        if self.patient_cancer_idx.ndim != 1:
            raise ValueError("patient_cancer_idx must be 1D")
        if len(patient_emb) != len(self.patient_cancer_idx):
            raise ValueError("patient embedding/cancer length mismatch")
        if len(cell_emb) != index.n_cells:
            raise ValueError("cell embedding rows do not align with condition index")
        if patient_emb.shape[1] != cell_emb.shape[1]:
            raise ValueError("patient and cell aligned embedding dimensions differ")

        self.keep_cosine = bool(keep_cosine)
        self.active_patient_indices = _canonical_active_patient_indices(
            active_patient_indices,
            len(patient_emb),
        )
        self.source_definition_sha256 = _matcher_source_sha256(
            index,
            patient_emb,
            self.patient_cancer_idx,
            cell_emb,
            self.active_patient_indices,
        )

        self.patient_local_row = np.full(len(patient_emb), -1, dtype=np.int32)
        active = np.zeros(len(patient_emb), dtype=bool)
        active[self.active_patient_indices] = True
        self.patient_indices_by_cancer: dict[int, np.ndarray] = {}
        self.best_rank_by_cancer: dict[int, np.ndarray] = {}
        self.best_cosine_by_cancer: dict[int, np.ndarray] = {}

        for cancer, candidate_sets in sorted(index.candidate_sets_by_cancer.items()):
            patient_indices = np.flatnonzero(
                (self.patient_cancer_idx == cancer) & active
            ).astype(np.int32)
            if len(patient_indices) == 0:
                continue
            cancer_cells = np.flatnonzero(index.cell_cancer_idx == cancer).astype(
                np.int32
            )
            if len(cancer_cells) == 0:
                raise ValueError(
                    f"cancer {cancer} has conditions but no cell embeddings"
                )

            global_to_local_cell = np.full(index.n_cells, -1, dtype=np.int32)
            global_to_local_cell[cancer_cells] = np.arange(
                len(cancer_cells), dtype=np.int32
            )
            similarity = patient_emb[patient_indices] @ cell_emb[cancer_cells].T

            max_candidates = max(len(x) for x in candidate_sets)
            if max_candidates > np.iinfo(np.uint16).max:
                raise ValueError(
                    "a candidate set is too large for compact uint16 ranks"
                )
            ranks = np.empty(
                (len(patient_indices), len(candidate_sets)), dtype=np.uint16
            )
            cosines = (
                np.empty((len(patient_indices), len(candidate_sets)), dtype=np.float32)
                if keep_cosine
                else None
            )

            row_idx = np.arange(len(patient_indices))
            for set_local, cells in enumerate(candidate_sets):
                local_cells = global_to_local_cell[cells]
                if np.any(local_cells < 0):
                    raise ValueError(
                        f"candidate set {cancer}/{set_local} crosses cancer"
                    )
                subset = similarity[:, local_cells]
                best_rank = np.argmax(subset, axis=1).astype(np.uint16)
                ranks[:, set_local] = best_rank
                if cosines is not None:
                    cosines[:, set_local] = subset[row_idx, best_rank]

            self.patient_local_row[patient_indices] = np.arange(
                len(patient_indices), dtype=np.int32
            )
            self.patient_indices_by_cancer[cancer] = patient_indices
            self.best_rank_by_cancer[cancer] = ranks
            if cosines is not None:
                self.best_cosine_by_cancer[cancer] = cosines

        self._validate_precomputed()

    def _validate_precomputed(self) -> None:
        canonical_active = _canonical_active_patient_indices(
            self.active_patient_indices,
            len(self.patient_cancer_idx),
        )
        if not np.array_equal(self.active_patient_indices, canonical_active):
            raise ValueError("matcher active patient indices are not canonical")
        active = np.zeros(len(self.patient_cancer_idx), dtype=bool)
        active[self.active_patient_indices] = True
        expected_cancers = {
            int(cancer)
            for cancer in self.index.candidate_sets_by_cancer
            if np.any((self.patient_cancer_idx == cancer) & active)
        }
        if set(self.best_rank_by_cancer) != expected_cancers:
            raise ValueError("precomputed matcher cancer keys are inconsistent")
        if set(self.patient_indices_by_cancer) != expected_cancers:
            raise ValueError("precomputed matcher patient cancer keys are inconsistent")
        if self.keep_cosine:
            if set(self.best_cosine_by_cancer) != expected_cancers:
                raise ValueError("precomputed cosine cancer keys are inconsistent")
        elif self.best_cosine_by_cancer:
            raise ValueError("matcher stores cosine arrays while keep_cosine is false")

        expected_local = np.full(len(self.patient_cancer_idx), -1, dtype=np.int32)
        for cancer in sorted(expected_cancers):
            patient_indices = np.flatnonzero(
                (self.patient_cancer_idx == cancer) & active
            ).astype(np.int32)
            if not np.array_equal(
                self.patient_indices_by_cancer.get(cancer), patient_indices
            ):
                raise ValueError(
                    f"matcher patient rows for cancer {cancer} are inconsistent"
                )
            expected_local[patient_indices] = np.arange(
                len(patient_indices), dtype=np.int32
            )
            candidate_sets = self.index.candidate_sets_by_cancer[cancer]
            ranks = self.best_rank_by_cancer[cancer]
            expected_shape = (len(patient_indices), len(candidate_sets))
            if ranks.dtype != np.uint16 or ranks.shape != expected_shape:
                raise ValueError(f"matcher rank matrix for cancer {cancer} is invalid")
            for set_local, cells in enumerate(candidate_sets):
                if np.any(ranks[:, set_local] >= len(cells)):
                    raise ValueError(
                        f"matcher rank exceeds candidate set {cancer}/{set_local}"
                    )
            if self.keep_cosine:
                cosines = self.best_cosine_by_cancer[cancer]
                if cosines.dtype != np.float32 or cosines.shape != expected_shape:
                    raise ValueError(
                        f"matcher cosine matrix for cancer {cancer} is invalid"
                    )
                if not np.isfinite(cosines).all() or np.any(np.abs(cosines) > 1.00001):
                    raise ValueError(
                        f"matcher cosine matrix for cancer {cancer} is non-finite"
                    )
        if not np.array_equal(self.patient_local_row, expected_local):
            raise ValueError("matcher patient_local_row is inconsistent")

    def to_payload(self) -> dict:
        return {
            "condition_matcher_schema_version": self.schema_version,
            "index_definition_sha256": self.index.definition_sha256(),
            "source_definition_sha256": self.source_definition_sha256,
            "definition_sha256": self.definition_sha256(),
            "keep_cosine": self.keep_cosine,
            "active_patient_indices": self.active_patient_indices,
            "best_rank_by_cancer": {
                str(cancer): values
                for cancer, values in self.best_rank_by_cancer.items()
            },
            "best_cosine_by_cancer": {
                str(cancer): values
                for cancer, values in self.best_cosine_by_cancer.items()
            },
        }

    def definition_sha256(self) -> str:
        digest = hashlib.sha256(b"condition-specific-matcher-v2\0")
        digest.update(self.index.definition_sha256().encode("ascii"))
        digest.update(self.source_definition_sha256.encode("ascii"))
        digest.update(b"\1" if self.keep_cosine else b"\0")
        _update_hash_array(digest, self.active_patient_indices)
        for cancer in sorted(self.best_rank_by_cancer):
            digest.update(np.int32(cancer).tobytes())
            _update_hash_array(digest, self.best_rank_by_cancer[cancer])
            if self.keep_cosine:
                _update_hash_array(digest, self.best_cosine_by_cancer[cancer])
        return digest.hexdigest()

    @classmethod
    def from_payload(
        cls,
        payload: Mapping,
        *,
        index: ConditionSpecificIndex,
        patient_aligned_embeddings: ArrayLike,
        patient_cancer_idx: ArrayLike,
        cell_aligned_embeddings: ArrayLike,
        active_patient_indices: ArrayLike | None = None,
    ) -> "ConditionSpecificMatcher":
        if (
            int(payload.get("condition_matcher_schema_version", -1))
            != cls.schema_version
        ):
            raise ValueError("unsupported condition matcher cache schema")
        if str(payload.get("index_definition_sha256", "")) != index.definition_sha256():
            raise ValueError("condition matcher cache uses a different condition index")

        patient_emb = _normalize_rows(patient_aligned_embeddings)
        cell_emb = _normalize_rows(cell_aligned_embeddings)
        cancers = np.asarray(patient_cancer_idx, dtype=np.int32)
        if cancers.ndim != 1 or len(cancers) != len(patient_emb):
            raise ValueError("patient matcher source metadata is invalid")
        if len(cell_emb) != index.n_cells or patient_emb.shape[1] != cell_emb.shape[1]:
            raise ValueError("matcher source embedding shapes are invalid")
        active = _canonical_active_patient_indices(
            active_patient_indices,
            len(patient_emb),
        )
        payload_active = np.asarray(
            payload.get("active_patient_indices", []), dtype=np.int32
        )
        if not np.array_equal(payload_active, active):
            raise ValueError("condition matcher cache active patient subset mismatch")
        source_hash = _matcher_source_sha256(
            index,
            patient_emb,
            cancers,
            cell_emb,
            active,
        )
        if str(payload.get("source_definition_sha256", "")) != source_hash:
            raise ValueError("condition matcher cache source fingerprint mismatch")

        out = cls.__new__(cls)
        out.index = index
        out.patient_cancer_idx = cancers
        out.keep_cosine = bool(payload.get("keep_cosine", False))
        out.active_patient_indices = active
        out.source_definition_sha256 = source_hash
        out.patient_local_row = np.full(len(cancers), -1, dtype=np.int32)
        out.patient_indices_by_cancer = {}
        active_mask = np.zeros(len(cancers), dtype=bool)
        active_mask[active] = True
        for cancer in index.candidate_sets_by_cancer:
            patient_indices = np.flatnonzero((cancers == cancer) & active_mask).astype(
                np.int32
            )
            if len(patient_indices):
                out.patient_indices_by_cancer[int(cancer)] = patient_indices
                out.patient_local_row[patient_indices] = np.arange(
                    len(patient_indices), dtype=np.int32
                )
        out.best_rank_by_cancer = {
            int(cancer): np.asarray(values, dtype=np.uint16)
            for cancer, values in payload.get("best_rank_by_cancer", {}).items()
        }
        out.best_cosine_by_cancer = {
            int(cancer): np.asarray(values, dtype=np.float32)
            for cancer, values in payload.get("best_cosine_by_cancer", {}).items()
        }
        out._validate_precomputed()
        stored_definition = str(payload.get("definition_sha256", ""))
        if stored_definition != out.definition_sha256():
            raise ValueError("condition matcher payload checksum mismatch")
        return out

    def match(self, patient_idx: int, condition_id: int) -> ProxyMatch:
        patient_idx = int(patient_idx)
        condition_id = int(condition_id)
        if patient_idx < 0 or patient_idx >= len(self.patient_cancer_idx):
            raise IndexError(f"patient_idx out of range: {patient_idx}")
        if condition_id < 0 or condition_id >= self.index.n_conditions:
            raise IndexError(f"condition_id out of range: {condition_id}")

        cancer = int(self.index.condition_cancer_idx[condition_id])
        if int(self.patient_cancer_idx[patient_idx]) != cancer:
            raise ValueError(
                "strict same-cancer matching failed: "
                f"patient_cancer={int(self.patient_cancer_idx[patient_idx])}, "
                f"condition_cancer={cancer}"
            )
        active_pos = int(np.searchsorted(self.active_patient_indices, patient_idx))
        if (
            active_pos >= len(self.active_patient_indices)
            or int(self.active_patient_indices[active_pos]) != patient_idx
        ):
            raise ValueError(
                f"patient {patient_idx} is outside the active matcher subset"
            )
        patient_local = int(self.patient_local_row[patient_idx])
        if patient_local < 0 or cancer not in self.best_rank_by_cancer:
            raise ValueError(
                f"patient {patient_idx} has no strict same-cancer proxy pool"
            )
        set_local = int(self.index.condition_set_local[condition_id])
        rank = int(self.best_rank_by_cancer[cancer][patient_local, set_local])
        cells, treated_rows = self.index.condition_candidates(condition_id)
        cosine = float("nan")
        if cancer in self.best_cosine_by_cancer:
            cosine = float(self.best_cosine_by_cancer[cancer][patient_local, set_local])
        return ProxyMatch(
            patient_idx=patient_idx,
            condition_id=condition_id,
            cell_idx=int(cells[rank]),
            treated_row_idx=int(treated_rows[rank]),
            cosine=cosine,
            candidate_n=int(len(cells)),
        )

    def storage_nbytes(self) -> int:
        total = self.active_patient_indices.nbytes + self.patient_local_row.nbytes
        total += sum(x.nbytes for x in self.patient_indices_by_cancer.values())
        total += sum(x.nbytes for x in self.best_rank_by_cancer.values())
        total += sum(x.nbytes for x in self.best_cosine_by_cancer.values())
        return int(total)


def build_fixed_proxy_mixture(
    *,
    fixed_cell_indices: ArrayLike,
    fixed_weights: ArrayLike,
    candidate_cell_indices: ArrayLike,
    candidate_treated_row_indices: ArrayLike,
    condition_id: int,
    atol: float = 1e-8,
) -> FixedProxyMixture:
    """Filter fixed proxies by condition, merge duplicates and normalize weights."""
    cells = _as_numpy(fixed_cell_indices, dtype=np.int64).reshape(-1)
    intended = _as_numpy(fixed_weights, dtype=np.float64).reshape(-1)
    candidates = _as_numpy(candidate_cell_indices, dtype=np.int64).reshape(-1)
    candidate_rows = _as_numpy(candidate_treated_row_indices, dtype=np.int64).reshape(
        -1
    )

    if len(cells) == 0 or len(cells) != len(intended):
        raise ValueError("fixed cells and weights must be nonempty aligned arrays")
    if len(candidates) == 0 or len(candidates) != len(candidate_rows):
        raise ValueError(
            "condition candidates and treated rows must be nonempty aligned arrays"
        )
    if np.any(cells < 0) or np.any(candidates < 0) or np.any(candidate_rows < 0):
        raise ValueError("cell and treated-row indices must be nonnegative")
    if not np.isfinite(intended).all() or np.any(intended < -atol):
        raise ValueError("fixed proxy weights must be finite and nonnegative")
    intended = np.maximum(intended, 0.0)
    intended_sum = float(intended.sum())
    if not np.isclose(intended_sum, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(f"fixed proxy weights must sum to one, got {intended_sum}")
    intended /= intended_sum
    if len(candidates) > 1 and np.any(candidates[1:] <= candidates[:-1]):
        raise ValueError("condition candidate cells must be strictly increasing")

    positions = np.searchsorted(candidates, cells)
    in_bounds = positions < len(candidates)
    available = np.zeros(len(cells), dtype=bool)
    available[in_bounds] = candidates[positions[in_bounds]] == cells[in_bounds]
    if not available.any():
        raise ValueError("condition is absent from every fixed proxy")

    available_cells = cells[available].astype(np.int32, copy=False)
    available_rows = candidate_rows[positions[available]].astype(np.int64, copy=False)
    available_before = intended[available]
    available_mass = float(available_before.sum())
    zero_mass_fallback = not np.isfinite(available_mass) or available_mass <= 0.0
    if zero_mass_fallback:
        # Keeping every condition covered by at least one fixed proxy was an
        # explicit design choice. Exact-zero simplex weights make ordinary
        # renormalization undefined, so use a deterministic auditable fallback.
        available_weights = np.full(
            len(available_cells), 1.0 / len(available_cells), dtype=np.float64
        )
    else:
        available_weights = available_before / available_mass

    merged_pos: dict[int, int] = {}
    merged_cells: list[int] = []
    merged_rows: list[int] = []
    merged_weights: list[float] = []
    for cell, row, weight in zip(available_cells, available_rows, available_weights):
        key = int(cell)
        pos = merged_pos.get(key)
        if pos is None:
            merged_pos[key] = len(merged_cells)
            merged_cells.append(key)
            merged_rows.append(int(row))
            merged_weights.append(float(weight))
        else:
            if merged_rows[pos] != int(row):
                raise AssertionError(
                    "one fixed cell-condition pair resolved to multiple treated rows"
                )
            merged_weights[pos] += float(weight)

    weights = np.asarray(merged_weights, dtype=np.float64)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("final fixed-proxy mixture has invalid total weight")
    weights /= total
    if np.any(weights < -atol) or not np.isclose(
        weights.sum(), 1.0, atol=1e-6, rtol=0.0
    ):
        raise AssertionError("final fixed-proxy weights are not on the simplex")

    return FixedProxyMixture(
        condition_id=int(condition_id),
        fixed_cell_indices=cells.astype(np.int32, copy=False),
        fixed_weights=intended.astype(np.float32),
        available_mask=available,
        available_cell_indices=available_cells,
        available_treated_row_indices=available_rows,
        available_weights_before_normalization=available_before.astype(np.float32),
        available_weights=available_weights.astype(np.float32),
        available_weight_mass=available_mass,
        zero_mass_fallback=bool(zero_mass_fallback),
        cell_indices=np.asarray(merged_cells, dtype=np.int32),
        treated_row_indices=np.asarray(merged_rows, dtype=np.int64),
        weights=weights.astype(np.float32),
    )


def build_proxy_mixture(
    *,
    matcher: ConditionSpecificMatcher,
    target_patient_idx: int,
    neighbor_patient_indices: ArrayLike,
    alpha: ArrayLike,
    condition_id: int,
    gate: float = 0.3,
    atol: float = 1e-6,
) -> ProxyMixture:
    """Apply ``g`` and H&E barycentric alpha, then merge duplicate proxies."""
    gate = float(gate)
    if not 0.0 <= gate <= 1.0:
        raise ValueError(f"gate must be in [0, 1], got {gate}")
    neighbors = _as_numpy(neighbor_patient_indices, dtype=np.int64)
    alpha_arr = _as_numpy(alpha, dtype=np.float64)
    if neighbors.ndim != 1 or alpha_arr.ndim != 1 or len(neighbors) != len(alpha_arr):
        raise ValueError("neighbor_patient_indices and alpha must be aligned 1D arrays")
    if len(alpha_arr) == 0:
        raise ValueError("at least one patient neighbor is required")
    if not np.isfinite(alpha_arr).all() or np.any(alpha_arr < -atol):
        raise ValueError("alpha must be finite and nonnegative")
    alpha_arr = np.maximum(alpha_arr, 0.0)
    alpha_sum = float(alpha_arr.sum())
    if not np.isclose(alpha_sum, 1.0, atol=atol, rtol=0.0):
        raise ValueError(f"alpha must sum to one, got {alpha_sum}")
    alpha_arr /= alpha_sum

    raw_patients = np.concatenate(
        [np.asarray([int(target_patient_idx)], dtype=np.int64), neighbors]
    )
    raw_weights = np.concatenate(
        [np.asarray([gate], dtype=np.float64), (1.0 - gate) * alpha_arr]
    )
    matches = [matcher.match(int(p), int(condition_id)) for p in raw_patients]
    raw_cells = np.asarray([x.cell_idx for x in matches], dtype=np.int32)
    raw_rows = np.asarray([x.treated_row_idx for x in matches], dtype=np.int64)
    raw_cosines = np.asarray([x.cosine for x in matches], dtype=np.float32)

    # Stable first-occurrence merge gives deterministic audits and naturally
    # handles target-neighbor and neighbor-neighbor proxy collisions.
    merged_pos: dict[int, int] = {}
    merged_cells = []
    merged_rows = []
    merged_weights = []
    for cell, row, weight in zip(raw_cells, raw_rows, raw_weights):
        key = int(cell)
        pos = merged_pos.get(key)
        if pos is None:
            merged_pos[key] = len(merged_cells)
            merged_cells.append(key)
            merged_rows.append(int(row))
            merged_weights.append(float(weight))
        else:
            if merged_rows[pos] != int(row):
                raise AssertionError(
                    "one cell-condition pair resolved to multiple treated rows"
                )
            merged_weights[pos] += float(weight)

    weights = np.asarray(merged_weights, dtype=np.float64)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("final proxy mixture has invalid total weight")
    weights /= total
    if np.any(weights < -atol) or not np.isclose(weights.sum(), 1.0, atol=atol):
        raise AssertionError("final proxy weights are not on the simplex")

    return ProxyMixture(
        condition_id=int(condition_id),
        raw_patient_indices=raw_patients,
        raw_cell_indices=raw_cells,
        raw_treated_row_indices=raw_rows,
        raw_cosines=raw_cosines,
        raw_weights=raw_weights.astype(np.float32),
        cell_indices=np.asarray(merged_cells, dtype=np.int32),
        treated_row_indices=np.asarray(merged_rows, dtype=np.int64),
        weights=weights.astype(np.float32),
    )


def weighted_expression_targets(
    *,
    mixture: ProxyMixture,
    treated_expression: ArrayLike,
    control_expression_by_cell: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted pseudo delta and weighted control for one mixture."""
    treated = _as_numpy(treated_expression, dtype=np.float32)
    control = _as_numpy(control_expression_by_cell, dtype=np.float32)
    if treated.ndim != 2 or control.ndim != 2:
        raise ValueError("treated and control expression must be 2D")
    if treated.shape[1] != control.shape[1]:
        raise ValueError("treated/control expression dimensions differ")
    rows = np.asarray(mixture.treated_row_indices, dtype=np.int64)
    cells = np.asarray(mixture.cell_indices, dtype=np.int64)
    weights = np.asarray(mixture.weights, dtype=np.float32)
    if rows.ndim != 1 or cells.ndim != 1 or weights.ndim != 1:
        raise ValueError("mixture rows, cells, and weights must be 1D")
    if not (len(rows) == len(cells) == len(weights)) or len(rows) == 0:
        raise ValueError(
            "mixture rows, cells, and weights must be nonempty and aligned"
        )
    if np.any(rows < 0) or np.any(rows >= len(treated)):
        raise IndexError("treated row index out of range")
    if np.any(cells < 0) or np.any(cells >= len(control)):
        raise IndexError("control cell index out of range")
    if (
        not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1.0, atol=1e-6, rtol=0.0)
    ):
        raise ValueError("mixture weights must be finite, nonnegative, and sum to one")
    w = weights[:, None]
    selected_treated = treated[rows]
    selected_control = control[cells]
    control_target = np.sum(w * selected_control, axis=0, dtype=np.float32)
    delta_target = np.sum(
        w * (selected_treated - selected_control), axis=0, dtype=np.float32
    )
    if not np.isfinite(delta_target).all() or not np.isfinite(control_target).all():
        raise ValueError("weighted expression target contains NaN or Inf")
    return delta_target, control_target
