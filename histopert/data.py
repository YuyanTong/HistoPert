"""Read, validate and align precomputed patient and drug representations."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def case_id(value):
    parts = str(value).strip().split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else str(value).strip()


def feature_table(path, *, unique=True):
    """Read a finite numeric feature table and validate its index."""
    table = pd.read_parquet(path)
    table.index = table.index.astype(str)
    if unique and not table.index.is_unique:
        raise ValueError(f"Duplicate feature IDs: {path}")
    values = table.to_numpy(dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError(f"Features must be a finite numeric matrix: {path}")
    return table


def load_image_features(path, sample_ids):
    """Load H&E features from indexed parquet rows or individual tensor files."""
    path = Path(path)
    table = feature_table(path) if path.is_file() else None
    vectors = []
    for sample_id in sample_ids:
        if table is not None:
            feature_id = sample_id if sample_id in table.index else case_id(sample_id)
            if feature_id not in table.index:
                raise ValueError(f"H&E embedding missing for {sample_id}")
            vector = torch.tensor(table.loc[feature_id].to_numpy(dtype=np.float32))
        else:
            vector = torch.load(
                path / f"{sample_id}.pt", map_location="cpu", weights_only=True
            )
            if not isinstance(vector, torch.Tensor):
                raise ValueError(f"H&E file must contain a tensor: {sample_id}")
            vector = vector.float().reshape(-1)
        if not torch.isfinite(vector).all():
            raise ValueError(f"Nonfinite H&E embedding for {sample_id}")
        vectors.append(vector)
    if not vectors:
        raise ValueError("No H&E embeddings selected")
    if len({v.numel() for v in vectors}) != 1:
        raise ValueError("Inconsistent H&E feature dimensions")
    return torch.stack(vectors)


def load_patient_embeddings(
    features_output_dir,
    txt_file_path,
    patient_csv_path,
    patient_uce_path,
    id_mode="case",
    dedup_case=True,
):
    """Align H&E, molecular and project metadata by unique TCGA case ID."""
    if id_mode != "case" or not dedup_case:
        raise ValueError("This release uses unique patient/case IDs")
    sample_ids = [
        v.strip() for v in Path(txt_file_path).read_text().splitlines() if v.strip()
    ]
    gene_table = feature_table(patient_uce_path)
    metadata = pd.read_csv(patient_csv_path, dtype=str)
    if metadata.shape[1] < 2:
        raise ValueError("Patient metadata needs patient_id and project_id columns")
    patient_id_column, project_column = metadata.columns[:2]
    patient_metadata = metadata[[patient_id_column, project_column]].drop_duplicates()
    if patient_metadata[patient_id_column].duplicated().any():
        raise ValueError("Conflicting project assignments in patient metadata")
    patient_to_project = dict(
        zip(
            patient_metadata[patient_id_column].str.strip(),
            patient_metadata[project_column].str.strip(),
        )
    )
    image_path = Path(features_output_dir)
    image_index = (
        set(pd.read_parquet(image_path, columns=[]).index.astype(str))
        if image_path.is_file()
        else None
    )
    (
        selected_sample_ids,
        patients,
        selected_gene_ids,
        patient_projects,
        seen_patient_ids,
    ) = [], [], [], [], set()
    skipped_samples = 0
    # Select the first eligible slide for each case, preserving input order.
    for sample_id in sample_ids:
        patient_id = case_id(sample_id)
        if patient_id in seen_patient_ids:
            continue
        image_exists = (
            (sample_id in image_index or patient_id in image_index)
            if image_index is not None
            else (image_path / f"{sample_id}.pt").is_file()
        )
        gene_key = patient_id if patient_id in gene_table.index else sample_id
        if not image_exists or gene_key not in gene_table.index:
            skipped_samples += 1
            continue
        cancer = patient_to_project.get(patient_id, patient_to_project.get(sample_id))
        if cancer is None:
            raise ValueError(f"Missing patient project: {patient_id}")
        selected_sample_ids.append(sample_id)
        patients.append(patient_id)
        selected_gene_ids.append(gene_key)
        patient_projects.append(cancer)
        seen_patient_ids.add(patient_id)
    image_embeddings = load_image_features(image_path, selected_sample_ids)
    gene_embeddings = torch.tensor(
        gene_table.loc[selected_gene_ids].to_numpy(dtype=np.float32)
    )
    print(
        f"[features] {len(patients)} aligned patients; {skipped_samples} unmatched slides skipped"
    )
    return image_embeddings, gene_embeddings, patients, patient_projects


def load_cell_cancer_map(cell_cancer_path, cell_ids):
    """Map cell identifiers to their annotated cancer groups."""
    table = pd.read_csv(cell_cancer_path, dtype=str)
    required = {"new_cid", "cancer_type"}
    if not required.issubset(table.columns):
        raise ValueError("Cell metadata needs new_cid and cancer_type columns")
    table = table[["new_cid", "cancer_type"]].drop_duplicates()
    if table["new_cid"].duplicated().any():
        raise ValueError("Conflicting cancer labels for a cell")
    mapping = dict(zip(table.new_cid.str.strip(), table.cancer_type.str.strip()))
    return {str(cid): mapping.get(str(cid), "Unknown") for cid in cell_ids}


def build_shared_cancer_label_encoder(patient_cancers, cell_cancers):
    labels = sorted(
        set(map(str, patient_cancers))
        | set(map(str, cell_cancers))
        | {"Unknown", "UNMATCHED"}
    )
    return {label: i for i, label in enumerate(labels)}


def encode_cancer_labels(labels, encoder):
    return np.asarray(
        [encoder.get(str(x), encoder.get("Unknown", 0)) for x in labels], dtype=np.int64
    )


def load_clinical_features_for_patients(patient_ids, **kwargs):
    # Context embeddings are loaded separately; return an empty clinical tensor.
    return torch.zeros((len(patient_ids), 0), dtype=torch.float32)


def drug_with_dose(base, dose):
    """Append sixteen log-scaled dose channels to a 512-dimensional drug vector."""
    base = np.asarray(base, dtype=np.float32).reshape(-1)
    if base.shape != (512,) or not np.isfinite(base).all():
        raise ValueError("Expected a finite 512-dimensional drug embedding")
    if not np.isfinite(dose) or dose < 0:
        raise ValueError(
            "Dose must be finite and nonnegative, in the training table's units"
        )
    # Same encoding as the condition-specific training dataset: 512 + 16 channels.
    return np.concatenate(
        [base, np.full(16, np.log10(float(dose) + 1.0), dtype=np.float32)]
    )
