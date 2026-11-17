"""Validate feature matrices, patient partitions and expression-table schemas."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .data import feature_table, load_cell_cancer_map, load_patient_embeddings


def validate_inputs(config):
    data = config["data"]
    for key, value in data.items():
        if not Path(value).exists():
            raise FileNotFoundError(f"{key}: {value}")
    image_embeddings, gene_embeddings, patients, projects = load_patient_embeddings(
        **{
            k: data[k]
            for k in (
                "features_output_dir",
                "txt_file_path",
                "patient_csv_path",
                "patient_uce_path",
            )
        }
    )
    if image_embeddings.shape[1] != 768 or gene_embeddings.shape[1] != 1280:
        raise ValueError("Expected H&E=768 and patient UCE=1280 features")
    split = json.loads(Path(data["split_path"]).read_text())
    seen = set()
    counts = {}
    for name in ("main_train", "main_val", "main_test"):
        ids = split[f"{name}_patient_ids"]
        if not ids or len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError(f"Empty, duplicated or overlapping {name} patients")
        indices = split[f"{name}_idx"]
        if any(not isinstance(i, int) or i < 0 or i >= len(patients) for i in indices):
            raise ValueError(f"Invalid {name} indices")
        if [patients[i] for i in indices] != ids:
            raise ValueError(
                f"{name} indices disagree with patient IDs; preserve the supplied ordering"
            )
        seen.update(ids)
        counts[name] = len(ids)
    if seen != set(patients):
        raise ValueError(
            "The fixed split must cover exactly the aligned patient cohort"
        )
    index = pd.read_csv(data["patient_text_embedding_index_path"], dtype=str)
    context_embeddings = np.load(data["patient_text_embedding_path"], mmap_mode="r")
    text_dim = config.get("model", {}).get("patient_text_dim", 1024)
    if "patient_id" not in index or index.patient_id.duplicated().any():
        raise ValueError("Context index requires unique patient_id values")
    if (
        context_embeddings.shape != (len(index), text_dim)
        or not np.isfinite(context_embeddings).all()
    ):
        raise ValueError("Context embedding shape/values do not match its index")
    if set(patients) - set(index.patient_id):
        raise ValueError("Context embeddings do not cover all patients")
    cells = feature_table(data["cell_emb_path"], unique=False)
    if cells.shape[1] != 1280:
        raise ValueError("Expected LINCS UCE=1280 features")
    load_cell_cancer_map(data["cell_cancer_path"], list(cells.index))
    drugs = feature_table(data["drug_embedding_path"])
    if drugs.shape[1] != 512:
        raise ValueError("Expected drug=512 features before dose encoding")
    # Read only parquet metadata for the multi-GB expression inputs here.
    treated = pq.ParquetFile(data["cell_drug_dose_expr_parquet"])
    control = pq.ParquetFile(data["cell_control_parquet"])
    metadata = json.loads(treated.schema_arrow.metadata[b"pandas"])
    index_columns = metadata["index_columns"]
    if len(index_columns) != 3 or any(not isinstance(x, str) for x in index_columns):
        raise ValueError("Treated expression needs a 3-level cell/drug/dose MultiIndex")
    genes = [x for x in treated.schema_arrow.names if x not in index_columns]
    output_dim = config.get("model", {}).get("stage3_output_dim", 978)
    if len(genes) != output_dim or len(set(genes)) != len(genes):
        raise ValueError("Treated gene dimension/order is invalid")
    if "new_cid" not in control.schema_arrow.names or set(genes) - set(
        control.schema_arrow.names
    ):
        raise ValueError(
            "Control expression must contain new_cid and every treated gene"
        )
    train_ids = set(split["main_train_patient_ids"])
    for project in set(projects):
        training_patient_count = sum(
            patient_id in train_ids and patient_project == project
            for patient_id, patient_project in zip(patients, projects)
        )
        if training_patient_count < 6:
            raise ValueError(
                f"{project} needs at least 6 training patients for 5 neighbors excluding self"
            )
    return {
        "patients": len(patients),
        "splits": counts,
        "he_dim": image_embeddings.shape[1],
        "uce_dim": gene_embeddings.shape[1],
        "context_dim": text_dim,
        "cell_embedding_rows": len(cells),
        "cell_ids": cells.index.nunique(),
        "drug_embedding_rows": len(drugs),
        "drug_dim": drugs.shape[1],
        "genes": len(genes),
        "treated_rows": treated.metadata.num_rows,
        "control_rows": control.metadata.num_rows,
        "validation": "Features, patient split and expression schemas checked; no full expression scan or training",
    }
