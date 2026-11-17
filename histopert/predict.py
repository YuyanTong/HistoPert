"""Predict perturbations from H&E, patient context, drug features and dose."""

import numpy as np
import pandas as pd
import torch

from .data import drug_with_dose, feature_table, load_image_features
from .fusion import PatientContextModel
from .io import load_checkpoint, output_path
from .perturbation import PerturbationPredictor


def load_predictor(checkpoint, device="cpu"):
    checkpoint = load_checkpoint(checkpoint)
    config = checkpoint["config"]
    model_spec = config["model_spec"]
    model = PatientContextModel(
        img_dim=model_spec["img_dim"],
        gene_dim=model_spec["gene_dim"],
        cell_dim=model_spec["cell_dim"],
        drug_dim=1,
        output_dim=1,
        clin_dim=0,
        use_clinical=False,
        num_cancers=model_spec["num_cancers"],
        hidden_dim=model_spec["hidden_dim"],
        cancer_emb_dim=model_spec["cancer_emb_dim"],
        dropout=model_spec["dropout"],
        use_patient_text=True,
        patient_text_dim=model_spec["patient_text_dim"],
        unicure_model=PerturbationPredictor(
            output_size=model_spec["output_dim"],
            hidden_dim=model_spec["perturb_hidden_dim"],
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    genes = list(map(str, config["gene_names"]))
    if len(genes) != model_spec["output_dim"] or len(set(genes)) != len(genes):
        raise ValueError("Invalid gene labels in checkpoint")
    return model, genes


def predict(
    checkpoint,
    pairs_path,
    he_path,
    context_path,
    context_index_path,
    drug_path,
    destination,
    device="cpu",
    batch_size=64,
):
    destination = output_path(destination)
    if destination.exists():
        raise FileExistsError(f"Prediction output already exists: {destination}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    pairs = pd.read_csv(
        pairs_path, dtype={"patient_id": str, "drug_id": str, "he_id": str}
    )
    if not {"patient_id", "drug_id", "dose"}.issubset(pairs.columns) or pairs.empty:
        raise ValueError(
            "Pairs CSV needs patient_id, drug_id, dose and at least one row"
        )
    if pairs[["patient_id", "drug_id", "dose"]].isna().any().any():
        raise ValueError("Missing prediction inputs")
    context = np.load(context_path)
    index = pd.read_csv(context_index_path, dtype=str)
    if (
        "patient_id" not in index
        or index.patient_id.duplicated().any()
        or len(index) != len(context)
    ):
        raise ValueError("Context index must uniquely match embedding rows")
    if context.ndim != 2 or not np.isfinite(context).all():
        raise ValueError("Invalid context feature matrix")
    context_row_by_patient = {
        patient_id: i for i, patient_id in enumerate(index.patient_id)
    }
    if set(pairs.patient_id) - set(context_row_by_patient):
        raise ValueError("Missing patient context embeddings")
    drugs = feature_table(drug_path)
    if set(pairs.drug_id) - set(drugs.index):
        raise ValueError("Missing drug embeddings")
    he_ids = pairs["he_id"] if "he_id" in pairs else pairs.patient_id
    image_embeddings = load_image_features(he_path, he_ids.tolist())
    context_embeddings = torch.tensor(
        context[
            [context_row_by_patient[patient_id] for patient_id in pairs.patient_id]
        ],
        dtype=torch.float32,
    )
    drug_embeddings = torch.tensor(
        np.stack(
            [
                drug_with_dose(drugs.loc[row.drug_id].to_numpy(), float(row.dose))
                for row in pairs.itertuples(index=False)
            ]
        )
    )
    model, genes = load_predictor(checkpoint, device)
    prediction_batches = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = slice(start, start + batch_size)
            predicted_delta, _, _ = model.predict_delta_with_text_fusion(
                image_embeddings[batch].to(device),
                context_embeddings[batch].to(device),
                drug_embeddings[batch].to(device),
            )
            prediction_batches.append(predicted_delta.cpu().numpy())
    values = np.concatenate(prediction_batches)
    if not np.isfinite(values).all():
        raise ValueError("Model produced nonfinite predictions")
    predictions = pd.DataFrame(values, columns=genes)
    predictions.index = pd.MultiIndex.from_frame(
        pairs[["patient_id", "drug_id", "dose"]]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(destination)
    return {"rows": len(predictions), "genes": len(genes), "output": str(destination)}
