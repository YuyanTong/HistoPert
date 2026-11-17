"""Generate synthetic inputs for the three-stage training example."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "examples" / "demo_data"
    if folder.exists():
        raise FileExistsError(f"Demo data already exists: {folder}")
    folder.mkdir()
    rng = np.random.default_rng(11)
    patients = [f"TCGA-DE-{i:04d}" for i in range(28)]
    projects = ["TCGA-BRCA"] * 14 + ["TCGA-ACC"] * 14
    genes = [f"GENE_{i:04d}" for i in range(978)]

    def features(ids, dim, name):
        pd.DataFrame(
            rng.normal(size=(len(ids), dim)).astype(np.float32),
            index=ids,
            columns=[f"f{i}" for i in range(dim)],
        ).to_parquet(folder / name)

    features(patients, 768, "he.parquet")
    features(patients, 1280, "patient_uce.parquet")
    np.save(folder / "context.npy", rng.normal(size=(28, 1024)).astype(np.float32))
    pd.DataFrame({"patient_id": patients}).to_csv(
        folder / "context_index.csv", index=False
    )
    pd.DataFrame({"patient_id": patients, "project_id": projects}).to_csv(
        folder / "patients.csv", index=False
    )
    (folder / "patient_ids.txt").write_text("\n".join(patients) + "\n")
    cells = ["CELL_B1", "CELL_B2", "CELL_L1", "CELL_L2"]
    features([c for c in cells for _ in range(2)], 1280, "cell_uce.parquet")
    pd.DataFrame(
        {
            "new_cid": cells,
            "cancer_type": ["Invasive Breast Carcinoma"] * 2
            + ["Non-Small Cell Lung Cancer"] * 2,
        }
    ).to_csv(folder / "cell_cancers.csv", index=False)
    drugs = ["DRUG_A", "DRUG_B"]
    features(drugs, 512, "drugs.parquet")
    control = rng.normal(size=(4, 978)).astype(np.float32)
    control_frame = pd.DataFrame(control, columns=genes)
    control_frame.insert(0, "new_cid", cells)
    control_frame.to_parquet(folder / "control.parquet", index=False)
    rows, index = [], []
    for i, cell in enumerate(cells):
        for j, drug in enumerate(drugs):
            for dose in (1.0, 10.0):
                index.append((cell, drug, dose))
                rows.append(control[i] + rng.normal(0, 0.2, 978) + 0.05 * j)
    pd.DataFrame(
        np.asarray(rows, dtype=np.float32),
        columns=genes,
        index=pd.MultiIndex.from_tuples(
            index, names=["new_cid", "cmap_name", "pert_idose"]
        ),
    ).to_parquet(folder / "treated.parquet")
    split = {"seed": 11, "split_note": "Synthetic training example"}
    for name, indices in {
        "main_train": list(range(10)) + list(range(14, 24)),
        "main_val": [10, 11, 24, 25],
        "main_test": [12, 13, 26, 27],
    }.items():
        split[name + "_idx"] = indices
        split[name + "_patient_ids"] = [patients[i] for i in indices]
    (folder / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    pd.DataFrame(
        {
            "patient_id": [patients[12], patients[26]],
            "drug_id": drugs,
            "dose": [1.0, 10.0],
        }
    ).to_csv(folder / "prediction_pairs.csv", index=False)
    print(f"Synthetic example created: {folder}")


if __name__ == "__main__":
    main()
