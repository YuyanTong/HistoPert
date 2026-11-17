# Input Data

Set paths in `configs/tcga.json` using this layout:

```text
data/
  splits/tcga_main_split.json
  tcga/
    he.parquet
    patient_ids.txt
    patients.csv
    patient_uce.parquet
    context.npy
    context_index.csv
  lincs/
    cell_uce.parquet
    cell_cancers.csv
    drugs.parquet
    treated.parquet
    control.parquet
```

| File | Format |
| --- | --- |
| `he.parquet` | Patient/slide index; 768 feature columns. A directory of per-slide `.pt` tensors is also supported. |
| `patient_ids.txt` | One patient/slide ID per line, in loading order. |
| `patients.csv` | First two columns: `patient_id,project_id`. |
| `patient_uce.parquet` | Unique case index; 1,280 feature columns. |
| `context.npy` | N × 1,024 array, ordered as `context_index.csv`. |
| `context_index.csv` | Unique `patient_id` column. |
| `cell_uce.parquet` | Cell `new_cid` index; 1,280 feature columns per control sample. |
| `cell_cancers.csv` | `new_cid,cancer_type` using LINCS cancer annotations. |
| `drugs.parquet` | Unique drug index matching `cmap_name`; 512 feature columns. |
| `treated.parquet` | MultiIndex: `new_cid,cmap_name,pert_idose`; 978 gene columns. |
| `control.parquet` | `new_cid` and the same 978 gene columns. |

Use preprocessed expression values and finite, nonnegative doses in the training table's units.

## Patient Split

All stages use the patient IDs and indices in `splits/tcga_main_split.json`.

## Prediction

Provide the final checkpoint, H&E, context and drug embeddings, and a CSV with `patient_id,drug_id,dose`. Add `he_id` when H&E features use slide IDs.
