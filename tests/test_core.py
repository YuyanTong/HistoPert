"""Core model, target construction and output-path tests."""

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from histopert.condition_specific_store import build_fixed_proxy_mixture
from histopert.data import drug_with_dose
from histopert.fusion import PatientContextModel
from histopert.io import RELEASE_ROOT, output_path
from histopert.lbpt_solver import solve_lbpt_simplex
from histopert.perturbation import PerturbationPredictor
from histopert.stage3 import (
    compute_topk_neighbors_same_project_patient,
    configure_stage3_parameters,
)


class CoreTests(unittest.TestCase):
    def test_random_predictor_trains_without_loading_weights(self):
        torch.manual_seed(7)
        with patch(
            "torch.load", side_effect=AssertionError("Unexpected checkpoint dependency")
        ):
            model = PatientContextModel(
                img_dim=768,
                gene_dim=1280,
                cell_dim=1280,
                drug_dim=1,
                output_dim=1,
                clin_dim=0,
                use_clinical=False,
                hidden_dim=16,
                num_cancers=2,
                use_patient_text=True,
                patient_text_dim=1024,
                unicure_model=PerturbationPredictor(hidden_dim=4, output_size=978),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                configure_stage3_parameters(model)
            loss = model.stage23_text_fusion_losses(
                img=torch.randn(2, 768),
                gene=torch.randn(2, 1280),
                text_emb=torch.randn(2, 1024),
                drug_emb=torch.randn(2, 528),
                delta_target=torch.randn(2, 978),
            )["loss"]
            loss.backward()
        self.assertIsNone(model.unicure_model.uce_lora)
        self.assertTrue(all(p.grad is None for p in model.tcga_align_head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.img_align_head.parameters()))
        for module in [model.patient_context_fusion, model.unicure_model]:
            self.assertTrue(
                all(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in module.parameters()
                )
            )

    def test_lbpt_simplex_reconstructs_symmetric_query(self):
        query = np.array([1.0, 0.0])
        neighbors = np.array([[0.8, 0.6], [0.8, -0.6]])
        result = solve_lbpt_simplex(query, neighbors)
        np.testing.assert_allclose(result.alpha, [0.5, 0.5], atol=1e-6)
        self.assertTrue((result.alpha >= 0).all())

    def test_neighbor_retrieval_excludes_self_and_other_projects(self):
        query = torch.tensor([[1.0, 0.0]])
        candidates = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.8, -0.6], [1.0, 0.0]])
        indices, alpha, _, _ = compute_topk_neighbors_same_project_patient(
            query,
            np.array(["TCGA-BRCA"]),
            ["P1"],
            candidates,
            np.array(["TCGA-BRCA"] * 3 + ["TCGA-ACC"]),
            ["P1", "P2", "P3", "P4"],
            k=2,
        )
        self.assertEqual(set(indices[0]), {1, 2})
        np.testing.assert_allclose(alpha.sum(axis=1), [1.0])

    def test_condition_filter_renormalizes_and_merges_proxy_collisions(self):
        result = build_fixed_proxy_mixture(
            fixed_cell_indices=[0, 1, 1, 2],
            fixed_weights=[0.5, 0.2, 0.2, 0.1],
            candidate_cell_indices=[1, 2, 3],
            candidate_treated_row_indices=[10, 20, 30],
            condition_id=4,
        )
        np.testing.assert_array_equal(result.cell_indices, [1, 2])
        np.testing.assert_allclose(result.weights, [0.8, 0.2])
        self.assertEqual(result.collision_n, 1)
        self.assertEqual(result.missing_n, 1)

    def test_missing_condition_does_not_select_unrelated_cell(self):
        with self.assertRaises(ValueError):
            build_fixed_proxy_mixture(
                fixed_cell_indices=[0],
                fixed_weights=[1.0],
                candidate_cell_indices=[1],
                candidate_treated_row_indices=[2],
                condition_id=0,
            )

    def test_dose_channels_preserve_drug_embedding(self):
        drug = np.arange(512, dtype=np.float32)
        result = drug_with_dose(drug, 9)
        np.testing.assert_array_equal(result[:512], drug)
        np.testing.assert_allclose(result[512:], np.ones(16))
        with self.assertRaises(ValueError):
            drug_with_dose(drug, -1)

    def test_output_cannot_escape_new_release(self):
        with self.assertRaises(ValueError):
            output_path(RELEASE_ROOT.parent / "A_final_case_results" / "overwrite.pt")
        with self.assertRaises(ValueError):
            output_path(RELEASE_ROOT / "outputs" / ".." / "histopert" / "overwrite.py")
        self.assertEqual(
            output_path(RELEASE_ROOT / "outputs" / "example.pt"),
            RELEASE_ROOT / "outputs" / "example.pt",
        )


if __name__ == "__main__":
    unittest.main()
