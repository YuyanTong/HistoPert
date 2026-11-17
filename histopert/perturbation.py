"""Sliding-window cross-attention for embedding-based perturbation prediction."""

import math

import torch
from torch import Tensor, nn


class PerturbationPredictor(nn.Module):
    """Combine patient and drug windows to predict gene-expression changes."""

    def __init__(
        self,
        uce_lora: nn.Module = None,
        drug_window_size: int = 32,
        drug_slide_step: int = 16,
        cell_window_size: int = 32,
        cell_slide_step: int = 16,
        hidden_dim: int = 64,
        output_size: int = 978,
        dropout_rate: float = 0.0,
    ):
        super(PerturbationPredictor, self).__init__()

        self.uce_lora = uce_lora
        self.drug_window_size = drug_window_size
        self.drug_slide_step = drug_slide_step
        self.cell_window_size = cell_window_size
        self.cell_slide_step = cell_slide_step

        # Patient windows supply queries; drug windows supply keys and values.
        self.query_map = nn.Linear(self.cell_window_size, hidden_dim)
        self.key_map = nn.Linear(self.drug_window_size, hidden_dim)
        self.value_map = nn.Linear(self.drug_window_size, hidden_dim)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.cell_windows_num = math.floor(
            (1280 - cell_window_size) / cell_slide_step + 1
        )

        self.fusion_decoder = nn.Sequential(
            nn.Linear(
                hidden_dim * self.cell_windows_num, hidden_dim * self.cell_windows_num
            ),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(
                hidden_dim * self.cell_windows_num, hidden_dim * self.cell_windows_num
            ),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * self.cell_windows_num, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, output_size),
        )

    def predict_perturbation(self, cell_emb: Tensor, drug_emb: Tensor):
        """Predict expression changes from patient and dose-augmented drug embeddings."""
        cell_windows = torch.nn.functional.unfold(
            cell_emb.unsqueeze(1).unsqueeze(-2),
            kernel_size=(1, self.cell_window_size),
            stride=(1, self.cell_slide_step),
        ).permute(0, 2, 1)
        drug_windows = torch.nn.functional.unfold(
            drug_emb.unsqueeze(1).unsqueeze(-2),
            kernel_size=(1, self.drug_window_size),
            stride=(1, self.drug_slide_step),
        ).permute(0, 2, 1)

        # Project each window into the shared attention space.
        attention_queries = self.query_map(cell_windows)
        attention_keys = self.key_map(drug_windows)
        attention_values = self.value_map(drug_windows)

        # Apply dropout after the query, key and value projections.
        attention_queries = self.dropout(attention_queries)
        attention_keys = self.dropout(attention_keys)
        attention_values = self.dropout(attention_values)

        # Aggregate drug values for each patient window.
        attention_weights = torch.softmax(
            torch.bmm(attention_queries, attention_keys.transpose(1, 2))
            / attention_keys.size(-1) ** 0.5,
            dim=-1,
        )
        fused_context = torch.bmm(attention_weights, attention_values)
        flattened_context = fused_context.flatten(start_dim=1)
        fused_context = self.fusion_decoder(flattened_context)
        predicted_delta = self.decoder(fused_context)
        return predicted_delta

    def predict_perturbation_with_pool(self, cell_emb: Tensor, drug_emb: Tensor):
        """Return expression predictions and concatenated mean/max attention pooling."""
        cell_windows = torch.nn.functional.unfold(
            cell_emb.unsqueeze(1).unsqueeze(-2),
            kernel_size=(1, self.cell_window_size),
            stride=(1, self.cell_slide_step),
        ).permute(0, 2, 1)
        drug_windows = torch.nn.functional.unfold(
            drug_emb.unsqueeze(1).unsqueeze(-2),
            kernel_size=(1, self.drug_window_size),
            stride=(1, self.drug_slide_step),
        ).permute(0, 2, 1)

        attention_queries = self.query_map(cell_windows)
        attention_keys = self.key_map(drug_windows)
        attention_values = self.value_map(drug_windows)

        attention_queries = self.dropout(attention_queries)
        attention_keys = self.dropout(attention_keys)
        attention_values = self.dropout(attention_values)

        attention_weights = torch.softmax(
            torch.bmm(attention_queries, attention_keys.transpose(1, 2))
            / attention_keys.size(-1) ** 0.5,
            dim=-1,
        )
        attended_values = torch.bmm(attention_weights, attention_values)

        mean_pooled = attended_values.mean(dim=1)
        max_pooled = attended_values.max(dim=1).values
        pooled_embedding = torch.cat([mean_pooled, max_pooled], dim=1)

        fusion_embedding = self.fusion_decoder(attended_values.flatten(start_dim=1))
        predicted_delta = self.decoder(fusion_embedding)
        return predicted_delta, pooled_embedding

    def forward(self, mode, *args, **kwargs):
        if mode == "pertrub_forward":
            return self.pertrub_forward(*args, **kwargs)
        if mode == "pertrub_forward_with_pool_embedding":
            return self.pertrub_forward_with_pool_embedding(*args, **kwargs)
        raise ValueError(f"Unsupported embedding-only forward mode: {mode}")

    # Retain the established dispatch methods for existing callers.
    pertrub_forward = predict_perturbation
    pertrub_forward_with_pool_embedding = predict_perturbation_with_pool


# Public aliases preserve compatibility with existing integrations.
UniCure = PerturbationPredictor
