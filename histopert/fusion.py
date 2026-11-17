"""Patient-context fusion and transcriptomic perturbation prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .alignment import PatientAlignmentModel


class PatientContextModel(PatientAlignmentModel):
    """Fuse aligned H&E and patient context for perturbation prediction."""

    def __init__(
        self,
        *args,
        use_patient_text: bool = True,
        patient_text_dim: int = 1024,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.use_patient_text = bool(use_patient_text)
        self.patient_text_dim = int(patient_text_dim)

        # Shared alignment dimensions are defined by the parent model.
        if not hasattr(self, "cell_dim"):
            raise AttributeError(
                "PatientAlignmentModel must define cell_dim."
            )

        self.cell_dim = int(self.cell_dim)
        dropout = float(kwargs.get("dropout", 0.1))

        if self.use_patient_text:
            self.patient_context_fusion = nn.Sequential(
                nn.Linear(self.cell_dim + self.patient_text_dim, self.cell_dim),
                nn.LayerNorm(self.cell_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.cell_dim, self.cell_dim),
                nn.LayerNorm(self.cell_dim),
                nn.GELU(),
            )
        else:
            self.patient_context_fusion = None

    def fuse_patient_context(
        self,
        cell_emb: torch.Tensor,
        text_emb: torch.Tensor | None,
    ):
        """cell_emb: [B, cell_dim]
        text_emb: [B, patient_text_dim]
        return:   [B, cell_dim]
        """
        if not self.use_patient_text:
            return cell_emb

        if self.patient_context_fusion is None:
            return cell_emb

        if text_emb is None:
            raise ValueError("use_patient_text=True but text_emb is None")

        text_emb = text_emb.to(device=cell_emb.device, dtype=cell_emb.dtype)

        if text_emb.ndim != 2:
            raise ValueError(f"text_emb must be 2D, got shape {tuple(text_emb.shape)}")

        if text_emb.shape[0] != cell_emb.shape[0]:
            raise ValueError(
                f"Batch size mismatch: cell_emb batch={cell_emb.shape[0]}, "
                f"text_emb batch={text_emb.shape[0]}"
            )

        if text_emb.shape[1] != self.patient_text_dim:
            raise ValueError(
                f"text_emb dim mismatch: expected {self.patient_text_dim}, "
                f"got {text_emb.shape[1]}"
            )

        patient_context = torch.cat([cell_emb, text_emb], dim=1)
        return self.patient_context_fusion(patient_context)

    def unfreeze_stage3_text_fusion(self):
        """Train context fusion and perturbation prediction while fixing both alignment heads."""
        self.freeze_all()

        # Enable the perturbation network.
        if self.unicure_model is not None:
            for parameter in self.unicure_model.parameters():
                parameter.requires_grad = True

        # Enable patient-context fusion.
        if self.patient_context_fusion is not None:
            for parameter in self.patient_context_fusion.parameters():
                parameter.requires_grad = True

        # Keep molecular alignment fixed.
        if hasattr(self, "tcga_align_head"):
            for parameter in self.tcga_align_head.parameters():
                parameter.requires_grad = False

        # Keep image alignment fixed.
        if hasattr(self, "img_align_head"):
            for parameter in self.img_align_head.parameters():
                parameter.requires_grad = False

    def predict_delta_with_text_fusion(
        self,
        img: torch.Tensor,
        text_emb: torch.Tensor,
        drug_emb: torch.Tensor,
    ):
        """Return predicted changes, fused patient features and aligned H&E."""
        if self.unicure_model is None:
            raise ValueError(
                "Attach a perturbation network before Stage 3."
            )

        # Encode H&E with the frozen alignment head.
        with torch.no_grad():
            aligned_image = self.align_img_to_cell(img)

        contextualized_embedding = self.fuse_patient_context(aligned_image, text_emb)

        predicted_delta = self.unicure_model(
            "pertrub_forward",
            contextualized_embedding,
            drug_emb,
        )

        return predicted_delta, contextualized_embedding, aligned_image

    def predict_delta_with_text_fusion_and_pool(
        self,
        img: torch.Tensor,
        text_emb: torch.Tensor,
        drug_emb: torch.Tensor,
    ):
        """Return predicted changes, fused features, aligned H&E and attention pooling."""
        if self.unicure_model is None:
            raise ValueError(
                "Attach a perturbation network before Stage 3."
            )

        # Stage2 is frozen in TextFusion Stage3 inference
        with torch.no_grad():
            aligned_image = self.align_img_to_cell(img)

        # patient context fusion: image patient embedding + text embedding
        contextualized_embedding = self.fuse_patient_context(aligned_image, text_emb)

        # UniCure internal patient-drug fusion pool
        predicted_delta, pooled_embedding = self.unicure_model(
            "pertrub_forward_with_pool_embedding",
            contextualized_embedding,
            drug_emb,
        )

        return (
            predicted_delta,
            contextualized_embedding,
            aligned_image,
            pooled_embedding,
        )

    def stage23_text_fusion_losses(
        self,
        img: torch.Tensor,
        gene: torch.Tensor,
        text_emb: torch.Tensor,
        drug_emb: torch.Tensor,
        delta_target: torch.Tensor,
        lambda_mse: float = 1.0,
        lambda_cos: float = 0.5,
        lambda_img_align: float = 0.1,
        lambda_img_cos: float = 0.05,
    ):
        """Compute expression MSE, cosine loss and alignment monitoring terms."""
        predicted_delta, contextualized_embedding, aligned_image = (
            self.predict_delta_with_text_fusion(
                img=img,
                text_emb=text_emb,
                drug_emb=drug_emb,
            )
        )

        with torch.no_grad():
            aligned_gene = self.align_tcga_to_cell(gene)

            loss_img_align = F.mse_loss(aligned_image, aligned_gene)
            loss_img_cos = (
                1.0
                - F.cosine_similarity(
                    aligned_image,
                    aligned_gene,
                    dim=1,
                    eps=1e-8,
                ).mean()
            )

        loss_mse = F.mse_loss(predicted_delta, delta_target)

        loss_cos = (
            1.0
            - F.cosine_similarity(
                predicted_delta,
                delta_target,
                dim=1,
                eps=1e-8,
            ).mean()
        )

        # Alignment terms provide monitoring values without updating the encoders.

        loss = (
            lambda_mse * loss_mse
            + lambda_cos * loss_cos
            + lambda_img_align * loss_img_align
            + lambda_img_cos * loss_img_cos
        )

        return {
            "loss": loss,
            "loss_mse": loss_mse,
            "loss_cos": loss_cos,
            "loss_img_align": loss_img_align,
            "loss_img_cos": loss_img_cos,
            "pred_delta": predicted_delta,
            "cell_emb_context": contextualized_embedding,
            "img_aligned": aligned_image,
            "gene_aligned": aligned_gene,
        }


# Public aliases preserve compatibility with existing integrations.
PatientDrugResponseModel_V3_TextFusion = PatientContextModel
