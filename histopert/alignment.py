"""Molecular and histology alignment networks with stage-specific losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """Project features through a linear, GELU, normalization and dropout block."""

    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MMDLoss(nn.Module):
    """Measure distribution discrepancy with a mixture of Gaussian kernels."""

    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def gaussian_kernel(self, source, target):
        combined_embeddings = torch.cat([source, target], dim=0)
        row_embeddings = combined_embeddings.unsqueeze(0)
        column_embeddings = combined_embeddings.unsqueeze(1)
        squared_distances = ((row_embeddings - column_embeddings) ** 2).sum(2)

        n_samples = combined_embeddings.size(0)
        if self.fix_sigma is not None:
            bandwidth = self.fix_sigma
        else:
            pair_count = max(n_samples * n_samples - n_samples, 1)
            bandwidth = torch.sum(squared_distances.detach()) / pair_count

        bandwidth = bandwidth / (self.kernel_mul ** (self.kernel_num // 2))
        bandwidth_list = [
            bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)
        ]
        kernel_values = [
            torch.exp(-squared_distances / (kernel_bandwidth + 1e-12))
            for kernel_bandwidth in bandwidth_list
        ]
        return sum(kernel_values)

    def forward(self, source, target):
        n_source = source.size(0)

        kernels = self.gaussian_kernel(source, target)

        source_kernel = kernels[:n_source, :n_source]
        target_kernel = kernels[n_source:, n_source:]
        cross_kernel = kernels[:n_source, n_source:]
        reverse_cross_kernel = kernels[n_source:, :n_source]

        return (
            source_kernel.mean()
            + target_kernel.mean()
            - cross_kernel.mean()
            - reverse_cross_kernel.mean()
        )


class PatientAlignmentModel(nn.Module):
    """Align patient molecular and H&E features to the cell embedding space."""

    def __init__(
        self,
        img_dim: int,
        gene_dim: int,
        cell_dim: int,
        drug_dim: int,
        output_dim: int,
        clin_dim: int = 0,
        num_cancers: int = 1,
        hidden_dim: int = 512,
        cancer_emb_dim: int = 64,
        dropout: float = 0.1,
        use_clinical: bool = True,
        unicure_model=None,
    ):
        super().__init__()

        self.use_clinical = use_clinical and clin_dim > 0
        self.output_dim = output_dim
        self.cell_dim = cell_dim
        self.hidden_dim = hidden_dim
        self.unicure_model = unicure_model

        # Stage1: TCGA/gene -> cell space

        self.tcga_align_head = nn.Sequential(
            MLPBlock(gene_dim, hidden_dim, dropout),
            nn.Linear(hidden_dim, cell_dim),
        )

        # Stage2: image -> cell space

        self.img_align_head = nn.Sequential(
            MLPBlock(img_dim, hidden_dim, dropout),
            nn.Linear(hidden_dim, cell_dim),
        )

        # optional clinical encoder
        if self.use_clinical:
            self.clin_encoder = nn.Sequential(
                MLPBlock(clin_dim, hidden_dim // 2, dropout),
                MLPBlock(hidden_dim // 2, hidden_dim // 2, dropout),
            )
        else:
            self.clin_encoder = None

        self.cancer_embedding = nn.Embedding(num_cancers, cancer_emb_dim)

        self.mmd_loss = MMDLoss()

    # basic encoders

    def _encode_clin(self, clin):
        if self.use_clinical:
            return self.clin_encoder(clin)
        return None

    def _encode_cancer(self, cancer_idx):
        if cancer_idx is None:
            raise ValueError("cancer_idx is required")
        return self.cancer_embedding(cancer_idx)

    def align_tcga_to_cell(self, tcga_emb):
        return self.tcga_align_head(tcga_emb)

    def align_img_to_cell(self, img_emb):
        return self.img_align_head(img_emb)

    # freeze / unfreeze control

    def freeze_all(self):
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze_stage1(self):
        for parameter in self.tcga_align_head.parameters():
            parameter.requires_grad = True

    def unfreeze_stage2(self):
        for parameter in self.img_align_head.parameters():
            parameter.requires_grad = True

    def unfreeze_stage3(self):
        if self.unicure_model is not None:
            for parameter in self.unicure_model.parameters():
                parameter.requires_grad = True

    def freeze_stage1(self):
        for parameter in self.tcga_align_head.parameters():
            parameter.requires_grad = False

    def freeze_stage2(self):
        for parameter in self.img_align_head.parameters():
            parameter.requires_grad = False

    def freeze_stage3(self):
        if self.unicure_model is not None:
            for parameter in self.unicure_model.parameters():
                parameter.requires_grad = False

    # stage3 forward helpers

    def predict_delta_with_unicure(self, img, drug_emb):
        if self.unicure_model is None:
            raise ValueError("unicure_model is None")

        aligned_image = self.align_img_to_cell(img)
        predicted_delta = self.unicure_model("pertrub_forward", aligned_image, drug_emb)
        return predicted_delta, aligned_image

    def predict_delta_with_unicure_and_pool(self, img, drug_emb):
        if self.unicure_model is None:
            raise ValueError("unicure_model is None")

        aligned_image = self.align_img_to_cell(img)
        predicted_delta, fusion_pool_128 = self.unicure_model(
            "pertrub_forward_with_pool_embedding",
            aligned_image,
            drug_emb,
        )
        return predicted_delta, aligned_image, fusion_pool_128

    def predict_delta(self, img, drug_emb):
        predicted_delta, _ = self.predict_delta_with_unicure(
            img=img,
            drug_emb=drug_emb,
        )
        return predicted_delta

    def predict_delta_with_unicure_and_teacher(self, img, gene, drug_emb):
        if self.unicure_model is None:
            raise ValueError("unicure_model is None")

        aligned_image = self.align_img_to_cell(img)
        with torch.no_grad():
            aligned_gene = self.align_tcga_to_cell(gene)

        predicted_delta = self.unicure_model("pertrub_forward", aligned_image, drug_emb)
        return predicted_delta, aligned_image, aligned_gene

    def predict_delta_with_unicure_teacher_and_pool(self, img, gene, drug_emb):
        if self.unicure_model is None:
            raise ValueError("unicure_model is None")

        aligned_image = self.align_img_to_cell(img)
        with torch.no_grad():
            aligned_gene = self.align_tcga_to_cell(gene)

        predicted_delta, fusion_pool_128 = self.unicure_model(
            "pertrub_forward_with_pool_embedding",
            aligned_image,
            drug_emb,
        )
        return predicted_delta, aligned_image, aligned_gene, fusion_pool_128

    # optional unified forward

    def forward(self, mode, **kwargs):
        if mode == "stage1_align":
            return self.align_tcga_to_cell(kwargs["tcga_emb"])

        elif mode == "stage2_align":
            return self.align_img_to_cell(kwargs["img"])

        elif mode == "stage3_delta":
            return self.predict_delta_with_unicure(
                img=kwargs["img"],
                drug_emb=kwargs["drug_emb"],
            )

        elif mode == "stage3_delta_with_teacher":
            return self.predict_delta_with_unicure_and_teacher(
                img=kwargs["img"],
                gene=kwargs["gene"],
                drug_emb=kwargs["drug_emb"],
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

    # utilities

    def sample_embeddings(self, emb, max_samples=512):
        if emb.size(0) <= max_samples:
            return emb
        idx = torch.randperm(emb.size(0), device=emb.device)[:max_samples]
        return emb[idx]

    # stage1 losses

    def stage1_align_losses(
        self,
        tcga_emb,
        tcga_cancer_idx,
        cell_centroids,
        cell_emb_all,
        cell_cancer_idx_all,
        unmatched_cancer_idx,
        matchable_cancer_indices,
        lambda_center=0.0,
        lambda_mmd_cancer=1.0,
        lambda_mmd_global=0.0,
        cell_emb_global=None,
        same_cancer_sample_n=512,
        global_sample_n=1000,
    ):
        """Compute cancer-matched MMD, global-pool MMD and center losses."""

        tcga_aligned = self.align_tcga_to_cell(tcga_emb)

        device = tcga_emb.device
        tcga_cancer_idx = tcga_cancer_idx.to(device)
        cell_emb_all = cell_emb_all.to(device)
        cell_cancer_idx_all = cell_cancer_idx_all.to(device)

        if cell_centroids is not None:
            cell_centroids = cell_centroids.to(device)

        if cell_emb_global is None:
            cell_emb_global = cell_emb_all
        else:
            cell_emb_global = cell_emb_global.to(device)

        loss_mmd_cancer = torch.tensor(0.0, device=device)
        loss_mmd_global = torch.tensor(0.0, device=device)
        loss_center = torch.tensor(0.0, device=device)

        unique_cancers = torch.unique(tcga_cancer_idx)

        if matchable_cancer_indices is None:
            matchable_cancer_indices = []
        matchable_cancer_indices = set(int(x) for x in matchable_cancer_indices)

        unmatched_idx = None
        if unmatched_cancer_idx is not None:
            unmatched_idx = int(unmatched_cancer_idx)

        valid_cancer_mmd_count = 0
        valid_global_mmd_count = 0
        valid_center_count = 0

        # 1. matched same-cancer MMD

        if lambda_mmd_cancer > 0:
            for c in unique_cancers:
                c_int = int(c.item())

                # Evaluate UNMATCHED patients with the global-pool objective below.
                if unmatched_idx is not None and c_int == unmatched_idx:
                    continue

                if c_int not in matchable_cancer_indices:
                    continue

                tcga_mask = tcga_cancer_idx == c
                if tcga_mask.sum().item() < 2:
                    continue

                cell_mask = cell_cancer_idx_all == c
                if cell_mask.sum().item() < 2:
                    continue

                tcga_c = tcga_aligned[tcga_mask]

                # Sample reference features from all same-cancer control replicates.
                cell_c = cell_emb_all[cell_mask]
                cell_c = self.sample_embeddings(
                    cell_c,
                    max_samples=same_cancer_sample_n,
                )

                if tcga_c.size(0) < 2 or cell_c.size(0) < 2:
                    continue

                loss_mmd_cancer = loss_mmd_cancer + self.mmd_loss(tcga_c, cell_c)
                valid_cancer_mmd_count += 1

            if valid_cancer_mmd_count > 0:
                loss_mmd_cancer = loss_mmd_cancer / valid_cancer_mmd_count

        # 2. unmatched global fallback MMD

        if lambda_mmd_global > 0 and unmatched_idx is not None:
            unmatched_mask = tcga_cancer_idx == unmatched_idx

            if unmatched_mask.sum().item() >= 2 and cell_emb_global.size(0) >= 2:
                tcga_unmatched = tcga_aligned[unmatched_mask]
                cell_global = self.sample_embeddings(
                    cell_emb_global,
                    max_samples=global_sample_n,
                )

                if tcga_unmatched.size(0) >= 2 and cell_global.size(0) >= 2:
                    loss_mmd_global = self.mmd_loss(tcga_unmatched, cell_global)
                    valid_global_mmd_count += 1

        # 3. center loss

        if lambda_center > 0 and cell_centroids is not None:
            for c in unique_cancers:
                c_int = int(c.item())

                if unmatched_idx is not None and c_int == unmatched_idx:
                    continue

                if c_int not in matchable_cancer_indices:
                    continue

                if c_int < 0 or c_int >= cell_centroids.size(0):
                    continue

                tcga_mask = tcga_cancer_idx == c
                if tcga_mask.sum().item() == 0:
                    continue

                tcga_c = tcga_aligned[tcga_mask]
                centroid_c = cell_centroids[c_int].unsqueeze(0).expand_as(tcga_c)

                loss_center = loss_center + F.mse_loss(tcga_c, centroid_c)
                valid_center_count += 1

            if valid_center_count > 0:
                loss_center = loss_center / valid_center_count

        # 4. total

        total = (
            lambda_mmd_cancer * loss_mmd_cancer
            + lambda_mmd_global * loss_mmd_global
            + lambda_center * loss_center
        )

        return {
            "loss": total,
            "loss_mmd_cancer": loss_mmd_cancer,
            "loss_center": loss_center,
            "loss_mmd_global": loss_mmd_global,
            "tcga_aligned": tcga_aligned,
            "valid_cancer_mmd_count": valid_cancer_mmd_count,
            "valid_global_mmd_count": valid_global_mmd_count,
            "valid_center_count": valid_center_count,
        }

    # stage2 losses

    def stage2_losses(
        self,
        img,
        gene,
        lambda_img_align=1.0,
        lambda_img_cos=0.5,
    ):
        with torch.no_grad():
            aligned_gene = self.align_tcga_to_cell(gene)

        aligned_image = self.align_img_to_cell(img)

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

        total = lambda_img_align * loss_img_align + lambda_img_cos * loss_img_cos

        return {
            "loss": total,
            "loss_img_align": loss_img_align,
            "loss_img_cos": loss_img_cos,
            "img_aligned": aligned_image,
            "gene_aligned": aligned_gene,
        }

    # stage3 losses

    def stage3_losses(
        self,
        img,
        drug_emb,
        delta_target,
        lambda_mse=1.0,
        lambda_cos=0.5,
    ):
        """Compute perturbation losses with fixed molecular and image alignment."""
        predicted_delta, aligned_image = self.predict_delta_with_unicure(
            img=img,
            drug_emb=drug_emb,
        )

        loss_mse = F.mse_loss(predicted_delta, delta_target)
        loss_cos = (
            1.0
            - F.cosine_similarity(predicted_delta, delta_target, dim=1, eps=1e-8).mean()
        )

        total = lambda_mse * loss_mse + lambda_cos * loss_cos

        return {
            "loss": total,
            "loss_mse": loss_mse,
            "loss_cos": loss_cos,
            "pred_delta": predicted_delta,
            "img_aligned": aligned_image,
        }

    # stage2 + stage3 joint losses

    def stage23_joint_losses(
        self,
        img,
        gene,
        drug_emb,
        delta_target,
        lambda_mse=1.0,
        lambda_cos=0.5,
        lambda_img_align=0.1,
        lambda_img_cos=0.05,
    ):
        """Combine perturbation prediction and image-to-molecular alignment losses."""
        predicted_delta, aligned_image, aligned_gene = (
            self.predict_delta_with_unicure_and_teacher(
                img=img,
                gene=gene,
                drug_emb=drug_emb,
            )
        )

        loss_mse = F.mse_loss(predicted_delta, delta_target)
        loss_cos = (
            1.0
            - F.cosine_similarity(predicted_delta, delta_target, dim=1, eps=1e-8).mean()
        )

        loss_img_align = F.mse_loss(aligned_image, aligned_gene)
        loss_img_cos = (
            1.0
            - F.cosine_similarity(aligned_image, aligned_gene, dim=1, eps=1e-8).mean()
        )

        total = (
            lambda_mse * loss_mse
            + lambda_cos * loss_cos
            + lambda_img_align * loss_img_align
            + lambda_img_cos * loss_img_cos
        )

        return {
            "loss": total,
            "loss_mse": loss_mse,
            "loss_cos": loss_cos,
            "loss_img_align": loss_img_align,
            "loss_img_cos": loss_img_cos,
            "pred_delta": predicted_delta,
            "img_aligned": aligned_image,
            "gene_aligned": aligned_gene,
        }

    def stage23_joint_losses_with_pool(
        self,
        img,
        gene,
        drug_emb,
        delta_target,
        lambda_mse=1.0,
        lambda_cos=0.5,
        lambda_img_align=0.1,
        lambda_img_cos=0.05,
    ):
        predicted_delta, aligned_image, aligned_gene, fusion_pool_128 = (
            self.predict_delta_with_unicure_teacher_and_pool(
                img=img,
                gene=gene,
                drug_emb=drug_emb,
            )
        )

        loss_mse = F.mse_loss(predicted_delta, delta_target)
        loss_cos = (
            1.0
            - F.cosine_similarity(predicted_delta, delta_target, dim=1, eps=1e-8).mean()
        )

        loss_img_align = F.mse_loss(aligned_image, aligned_gene)
        loss_img_cos = (
            1.0
            - F.cosine_similarity(aligned_image, aligned_gene, dim=1, eps=1e-8).mean()
        )

        total = (
            lambda_mse * loss_mse
            + lambda_cos * loss_cos
            + lambda_img_align * loss_img_align
            + lambda_img_cos * loss_img_cos
        )

        return {
            "loss": total,
            "loss_mse": loss_mse,
            "loss_cos": loss_cos,
            "loss_img_align": loss_img_align,
            "loss_img_cos": loss_img_cos,
            "pred_delta": predicted_delta,
            "img_aligned": aligned_image,
            "gene_aligned": aligned_gene,
            "fusion_pool_128": fusion_pool_128,
        }


# Public aliases preserve compatibility with existing integrations.
PatientDrugResponseModel_V3 = PatientAlignmentModel
