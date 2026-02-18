import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.vqvae_model import VQVAEModel, VanillaStructureTokenEncoder, node_gather, VanillaStructureTokenDecoder, VanillaCategoricalMixture
from protein_chain import WrappedProteinChain
from protein_complex import WrappedProteinComplex

# from esm.utils.structure.affine3d import (
#     Affine3D,
#     build_affine3d_from_coordinates,
# )
from esm.utils.structure.protein_structure import infer_cbeta_from_atom37
from esm.utils.structure.predicted_aligned_error import (
    compute_predicted_aligned_error,
    compute_tm,
)
from copy import deepcopy
from einops import rearrange

from src.vqvae.encoders import ComplexStructureTokenEncoder
from src.vqvae.aux_module import ContactPredictionHead

def compute_lddt_ca(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
    chain_id = None
) -> torch.Tensor:

    n = all_atom_pred_pos.shape[0]
    diff = all_atom_positions[:, None, :] - all_atom_positions[None, :, :]  # [N, N, 3]
    dmat_true = torch.sqrt(torch.sum(diff ** 2, dim=-1) + eps)  # [N, N]

    diff = all_atom_pred_pos[:, None, :] - all_atom_pred_pos[None, :, :]  # [N, N, 3]
    dmat_pred = torch.sqrt(torch.sum(diff ** 2, dim=-1) + eps)  # [N, N]

    dists_to_score = ((dmat_true < cutoff) * (1.0 - torch.eye(n, device=dmat_true.device)))

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    lddt = norm * (eps + torch.sum(dists_to_score * score, dim=dims))
    zero_index = torch.sum(dists_to_score, dim=dims) == 0
    lddt = lddt[~zero_index]

    if chain_id is not None:
        # import pdb; pdb.set_trace()
        intra_chain_mask = chain_id.unsqueeze(-1) == chain_id.unsqueeze(-2)
        inter_chain_mask = chain_id.unsqueeze(-1) != chain_id.unsqueeze(-2)
        
        intra_dists_to_score = dists_to_score * intra_chain_mask
        inter_dists_to_score = dists_to_score * inter_chain_mask

        norm = 1.0 / (eps + torch.sum(intra_dists_to_score, dim=dims))
        intra_lddt = norm * (eps + torch.sum(intra_dists_to_score * score, dim=dims))
        zero_index = torch.sum(intra_dists_to_score, dim=dims) == 0
        intra_lddt = intra_lddt[~zero_index]

        norm = 1.0 / (eps + torch.sum(inter_dists_to_score, dim=dims))
        inter_lddt = norm * (eps + torch.sum(inter_dists_to_score * score, dim=dims))
        zero_index = torch.sum(inter_dists_to_score, dim=dims) == 0
        inter_lddt = inter_lddt[~zero_index]

        inter_chain_mask = chain_id.unsqueeze(-1) != chain_id.unsqueeze(-2)
        antigen_mask = (chain_id >= 2).unsqueeze(-1) + (chain_id >= 2).unsqueeze(-2)
        boundary_mask = inter_chain_mask * antigen_mask
        boundary_dists_to_score = dists_to_score * boundary_mask

        norm = 1.0 / (eps + torch.sum(boundary_dists_to_score, dim=dims))
        boundary_lddt = norm * (eps + torch.sum(boundary_dists_to_score * score, dim=dims))
        zero_index = torch.sum(boundary_dists_to_score, dim=dims) == 0
        boundary_lddt = boundary_lddt[~zero_index]

        return lddt, intra_lddt, inter_lddt, boundary_lddt
    else:
        return lddt

class ComplexStructureTokenDecoder(VanillaStructureTokenDecoder):
    """
    Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/models/vqvae.py#L335
    """
    def __init__(
        self,
        encoder_d_out,
        d_model,
        n_heads,
        n_layers,
        cdr
    ):
        super().__init__(
        encoder_d_out,
        d_model,
        n_heads,
        n_layers)

        self.cdr = cdr
        if self.cdr:
            self.cdr_emb = nn.Embedding(8, d_model)

    def decode(
        self,
        quantized_z: torch.Tensor,
        structure_tokens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        chain_id: torch.Tensor | None = None,
        cdr_index: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ):
        if sequence_id is None:
            sequence_id = torch.zeros_like(structure_tokens, dtype=torch.int64)

        if chain_id is None:
            chain_id = torch.zeros_like(structure_tokens, dtype=torch.int64)

        assert (
            (structure_tokens < 0).sum() == 0
        ), "All structure tokens set to -1 should be replaced with BOS, EOS, PAD, or MASK tokens by now, but that isn't the case!"

        x = self.post_vq_proj(quantized_z) # [B, L, hidden_dim=128] -> [B, L, d_model=1024]
        # !!! NOTE: Attention mask is actually unused here so watch out

        if self.cdr:
            x = x + self.cdr_emb(cdr_index)

        attn_weights, x, _ = self.decoder_stack.forward(
            x, attention_mask=attention_mask, affine=None, affine_mask=None, sequence_id=sequence_id, chain_id=chain_id, attn_bias=attn_bias) # [B, L, d_model], [B, L, d_model]


        tensor7_affine, bb_pred = self.affine_output_projection(
            x, affine=None, affine_mask=torch.zeros_like(attention_mask)
        ) # [B, L, 12], [B, L, 3, 3]

        pae, ptm = None, None
        pairwise_logits = self.pairwise_classification_head(x) # [B, L, L, 64 + 96 + 64]
        pairwise_dist_logits, pairwise_dir_logits, pae_logits = [
            (o if o.numel() > 0 else None)
            for o in pairwise_logits.split(self.pairwise_bins, dim=-1)
        ] # [B, L, L, 64], [B, L, L, 96], [B, L, L, 64]

        special_tokens_mask = structure_tokens >= min(self.special_tokens.values())
        pae = compute_predicted_aligned_error(
            pae_logits,  # type: ignore
            aa_mask=~special_tokens_mask,
            sequence_id=sequence_id,
            max_bin=self.max_pae_bin,
        ) # [B, L, L]

        # This might be broken for chainbreak tokens? We might align to the chainbreak
        ptm = compute_tm(
            pae_logits,  # type: ignore
            aa_mask=~special_tokens_mask,
            max_bin=self.max_pae_bin,
        ) # [B,]

        plddt_logits = self.plddt_head(x) # [B, L, 50]
        plddt_value = VanillaCategoricalMixture(
            plddt_logits, bins=plddt_logits.shape[-1]
        ).mean() # [B, L]

        return dict(
            tensor7_affine=tensor7_affine,
            bb_pred=bb_pred,
            plddt=plddt_value,
            ptm=ptm,
            predicted_aligned_error=pae,
            pairwise_dist_logits=pairwise_dist_logits,
            pairwise_dir_logits=pairwise_dir_logits,
            last_hidden_state=x
        )

class VQVAEModelComplex(VQVAEModel):
    """
    A VQ-VAE model for complex protein structures, extending the base VQVAEModel.
    This class is designed to handle the complexities of protein chains and their interactions.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = ComplexStructureTokenEncoder(
            **self.model_cfg.encoder,
            n_codes=self.quantizer_cfg.codebook_size
        ) 

        self.model_cfg.decoder["encoder_d_out"] = self.model_cfg.encoder.d_out
        self.decoder = ComplexStructureTokenDecoder(**self.model_cfg.decoder)

        self.encoder_attention = self.model_cfg.encoder_attention # intra, all
        self.decoder_attention = self.model_cfg.decoder_attention # intra, all

        self.decoding = 'complex' # heavy, complex
        self.kabsch = 'complex' # heavy, complex
        self.attn_bias = self.model_cfg.get("attn_bias", True)

        self.boundary_weight = self.model_cfg.get("boundary_weight", 0.0)
        self.boundary_loss_weight = self.model_cfg.get("boundary_loss_weight", 1.0)
        self.contrastive_loss_weight = self.model_cfg.get("contrastive_loss_weight", 1.0)
        self.recycle_loss_weight = self.model_cfg.get("recycle_loss_weight", 1.0)

    def _valid_or_test_step(self, batch, batch_idx, split="validation"):
        outputs = self.model.forward_single(batch["input_list"])

        loss, metrics = outputs[0]

        log_metrics = {
            f"{split}_{k}": v for k, v in metrics.items()
        }

        self.log_dict(
            {f"{split}_loss": loss, **log_metrics},
            prog_bar=True,
            batch_size=self.optimizer_cfg.micro_batch_size,
            logger=True,
            add_dataloader_idx=False,
        )

        return {
            f"{split}_loss": loss,
            **log_metrics,
        }


    def forward(self, input_list, use_as_tokenizer=False):
        self._step_count += 1

        coords, attention_mask, residue_index, chain_index, seq_residue_tokens, pdb_complex = input_list
        sequence_id = None

        if attention_mask is None:
            attention_mask = torch.ones_like(seq_residue_tokens, dtype=torch.bool)
        else:
            attention_mask = ~attention_mask # NOTE: due to data loading processing
        attention_mask = attention_mask.bool()

        hotspot_mask, attn_bias = self.compute_hot_spot_mask(coords, attention_mask, chain_index)
        cdr_index, attn_bias = self.compute_cdr_index(residue_index, chain_index, hotspot_mask, attention_mask)
        if self.attn_bias == False:
            attn_bias = None

        if self.encoder_attention == 'all':
            z = self.encoder.encode(coords, attention_mask, sequence_id, residue_index, None)
        else:
            z = self.encoder.encode(coords, attention_mask, sequence_id, residue_index, chain_index)

        assert self.quantizer.codebook_embed_size == self.encoder.d_out
        quantized_z, quantized_indices, partial_loss, partial_metrics = self.quantizer(z)
        assert not z.isnan().any() and not quantized_indices.isnan().any()
        if use_as_tokenizer:
            return quantized_z, quantized_indices, z

        if self.decoder_attention == 'all':
            decoded_states = self.decoder.decode(quantized_z, quantized_indices, attention_mask, chain_id=torch.ones_like(chain_index, device=chain_index.device), cdr_index=cdr_index, attn_bias=attn_bias)
        else:
            decoded_states = self.decoder.decode(quantized_z, quantized_indices, attention_mask, chain_id=chain_index)

        # reconstructed proteins
        chainbreak_mask = residue_index != -1
        bb_pred = decoded_states["bb_pred"]
        bb_rmsd_list, lddt_list = [], []
        for i in range(len(bb_pred)):
            # import pdb; pdb.set_trace()
            mask = torch.logical_and(chainbreak_mask[i], attention_mask[i]) # [L]
            pdb_chain_recon = WrappedProteinChain.from_backbone_atom_coordinates(bb_pred[i][mask].detach())

            pdb_chainbreak_mask = (pdb_complex[i].residue_index != -1)
            pdb_chain_ori = WrappedProteinChain.from_backbone_atom_coordinates(torch.tensor(pdb_complex[i].atom37_positions[pdb_chainbreak_mask, :3, :]))

            bb_rmsd = pdb_chain_recon.rmsd(pdb_chain_ori, only_compute_backbone_rmsd=True)
            bb_rmsd_list.append(bb_rmsd)

            new_complex = copy.deepcopy(pdb_complex[i]) 
            new_atom37_positions = np.full([len(new_complex), 37, 3], np.nan)
            new_atom37_positions[:, :3, :] = bb_pred[i, :len(pdb_complex[i])].detach().to('cpu').numpy()

            new_atom37_mask = np.zeros([len(new_complex), 37], dtype=bool)
            new_atom37_mask[:,:3] = True

            new_complex.atom37_positions = new_atom37_positions
            new_complex.atom37_mask = new_atom37_mask

            lddt = new_complex.lddt_ca(pdb_complex[i], compute_chain_assignment=False)
            lddt_list.append(lddt.mean())

        # pair mask
        # 1) padding 제외 mask (기존)
        pair_pad_mask = attention_mask.unsqueeze(-1) & attention_mask.unsqueeze(-2)  # [B, L, L] bool

        # 2) antigen / antibody 구분 (예시: chain_index < 2가 antibody, >=2가 antigen 이라고 가정)
        is_antigen  = (chain_index >= 2)  # [B, L] bool
        is_antibody = ~is_antigen         # [B, L] bool  (혹은 (chain_index < 2))

        # 항원-항체 "cross" pair만 True
        boundary_mask = (is_antigen.unsqueeze(-1) & is_antibody.unsqueeze(-2)) | (is_antibody.unsqueeze(-1) & is_antigen.unsqueeze(-2))   # [B, L, L] bool

        # 3) (선택) 거리 기반으로 'binding pair'만 더 좁히기
        ca = coords[:, :, 1, :]               # [B, L, 3]
        dist = torch.cdist(ca, ca)            # [B, L, L]
        contact_mask = dist < 8.0             # [B, L, L] bool  (8Å 이내만)

        # 최종 pair mask: padding 제외 + 항원-항체 + (선택) contact
        # final_pair_mask = pair_pad_mask & contact_mask   # contact 빼고 싶으면 &contact_mask 제거
        final_pair_mask = pair_pad_mask & boundary_mask & contact_mask   # contact 빼고 싶으면 &contact_mask 제거

        # reconstruction loss: 
        coords_recon = decoded_states["bb_pred"]
        chainbreak_mask = residue_index != -1
        extended_chainbreak_mask = chainbreak_mask.clone()
        extended_chainbreak_mask[:, 1:] &= chainbreak_mask[:, :-1]  # 뒤의 인덱스가 False일 때 앞 인덱스도 False
        extended_chainbreak_mask[:, :-1] &= chainbreak_mask[:, 1:]  # 앞의 인덱스가 False일 때 뒤 인덱스도 False
        extended_chainbreak_mask &= chainbreak_mask # False인 부분도 유지 (자기 자신)

        mask = torch.logical_and(attention_mask, chainbreak_mask) # [B, L]
        direction_mask = torch.logical_and(attention_mask, extended_chainbreak_mask) # [B, L]

        # (1) backbone geometric distance loss: pairwise L2 distance matrix for 
        geom_dist_loss, geom_dist_metrics = self.compute_geometric_distance(coords_recon, coords[:, :, :3, :], mask, final_pair_mask, self.boundary_weight) # [B, L, 3, 3]   

        # (2) backbone geometric direction loss
        geom_dir_loss, geom_dir_metrics = self.compute_geometric_direction(coords_recon, coords[:, :, :3, :], direction_mask) #, final_pair_mask, self.boundary_weight)
            # coords_recon, coords[:, :, :3, :], attention_mask, chainbreak_mask)

        # (3) backbone binned distance classification
        binned_dist_loss, binned_dist_metrics = self.compute_binned_distance(decoded_states["pairwise_dist_logits"], coords, mask, final_pair_mask, self.boundary_weight)
            # decoded_states["pairwise_dist_logits"], coords, attention_mask)

        # (4) backbone binned direction classification
        binned_dir_loss, binned_dir_metrics = self.compute_binned_direction(
            decoded_states["pairwise_dir_logits"], coords[:, :, :3, :], direction_mask)

        # (5) inverse folding 
        inverse_folding_loss, inverse_folding_metrics, inverse_folding_logits, inverse_folding_ppl, true_masks = self.compute_inverse_folding(decoded_states["last_hidden_state"], seq_residue_tokens, mask, reduction='none')

        # reconstruction_loss = (geom_dist_loss + geom_dir_loss + binned_dist_loss + binned_dir_loss + inverse_folding_loss.mean()).mean()

        # # hinge distance loss
        hotspot_loss, hotspot_mask = self.compute_interface_hotspot_min_dist_loss(
            gt_coords=coords,                 # GT atom37 등
            pred_coords=coords_recon,       # [B,L,3,3]
            attention_mask=attention_mask,
            chain_index=chain_index,
            contact_cut=8.0,
            d0=8.0,
        )


        reconstruction_loss = (geom_dist_loss + geom_dir_loss + binned_dist_loss + binned_dir_loss + inverse_folding_loss.mean() + self.boundary_loss_weight * hotspot_loss).mean()


        recycle_z = self.encoder.encode(coords_recon, attention_mask, sequence_id, residue_index, None)
        recycle_quantized_z, recycle_quantized_indices, recycled_partial_loss, partial_metrics = self.quantizer(recycle_z)
        recycle_decoded_states = self.decoder.decode(recycle_quantized_z, recycle_quantized_indices, attention_mask, chain_id=torch.ones_like(chain_index, device=chain_index.device), cdr_index=cdr_index, attn_bias=attn_bias)


        z1 = F.normalize(z, dim=-1) # [B, L, d_model]
        z2 = F.normalize(recycle_z, dim=-1) # [B, L, d_model]
        sim_z1 = torch.matmul(z1, z1.transpose(1,2)) # [B, L, L]
        sim_z2 = torch.matmul(z2, z2.transpose(1,2)) # [B, L, L]
        contrastive_loss = F.mse_loss(sim_z1[final_pair_mask], sim_z2[final_pair_mask])

        loss = reconstruction_loss * self.loss_weight["reconstruction_loss_weight"] + partial_loss + self.contrastive_loss_weight * contrastive_loss

        recycled_inverse_folding_loss, recycled_inverse_folding_metrics, recycled_inverse_folding_logits, recycled_inverse_folding_ppl, _ = self.compute_inverse_folding(recycle_decoded_states["last_hidden_state"], seq_residue_tokens, mask, reduction='none')

        orig_logits = inverse_folding_logits.detach()
        recy_logits = recycled_inverse_folding_logits

        kl_loss = F.kl_div(
            F.log_softmax(recy_logits, dim=-1),
            F.softmax(orig_logits, dim=-1),
            reduction='none'
        ).sum(-1)
        # import pdb; pdb.set_trace()

        confidence_mask = inverse_folding_ppl < 2
        inverse_folding_consistency_loss = kl_loss[confidence_mask].mean()
        loss += self.recycle_loss_weight * inverse_folding_consistency_loss

        metrics = {
            **geom_dist_metrics,
            **geom_dir_metrics,
            **binned_dist_metrics,
            **binned_dir_metrics,
            **inverse_folding_metrics,
            **partial_metrics,
            "reconstruction_loss": reconstruction_loss,
            "bb_rmsd": torch.tensor(bb_rmsd_list, device=coords.device).mean(),
            "lddt": torch.tensor(lddt_list, device=coords.device).mean(),
        }

        loss_and_metrics = (loss, metrics)
        
        return (loss_and_metrics, )
    


    def forward_single(self, input_list, use_as_tokenizer=False):
        self._step_count += 1

        coords, attention_mask, residue_index, chain_index, seq_residue_tokens, pdb_complex = input_list
        sequence_id = None

        if attention_mask is None:
            attention_mask = torch.ones_like(seq_residue_tokens, dtype=torch.bool)
        else:
            attention_mask = ~attention_mask # NOTE: due to data loading processing
        attention_mask = attention_mask.bool()

        hotspot_mask, attn_bias = self.compute_hot_spot_mask(coords, attention_mask, chain_index)
        cdr_index, attn_bias = self.compute_cdr_index(residue_index, chain_index, hotspot_mask, attention_mask)
        if self.attn_bias == False:
            attn_bias = None

        if self.encoder_attention == 'all':
            z = self.encoder.encode(coords, attention_mask, sequence_id, residue_index, None)
        else:
            z = self.encoder.encode(coords, attention_mask, sequence_id, residue_index, chain_index)

        assert self.quantizer.codebook_embed_size == self.encoder.d_out
        quantized_z, quantized_indices, partial_loss, partial_metrics = self.quantizer(z)
        assert not z.isnan().any() and not quantized_indices.isnan().any()
        if use_as_tokenizer:
            return quantized_z, quantized_indices, z

        _, heavybreak_index = torch.where(residue_index == -1)
        heavybreak_index = heavybreak_index[0].item()        
        if self.decoding == 'complex':
            if self.decoder_attention == 'all':
                decoded_states = self.decoder.decode(quantized_z, quantized_indices, attention_mask, chain_id=torch.ones_like(chain_index, device=chain_index.device), cdr_index=cdr_index, attn_bias=attn_bias)
            else:
                decoded_states = self.decoder.decode(quantized_z, quantized_indices, attention_mask, chain_id=chain_index)


        if self.kabsch == 'complex':
            chainbreak_mask = residue_index != -1
        bb_pred = decoded_states["bb_pred"]
        bb_rmsd_list, lddt_list, intra_lddt_list, inter_lddt_list, boundary_lddt_list = [], [], [], [], []
        for i in range(len(bb_pred)):
            mask = torch.logical_and(chainbreak_mask[i], attention_mask[i]) # [L]
            pdb_chain_recon = WrappedProteinChain.from_backbone_atom_coordinates(bb_pred[i][mask].detach())
            pdb_chainbreak_mask = (pdb_complex[i].residue_index != -1)
            pdb_chain_ori = WrappedProteinChain.from_backbone_atom_coordinates(torch.tensor(pdb_complex[i].atom37_positions[pdb_chainbreak_mask, :3, :]))

            bb_rmsd = pdb_chain_recon.rmsd(pdb_chain_ori, only_compute_backbone_rmsd=True)
            bb_rmsd_list.append(bb_rmsd)

            pred_ca_position = bb_pred[i,mask,1,:].detach().to('cpu')
            ori_ca_position = torch.tensor(pdb_complex[i].atom37_positions[pdb_chainbreak_mask, 1, :])
            lddt, intra_lddt, inter_lddt, boundary_lddt = compute_lddt_ca(pred_ca_position, ori_ca_position, chain_id=chain_index[i, pdb_chainbreak_mask].to('cpu'))

            reconstructed_complex = deepcopy(pdb_complex[0])
            reconstructed_positions = np.full_like(reconstructed_complex.atom37_positions, np.nan)
            reconstructed_positions[mask.to('cpu'), :3, :] = bb_pred[i][mask].detach().to('cpu').numpy()
            reconstructed_complex.atom37_positions = reconstructed_positions                

            reconstructed_complex = reconstructed_complex.infer_cbeta()
            reconstructed_complex = reconstructed_complex.infer_oxygen()
            reconstructed_mask = ~np.isnan(reconstructed_complex.atom37_positions).any(axis=-1)
            reconstructed_complex.atom37_mask = reconstructed_mask            

            generated_path = os.path.join(self.save_path, f"{pdb_complex[i].id}.pdb")
            reconstructed_complex.to_pdb(generated_path)

            lddt_list.append(lddt.mean())
            intra_lddt_list.append(intra_lddt.mean())
            inter_lddt_list.append(inter_lddt.mean())
            boundary_lddt_list.append(boundary_lddt.mean())


        metrics = {
            "bb_rmsd": torch.tensor(bb_rmsd_list, device=coords.device).mean(),
            "Intra_lddt": torch.tensor(intra_lddt_list, device=coords.device).mean(),
            "Inter_lddt": torch.tensor(inter_lddt_list, device=coords.device).mean(),
            "Binding_lddt": torch.tensor(boundary_lddt_list, device=coords.device).mean(),
            "lddt": torch.tensor(lddt_list, device=coords.device).mean(),
        }
        loss = torch.tensor([0.0], device=coords.device)
        loss_and_metrics = (loss, metrics)

        return (loss_and_metrics, )


    def compute_geometric_distance(self, x_recon, x, attention_mask, pair_mask=None, boundary_weight=0.0, clamp_value=25):
        """
        x_recon: [B, L, 3, 3]
        x: [B, L, 3, 3]
        """
        assert x_recon.shape[-2] == 3 and x_recon.shape[-1] == 3
        
        # ignore padding regions
        x_recon[~attention_mask] = 0
        x[~attention_mask] = 0
        B, L, E = x.shape[0], x.shape[1], x.shape[-1]
        x_recon, x = x_recon.reshape(B, -1, E), x.reshape(B, -1, E) # [B, L, 3, 3] -> [B, L * 3, 3] 

        dist_pred = torch.cdist(x_recon, x_recon, p=2.0) # [B, L * 3, L * 3]
        dist_true = torch.cdist(x, x, p=2.0)

        dist_mask = attention_mask.repeat(1, 3)
        dist_mask = torch.logical_and(dist_mask.unsqueeze(-1), dist_mask.unsqueeze(1)) # [B, L * 3, L * 3]

        if pair_mask is not None and boundary_weight != 0.0:
            # residue pair 마스크를 atom pair로 확장: 각 residue가 3 atoms이므로 3x3 블록 반복
            pair_mask_atom = pair_mask.repeat_interleave(3, dim=1).repeat_interleave(3, dim=2)  # [B, L*3, L*3]

            # weight: 기본 1, pair_mask인 곳은 1 + boundary_weight
            loss_w = torch.ones_like(dist_true, dtype=dist_true.dtype)
            loss_w = loss_w + float(boundary_weight) * pair_mask_atom.to(dist_true.dtype)

        else:
            loss_w = None

        dist_pred_f, dist_true_f = dist_pred[dist_mask], dist_true[dist_mask]

        # elementwise mse
        loss = (dist_pred_f - dist_true_f).pow(2)
        loss = torch.clamp(loss, max=float(clamp_value))

        # weighted mean
        if loss_w is not None:
            w_f = loss_w[dist_mask]
            denom = w_f.sum().clamp_min(1.0)
            loss_mean = (loss * w_f).sum() / denom
        else:
            loss_mean = loss.mean() if loss.numel() > 0 else dist_pred_f.new_tensor(0.0)

        metric = {
            f"geom_dist_loss": loss_mean,
            f"geom_dist_loss_below_clamp": loss[loss != clamp_value].mean(),
            f"geom_dist_loss_clamp_ratio_{clamp_value}": (loss != clamp_value).float().mean(),
        }
        # metrics like spearman R is too time consuming to calculate

        return loss_mean, metric

    def compute_binned_distance(self, pairwise_logits, coords, attention_mask, pair_mask=None, boundary_weight=0.0):
        """
        pairwise_logits: [B, L, L, 64]
        coords: [B, L, 37, 3]
        attention_mask: [B, L]
        """

        # calculate Cbeta
        cbeta = infer_cbeta_from_atom37(coords) # [B, L, 3]

        # pairwise Cbeta distance
        NUM_BIN = 64
        dist_true = torch.cdist(cbeta, cbeta, p=2.0)
        bin_edges = [0] + [(2.3125 + 0.3075 * i) ** 2 for i in range(NUM_BIN)]
        bin_edges = torch.tensor(bin_edges, device=pairwise_logits.device)
        binned_labels = torch.bucketize(dist_true, bin_edges, right=True) - 1 # [B, L, L]
        binned_labels = torch.clamp(binned_labels, max=NUM_BIN - 1, min=0)
        assert binned_labels.min() >= 0 and binned_labels.max() < NUM_BIN

        mask = torch.logical_and(attention_mask.unsqueeze(-1), attention_mask.unsqueeze(1)) # [B, L, L]

        if pair_mask is not None and boundary_weight != 0.0:
            pair_mask_bool = pair_mask & mask
            w = torch.ones_like(dist_true, dtype=pairwise_logits.dtype)  # [B, L, L]
            w = w + float(boundary_weight) * pair_mask_bool.to(pairwise_logits.dtype)
        else:
            w = None


        pairwise_logits, binned_labels = pairwise_logits[mask], binned_labels[mask]
        
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(pairwise_logits, binned_labels)

        if w is not None:
            w_f = w[mask]                     # [N]
            denom = w_f.sum().clamp_min(1.0)
            loss_mean = (loss * w_f).sum() / denom
        else:
            loss_mean = loss.mean() if loss.numel() > 0 else loss.new_tensor(0.0)

        metric = {
            f"binned_dist_loss": loss.mean(),
            f"binned_dist_accuracy": (pairwise_logits.argmax(dim=-1) == binned_labels).float().mean(),
        }
        return loss_mean, metric

    def compute_hot_spot_mask(
        self,
        gt_coords: torch.Tensor,       # [B, L, *, 3]
        attention_mask: torch.Tensor,  # [B, L] bool
        chain_index: torch.Tensor,     # [B, L] int (0,1=Ab; >=2=Ag)
        contact_cut: float = 8.0,      # hotspot 판정용 (GT에서 interface residue 정의)
        ca_idx: int = 1,               # CA index
        beta_cross: float = 3.0,      # attention bias weight
    ):
        """
        Returns:
        hotspot_mask: [B, L] bool  (GT로 정의된 interface residue)
        """
        valid = attention_mask.bool()
        B, L = valid.shape
        pair_pad_mask = valid[:, :, None] & valid[:, None, :]

        is_antibody = (chain_index < 2)
        is_antigen  = ~is_antibody

        # ---------- (A) hotspot mask: GT로 계산 ----------
        gt_ca = gt_coords[:, :, ca_idx, :]                 # [B, L, 3]
        gt_dist = torch.cdist(gt_ca, gt_ca, p=2.0)         # [B, L, L]

        boundary_mask = (
            (is_antibody[:, :, None] & is_antigen[:, None, :]) |
            (is_antigen[:, :, None]  & is_antibody[:, None, :])
        )
        gt_contact = gt_dist < contact_cut
        gt_pair = pair_pad_mask & boundary_mask & gt_contact

        hotspot_mask = (gt_pair.any(dim=-1) | gt_pair.any(dim=-2)) & valid  # [B, L]

        # --- (B) attention bias 만들기 (추가 bias만) ---
        # dtype/device는 나중에 query dtype에 맞춰 cast해도 됨. 여기선 float32로 만들어도 무방.
        attn_bias = torch.zeros(B, 1, L, L, device=gt_coords.device, dtype=torch.float32)

        hs = hotspot_mask.bool()

        # Ab/Ag split
        ab = is_antibody.bool()
        ag = is_antigen.bool()

        # Ab-hotspot -> Ag-hotspot
        q_hs_ab = hs & ab                      # [B, L]
        k_hs_ag = hs & ag                      # [B, L]

        cross_hs_mask = q_hs_ab[:, :, None] & k_hs_ag[:, None, :]   # [B, L, L]
        cross_hs_mask = cross_hs_mask & pair_pad_mask               # pad 제외

        if beta_cross != 0.0:
            attn_bias = attn_bias.masked_fill(cross_hs_mask[:, None, :, :], beta_cross)

        # (옵션) 반대 방향도 주고 싶으면 (Ag-hotspot -> Ab-hotspot)
        cross_hs_mask_rev = (hs & ag)[:, :, None] & (hs & ab)[:, None, :]
        cross_hs_mask_rev = cross_hs_mask_rev & pair_pad_mask
        attn_bias = attn_bias.masked_fill(cross_hs_mask_rev[:, None, :, :], beta_cross)

        return hotspot_mask, attn_bias
    
    def compute_cdr_index(
        self,
        residue_index: torch.Tensor,   # [B, L] int
        chain_index: torch.Tensor,     # [B, L] int (0=H, 1=L, >=2=Ag, -1=special)
        hotspot_mask: torch.Tensor,    # [B, L] bool (Ag epitope)
        attention_mask: torch.Tensor,
        beta_cdr_epi: float = 4.0,     # CDR <-> epitope bias strength
        bidirectional: bool = True,    # epitope->cdr도 줄지
    ):
        """
        Returns:
        cdr_epi_index: [B, L] int64
            0: none
            1,2,3: Heavy CDR1/2/3
            4,5,6: Light CDR1/2/3
            7: Antigen epitope
        """
        device = residue_index.device
        B, L = residue_index.shape

        resseq = residue_index
        ci = chain_index
        hs = hotspot_mask.bool()

        out = torch.zeros((B, L), device=device, dtype=torch.long)

        # -----------------------
        # valid residue
        valid = (resseq >= 0) & (ci >= 0)
        valid = valid & attention_mask.bool()

        # chain masks
        is_h = (ci == 0)
        is_l = (ci == 1)
        is_ag = (ci >= 2)

        # -----------------------
        # Heavy chain CDRs (1,2,3)
        m = valid & is_h
        out[m & (resseq >= 27) & (resseq <= 38)] = 1   # H-CDR1
        out[m & (resseq >= 56) & (resseq <= 65)] = 2   # H-CDR2
        out[m & (resseq >= 105) & (resseq <= 117)] = 3 # H-CDR3

        # -----------------------
        # Light chain CDRs (4,5,6)
        m = valid & is_l
        out[m & (resseq >= 27) & (resseq <= 38)] = 4   # L-CDR1
        out[m & (resseq >= 56) & (resseq <= 65)] = 5   # L-CDR2
        out[m & (resseq >= 105) & (resseq <= 117)] = 6 # L-CDR3

        # -----------------------
        # Antigen epitope (7)
        out[valid & is_ag & hs] = 7

        attn_bias = torch.zeros((B, 1, L, L), device=device, dtype=torch.float32)

        cdr_mask = (out >= 1) & (out <= 6)    # [B, L]
        epi_mask = (out == 7)                 # [B, L]

        # padding 제외 pair mask
        pair_valid = valid[:, :, None] & valid[:, None, :]  # [B, L, L]

        # CDR(query) -> EPI(key)
        cdr_to_epi = cdr_mask[:, :, None] & epi_mask[:, None, :] & pair_valid  # [B, L, L]
        if beta_cdr_epi != 0.0:
            attn_bias = attn_bias.masked_fill(cdr_to_epi[:, None, :, :], beta_cdr_epi)

        # (옵션) EPI(query) -> CDR(key) 도 추가
        if bidirectional and beta_cdr_epi != 0.0:
            epi_to_cdr = epi_mask[:, :, None] & cdr_mask[:, None, :] & pair_valid
            attn_bias = attn_bias.masked_fill(epi_to_cdr[:, None, :, :], beta_cdr_epi)

        return out, attn_bias

    def compute_interface_hotspot_min_dist_loss(
        self,
        gt_coords: torch.Tensor,       # [B, L, *, 3]  (GT, hotspot 판정용)
        pred_coords: torch.Tensor,     # [B, L, 3, 3] or [B, L, *, 3] (Pred, loss 계산용)
        attention_mask: torch.Tensor,  # [B, L] bool
        chain_index: torch.Tensor,     # [B, L] int (0,1=Ab; >=2=Ag)
        contact_cut: float = 8.0,      # hotspot 판정용 (GT에서 interface residue 정의)
        d0: float = 8.0,               # pred에서 min-distance hinge threshold
        big: float = 1e6,
        ca_idx: int = 1,               # CA index
    ):
        """
        Returns:
        hotspot_dist_loss: scalar
        hotspot_mask: [B, L] bool  (GT로 정의된 interface residue)
        """
        valid = attention_mask.bool()
        pair_pad_mask = valid[:, :, None] & valid[:, None, :]

        is_antibody = (chain_index < 2)
        is_antigen  = ~is_antibody

        # ---------- (A) hotspot mask: GT로 계산 ----------
        gt_ca = gt_coords[:, :, ca_idx, :]                 # [B, L, 3]
        gt_dist = torch.cdist(gt_ca, gt_ca, p=2.0)         # [B, L, L]

        boundary_mask = (
            (is_antibody[:, :, None] & is_antigen[:, None, :]) |
            (is_antigen[:, :, None]  & is_antibody[:, None, :])
        )
        gt_contact = gt_dist < contact_cut
        gt_pair = pair_pad_mask & boundary_mask & gt_contact

        hotspot_mask = (gt_pair.any(dim=-1) | gt_pair.any(dim=-2)) & valid  # [B, L]

        # ---------- (B) loss: Pred 좌표로 계산 ----------
        # pred_coords가 [B,L,3,3] (N,CA,C)면 CA index=1로 동일하게 사용 가능
        pred_ca = pred_coords[:, :, ca_idx, :] if pred_coords.dim() == 4 else pred_coords[:, :, ca_idx, :]
        pred_dist = torch.cdist(pred_ca, pred_ca, p=2.0)   # [B, L, L]

        # min-distance hinge (soft-OR)
        hs_ab = hotspot_mask & is_antibody
        mask_ab_to_ag = hs_ab[:, :, None] & is_antigen[:, None, :] & pair_pad_mask
        dist_ab_to_ag = pred_dist.masked_fill(~mask_ab_to_ag, big)
        min_ab_to_ag, _ = dist_ab_to_ag.min(dim=-1)
        loss_ab = (torch.relu(min_ab_to_ag - d0).pow(2)[hs_ab].mean()
                if hs_ab.any() else pred_dist.new_tensor(0.0))

        hs_ag = hotspot_mask & is_antigen
        mask_ag_to_ab = hs_ag[:, :, None] & is_antibody[:, None, :] & pair_pad_mask
        dist_ag_to_ab = pred_dist.masked_fill(~mask_ag_to_ab, big)
        min_ag_to_ab, _ = dist_ag_to_ab.min(dim=-1)
        loss_ag = (torch.relu(min_ag_to_ab - d0).pow(2)[hs_ag].mean()
                if hs_ag.any() else pred_dist.new_tensor(0.0))

        hotspot_dist_loss = loss_ab + loss_ag
        return hotspot_dist_loss, hotspot_mask

    def compute_cdr_epitope_min_dist_loss(
        self,
        pred_coords: torch.Tensor,     # [B, L, 3, 3] or [B, L, *, 3] (Pred, loss 계산용)
        attention_mask: torch.Tensor,  # [B, L] bool
        cdr_index: torch.Tensor,       # [B, L] int (CDR 부분 인덱스, 1~6 = CDR, 7 = epitope)
        d0: float = 8.0,               # pred에서 min-distance hinge threshold
        big: float = 1e6,
        ca_idx: int = 1,               # CA index
    ):
        """
        Returns:
        min_dist_loss: scalar
        hotspot_mask: [B, L] bool  (GT로 정의된 interface residue)
        """
        valid = attention_mask.bool()
        pair_pad_mask = valid[:, :, None] & valid[:, None, :]

        # CDR과 Epitope 구분 (CDR 부분: cdr_index == 1, 2, 3, 4, 5, 6 / Epitope: cdr_index == 7)
        is_cdr = cdr_index <= 6  # CDR 부분
        is_epitope = cdr_index == 7  # Epitope 부분

        # ---------- (A) Pred 좌표로 CDR과 Epitope 간 거리 계산 ---------- 
        pred_ca = pred_coords[:, :, ca_idx, :] if pred_coords.dim() == 4 else pred_coords[:, :, ca_idx, :]
        pred_dist = torch.cdist(pred_ca, pred_ca, p=2.0)  # [B, L, L]  각 residue 간의 거리 계산

        # CDR과 Epitope 간 거리 계산 (CDR 부분과 Epitope 부분 사이)
        mask_cdr_to_epitope = is_cdr[:, :, None] & is_epitope[:, None, :] & pair_pad_mask  # CDR과 Epitope 사이의 거리
        dist_cdr_to_epitope = pred_dist.masked_fill(~mask_cdr_to_epitope, big)  # mask를 통해 해당 거리만 계산
        min_cdr_to_epitope, _ = dist_cdr_to_epitope.min(dim=-1)  # 각 CDR에 대해 최소 거리를 구함

        # Epitope에서 CDR로의 거리 계산 (Epitope 부분과 CDR 부분 사이)
        mask_epitope_to_cdr = is_epitope[:, :, None] & is_cdr[:, None, :] & pair_pad_mask  # Epitope과 CDR 사이의 거리
        dist_epitope_to_cdr = pred_dist.masked_fill(~mask_epitope_to_cdr, big)  # mask를 통해 해당 거리만 계산
        min_epitope_to_cdr, _ = dist_epitope_to_cdr.min(dim=-1)  # 각 Epitope에 대해 최소 거리를 구함

        # ---------- (B) min-distance hinge loss (soft-OR) ---------- 
        loss_cdr_to_epitope = (torch.relu(min_cdr_to_epitope - d0).pow(2)[is_cdr].mean()
                            if is_cdr.any() else pred_dist.new_tensor(0.0))

        loss_epitope_to_cdr = (torch.relu(min_epitope_to_cdr - d0).pow(2)[is_epitope].mean()
                            if is_epitope.any() else pred_dist.new_tensor(0.0))

        # 대칭 손실을 위해 두 손실을 합산
        min_dist_loss = (loss_cdr_to_epitope + loss_epitope_to_cdr) / 2.0

        return min_dist_loss

