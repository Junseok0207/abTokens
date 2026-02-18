import torch
import torch.nn as nn
from esm.utils.structure.affine3d import (
    Affine3D,
    build_affine3d_from_coordinates,
)
from src.vqvae_model import VanillaStructureTokenEncoder, node_gather
from vqvae.blocks import VanillaUnifiedTransformerBlock

from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, SAGEConv, Sequential, APPNP

class ComplexRelativePositionEmbedding(nn.Module):
    """
    Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/models/vqvae.py#L20C1-L53C1
    """

    def __init__(self, bins, embedding_dim, init_std=0.02):
        super().__init__()
        self.bins = bins

        self.embedding = torch.nn.Embedding(2 * bins + 2, embedding_dim)
        self.embedding.weight.data.normal_(0, init_std)

    def forward(self, query_residue_index, key_residue_index, query_chain_id=None, key_chain_id=None, knn_edge_mask=None):
        """
        Input:
          query_residue_index: (B, ) tensor of source indices (dytpe=torch.long)
          key_residue_index: (B, L) tensor of target indices (dytpe=torch.long)
        Output:
          embeddings: B x L x embedding_dim tensor of embeddings
        """

        assert query_residue_index.dtype == torch.long
        assert key_residue_index.dtype == torch.long
        assert query_residue_index.ndim == 1
        assert key_residue_index.ndim == 2

        diff = key_residue_index - query_residue_index.unsqueeze(1)
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1  # add 1 to adjust for padding index

        if key_chain_id is not None:
            same_chain = key_chain_id == query_chain_id.unsqueeze(1) # [B * L, 16]
            diff = torch.where(same_chain, diff, 0) # 0번 index가 padding index인거 같은데 different chain인 경우 이걸로 우선 배정해줬음

        # knn_edge_mask # 해야됨 - 지금은 padding된 mask에 대해서 z가 들어가고 있음
        output = self.embedding(diff) # [B * L, 16, d_model=1024]
        return output


class GNNLayer(nn.Module):
    def __init__(self, layer_sizes, batchnorm_mm=0.99):
        super().__init__()

        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        
        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            #layers.append( (nn.Dropout(drop_rate), 'x -> x'), )
            layers.append((GCNConv(in_dim, out_dim), 'x, edge_index -> x'),)
            layers.append(LayerNorm(out_dim))
            # layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            layers.append(nn.PReLU())

        self.model = Sequential('x, edge_index', layers)

    def forward(self, x, edge_index, edge_mask):
        edge_index = self.knn_edges_to_edge_index(edge_index, edge_mask)        
        # TODO 여기서 edge_index에 맞게 x를 다시 gathering 해줘야 할 듯

        return self.model(x[:,0,:], edge_index)

    def knn_edges_to_edge_index(self, knn_edges, knn_edge_mask):
        # remove self-loops
        knn_edges = knn_edges[:,:,1:]
        knn_edge_mask = knn_edge_mask[:,:,1:]

        batch_size, num_nodes, num_neighbors = knn_edges.shape

        batch_offsets = (torch.arange(batch_size, device=knn_edges.device) * num_nodes).unsqueeze(1)

        source_nodes = torch.arange(num_nodes, device=knn_edges.device).unsqueeze(0).repeat(batch_size, 1)  # [batch_size, num_nodes]
        source_nodes += batch_offsets
        source_nodes = source_nodes.unsqueeze(-1).repeat(1, 1, num_neighbors)  # [batch_size, num_nodes, num_neighbors]
        source_nodes = source_nodes.view(-1) # Flatten the source nodes

        dest_nodes = knn_edges + batch_offsets.unsqueeze(-1)  # Apply offset for destination nodes
        dest_nodes = dest_nodes.view(-1) # Flatten the destination nodes

        valid_edge_mask = knn_edge_mask.reshape(-1)  # Shape: [batch_size * num_nodes * num_neighbors]
        valid_edge_indices = torch.nonzero(valid_edge_mask).squeeze(-1)  # Indices of valid edges

        valid_source_nodes = source_nodes[valid_edge_indices]
        valid_dest_nodes = dest_nodes[valid_edge_indices]
        edge_index = torch.stack([valid_source_nodes, valid_dest_nodes], dim=0)

        return edge_index

    def reset_parameters(self):
        self.model.reset_parameters()


class GeometricEncoderStack_w_adapter(nn.Module):
    """
    A stack of transformer blocks used in the ESM-3 model. Each block is a UnifiedTransformerBlock,
    which can either be geometric attention or standard multi-head attention.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads.
        v_heads (int): The number of voting heads.
        n_layers (int): The number of transformer blocks in the stack.
        n_layers_geom (int, optional): The number of transformer blocks that use geometric attention.
        scale_residue (bool, optional): Whether to scale the residue connections in each transformer block.
        mask_and_zero_frameless (bool, optional): Whether to mask and zero frameless positions in the input.
            Only applies in the geometric attention blocks, which is conditioned on the structure
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        v_heads: int | None,
        n_layers: int,
        adapter: str
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                VanillaUnifiedTransformerBlock(
                    d_model,
                    n_heads,
                    v_heads=v_heads,
                    use_geom_attn=True,
                    use_plain_attn=False,
                    expansion_ratio=4,
                    bias=True,
                )
                for i in range(n_layers)
            ]
        )

        # GNN으로 바꿔야 됨
        if adapter == 'mlp':
            self.adapter_blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(d_model, elementwise_affine=False),
                        nn.Linear(d_model, d_model // 4),
                        nn.ReLU(),
                        nn.Linear(d_model // 4, d_model),
                    )
                    for _ in range(n_layers)
                ]
            )

        elif adapter == 'gnn':
            self.adapter_blocks = nn.ModuleList(
                [
                    GNNLayer([d_model, d_model // 4, d_model])
                    for _ in range(n_layers)
                ]
            )

        self.norm = nn.LayerNorm(d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        affine: Affine3D | None = None,
        affine_mask: torch.Tensor | None = None,
        chain_id: torch.Tensor | None = None,
        knn_edges: torch.Tensor | None = None,
        knn_edge_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the TransformerStack.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, d_model).
            sequence_id (torch.Tensor): The sequence ID tensor of shape (batch_size, sequence_length).
            affine (Affine3D | None): The affine transformation tensor or None.
            affine_mask (torch.Tensor | None): The affine mask tensor or None.
            chain_id (torch.Tensor): The protein chain tensor of shape (batch_size, sequence_length).
                Only used in geometric attention.

        Returns:
            post_norm: The output tensor of shape (batch_size, sequence_length, d_model).
            pre_norm: The embedding of shape (batch_size, sequence_length, d_model).
        """
        *batch_dims, _ = x.shape
        if chain_id is None:
            chain_id = torch.ones(size=batch_dims, dtype=torch.int64, device=x.device)

        for i in range(len(self.blocks)):
            block = self.blocks[i]
            adapter = self.adapter_blocks[i]
            x = block(x, attention_mask, sequence_id, affine, affine_mask, chain_id) # [B * L, 16, d_model]
            self_x = adapter(x, knn_edges, knn_edge_mask)   # [B * L, d_model]
            # import pdb; pdb.set_trace()
            # x = x + self_x.unsqueeze(1)  # broadcast to [B * L, 16, d_model]

            updated_x = x.clone()  # Clone the original x to avoid inplace operation
            updated_x[:, 0, :] = updated_x[:, 0, :] + self_x  # No inplace operation
            x = updated_x  # Update x with the new tensor

        return self.norm(x), x


class ComplexStructureTokenEncoder(VanillaStructureTokenEncoder):
    """
    A specialized encoder for complex protein structures, extending the base structure token encoder.
    This class is designed to handle the complexities of protein chains and their interactions.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.blocked = kwargs.get('blocked', True)
        self.adapter = kwargs.get('adapter', None)

        d_model = kwargs['d_model']
        n_heads = kwargs['n_heads']
        v_heads = kwargs['v_heads']
        n_layers = kwargs['n_layers']
        self.relative_positional_embedding = ComplexRelativePositionEmbedding(32, d_model, init_std=0.02)

        if self.adapter is not None:
            self.transformer = GeometricEncoderStack_w_adapter(d_model, n_heads, v_heads, n_layers, self.adapter)

    def encode(
        self,
        coords: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        residue_index: torch.Tensor | None = None,
        chain_index: torch.Tensor | None = None
    ):
        coords = coords[..., :3, :] # -> [B, L, 3, 3]
        # backbone coordinate을 받아서 translation과 rotation을 계산 (affine_mask는 padding같은 유효하지 않은 부분 - 평균값으로 계산된 affine)

        affine, affine_mask = build_affine3d_from_coordinates(coords=coords) # affine: [B, L], affine_mask: [B, L]

        if sequence_id is None:
            sequence_id = torch.zeros_like(affine_mask, dtype=torch.int64)

        # if self.blocked == False:
        #     chain_index = None # chain_index를 안 넣으면 attention 계산시에 attn_bias가 -inf가 안 되서 chain 구분 없이 다 attention을 보게 됨

        if self.adapter == "gnn":
            z = self.encode_local_structure_w_gnn(
                coords=coords,
                affine=affine,
                attention_mask=attention_mask,
                sequence_id=sequence_id,
                affine_mask=affine_mask,
                residue_index=residue_index,
                chain_index=chain_index
            ) # [B, L, d_model]

        elif self.adapter is None:
            z = self.encode_local_structure(
                coords=coords,
                affine=affine,
                attention_mask=attention_mask,
                sequence_id=sequence_id,
                affine_mask=affine_mask,
                residue_index=residue_index,
                chain_index=chain_index
            ) # [B, L, d_model]

        z = z.masked_fill(~affine_mask.unsqueeze(2), 0) # [B, L, d_model]
        z = self.pre_vq_proj(z) # [B, L, d_out]

        return z

    def encode_w_aug(
        self,
        coords: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        residue_index: torch.Tensor | None = None,
        chain_index: torch.Tensor | None = None
    ):
        coords = coords[..., :3, :] # -> [B, L, 3, 3]
        # backbone coordinate을 받아서 translation과 rotation을 계산 (affine_mask는 padding같은 유효하지 않은 부분 - 평균값으로 계산된 affine)

        affine, affine_mask = build_affine3d_from_coordinates(coords=coords) # affine: [B, L], affine_mask: [B, L]

        if sequence_id is None:
            sequence_id = torch.zeros_like(affine_mask, dtype=torch.int64)

        z1, z2 = self.encode_local_structure_w_aug(
            coords=coords,
            affine=affine,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            affine_mask=affine_mask,
            residue_index=residue_index,
            chain_index=chain_index
        ) # [B, L, d_model]

        z1 = z1.masked_fill(~affine_mask.unsqueeze(2), 0) # [B, L, d_model]
        z1 = self.pre_vq_proj(z1) # [B, L, d_out]



        return z1, z2


    def encode_local_structure_w_aug(
        self,
        coords: torch.Tensor,
        affine: Affine3D,
        attention_mask: torch.Tensor,
        sequence_id: torch.Tensor | None,
        affine_mask: torch.Tensor,
        residue_index: torch.Tensor | None = None,
        chain_index: torch.Tensor | None = None
    ):
        """This function allows for a multi-layered encoder to encode tokens with a local receptive fields. The implementation is as follows:

        1. Starting with (B, L) frames, we find the KNN in structure space. This now gives us (B, L, K) where the last dimension is the local
        neighborhood of all (B, L) residues.
        2. We reshape these frames to (B*L, K) so now we have a large batch of a bunch of local neighborhoods.
        3. Pass the (B*L, K) local neighborhoods through a stack of geometric reasoning blocks, effectively getting all to all communication between
        all frames in the local neighborhood.
        4. This gives (B*L, K, d_model) embeddings, from which we need to get a single embedding per local neighborhood. We do this by simply
        taking the embedding corresponding to the query node. This gives us (B*L, d_model) embeddings.
        5. Reshape back to (B, L, d_model) embeddings
        """
        assert coords.size(-1) == 3 and coords.size(-2) == 3, "need N, CA, C"
        
        with torch.no_grad():
            knn_edges, knn_edge_mask = self.find_knn_edges(
                coords,
                ~attention_mask,
                coord_mask=affine_mask,
                sequence_id=sequence_id,
                knn=self.knn,
            )

            B, L, E = knn_edges.shape
            knn_edge_mask = knn_edge_mask.view(-1, E) # (B * L, 16)

            affine_tensor = affine.tensor  # for easier manipulation # [B, L, 12]
            T_D = affine_tensor.size(-1)
            knn_affine_tensor = node_gather(affine_tensor, knn_edges) # [B, L, 16, 12]
            knn_affine_tensor = knn_affine_tensor.view(-1, E, T_D).contiguous() # [B * L, 16, 12]
            affine = Affine3D.from_tensor(knn_affine_tensor) # [B * L, 16] # 각 dimension에 affine이 들어가있음
            knn_sequence_id = (
                node_gather(sequence_id.unsqueeze(-1), knn_edges).view(-1, E)
                if sequence_id is not None
                else torch.zeros(B * L, E, dtype=torch.int64, device=coords.device)
            ) # [B * L, 16]

            knn_attention_mask = (
                node_gather(attention_mask.unsqueeze(-1), knn_edges).view(-1, E)
                if attention_mask is not None
                else torch.zeros(B * L, E, dtype=torch.int64, device=coords.device)
            ) # [B * L, 16]
            knn_attention_mask = torch.logical_and(knn_attention_mask, knn_edge_mask)

            knn_affine_mask = node_gather(affine_mask.unsqueeze(-1), knn_edges).view(
                -1, E
            ) # [B * L, 16]
            knn_affine_mask = torch.logical_and(knn_affine_mask, knn_edge_mask)

            if residue_index is None:
                res_idxs = knn_edges.view(-1, E)
            else:
                res_idxs = node_gather(residue_index.unsqueeze(-1), knn_edges).view(
                    -1, E
                ) # [B * L, 16]

            if chain_index is None:
                knn_chain_id = None
                z = self.relative_positional_embedding(res_idxs[:, 0], res_idxs) # [B * L, 16, d_model] # 이것도 chain반영 필요
            else:
                knn_chain_id = node_gather(chain_index.unsqueeze(-1), knn_edges).view(-1, E) # [B * L, 16]
                z = self.relative_positional_embedding(res_idxs[:, 0], res_idxs, knn_chain_id[:, 0], knn_chain_id, knn_edge_mask) # [B * L, 16, d_model]

        # augmentation
        knn_affine_mask_aug1 = knn_affine_mask.clone()
        knn_affine_mask_aug2 = knn_affine_mask.clone()

        # make false in some aff
        prob = 0.1
        random_tensor = torch.rand(knn_affine_mask.shape, device=knn_affine_mask.device)
        random_mask = random_tensor < prob
        knn_affine_mask_aug1 = torch.logical_and(knn_affine_mask_aug1, ~random_mask)
        random_tensor = torch.rand(knn_affine_mask.shape, device=knn_affine_mask.device)
        random_mask = random_tensor < prob
        knn_affine_mask_aug2 = torch.logical_and(knn_affine_mask_aug2, ~random_mask)

        _, z1, _ = self.transformer.forward(
            x=z,
            attention_mask=knn_attention_mask,
            sequence_id=knn_sequence_id,
            affine=affine,
            affine_mask=knn_affine_mask_aug1,
            chain_id=knn_chain_id,
        ) # [B * L, 16, d_model]

        z1 = z1.view(B, L, E, -1) # [B, L, 16, d_model]
        z1 = z1[:, :, 0, :] # [B, L, d_model]

        _, z2, _ = self.transformer.forward(
            x=z,
            attention_mask=knn_attention_mask,
            sequence_id=knn_sequence_id,
            affine=affine,
            affine_mask=knn_affine_mask_aug2,
            chain_id=knn_chain_id,
        ) # [B * L, 16, d_model]

        z2 = z2.view(B, L, E, -1) # [B, L, 16, d_model]
        z2 = z2[:, :, 0, :] # [B, L, d_model]

        return z1, z2

    def encode_local_structure(
        self,
        coords: torch.Tensor,
        affine: Affine3D,
        attention_mask: torch.Tensor,
        sequence_id: torch.Tensor | None,
        affine_mask: torch.Tensor,
        residue_index: torch.Tensor | None = None,
        chain_index: torch.Tensor | None = None
    ):
        """This function allows for a multi-layered encoder to encode tokens with a local receptive fields. The implementation is as follows:

        1. Starting with (B, L) frames, we find the KNN in structure space. This now gives us (B, L, K) where the last dimension is the local
        neighborhood of all (B, L) residues.
        2. We reshape these frames to (B*L, K) so now we have a large batch of a bunch of local neighborhoods.
        3. Pass the (B*L, K) local neighborhoods through a stack of geometric reasoning blocks, effectively getting all to all communication between
        all frames in the local neighborhood.
        4. This gives (B*L, K, d_model) embeddings, from which we need to get a single embedding per local neighborhood. We do this by simply
        taking the embedding corresponding to the query node. This gives us (B*L, d_model) embeddings.
        5. Reshape back to (B, L, d_model) embeddings
        """
        assert coords.size(-1) == 3 and coords.size(-2) == 3, "need N, CA, C"
        
        with torch.no_grad():
            knn_edges, knn_edge_mask = self.find_knn_edges(
                coords,
                ~attention_mask,
                coord_mask=affine_mask,
                sequence_id=sequence_id,
                knn=self.knn,
            )

            B, L, E = knn_edges.shape

            try:
                knn_edge_mask = knn_edge_mask.view(-1, E) # (B * L, 16)
            except:
                import pdb; pdb.set_trace()

            affine_tensor = affine.tensor  # for easier manipulation # [B, L, 12]
            T_D = affine_tensor.size(-1)
            knn_affine_tensor = node_gather(affine_tensor, knn_edges) # [B, L, 16, 12]
            knn_affine_tensor = knn_affine_tensor.view(-1, E, T_D).contiguous() # [B * L, 16, 12]
            affine = Affine3D.from_tensor(knn_affine_tensor) # [B * L, 16] # 각 dimension에 affine이 들어가있음
            knn_sequence_id = (
                node_gather(sequence_id.unsqueeze(-1), knn_edges).view(-1, E)
                if sequence_id is not None
                else torch.zeros(B * L, E, dtype=torch.int64, device=coords.device)
            ) # [B * L, 16]

            knn_attention_mask = (
                node_gather(attention_mask.unsqueeze(-1), knn_edges).view(-1, E)
                if attention_mask is not None
                else torch.zeros(B * L, E, dtype=torch.int64, device=coords.device)
            ) # [B * L, 16]
            knn_attention_mask = torch.logical_and(knn_attention_mask, knn_edge_mask)

            knn_affine_mask = node_gather(affine_mask.unsqueeze(-1), knn_edges).view(
                -1, E
            ) # [B * L, 16]
            knn_affine_mask = torch.logical_and(knn_affine_mask, knn_edge_mask)

            if residue_index is None:
                res_idxs = knn_edges.view(-1, E)
            else:
                res_idxs = node_gather(residue_index.unsqueeze(-1), knn_edges).view(
                    -1, E
                ) # [B * L, 16]

            if chain_index is None:
                knn_chain_id = None
                z = self.relative_positional_embedding(res_idxs[:, 0], res_idxs) # [B * L, 16, d_model] # 이것도 chain반영 필요
            else:
                knn_chain_id = node_gather(chain_index.unsqueeze(-1), knn_edges).view(-1, E) # [B * L, 16]
                z = self.relative_positional_embedding(res_idxs[:, 0], res_idxs, knn_chain_id[:, 0], knn_chain_id, knn_edge_mask) # [B * L, 16, d_model]

        attn_weights, z, _ = self.transformer.forward(
            x=z,
            attention_mask=knn_attention_mask,
            sequence_id=knn_sequence_id,
            affine=affine,
            affine_mask=knn_affine_mask,
            chain_id=knn_chain_id,
        ) # [B * L, 16, d_model]

        # import pdb; pdb.set_trace()
        # Unflatten the output and take the query node embedding, which will always be the first one because
        # a node has distance 0 with itself and the KNN are sorted.
        z = z.view(B, L, E, -1) # [B, L, 16, d_model]
        z = z[:, :, 0, :] # [B, L, d_model]

        return z
    

    def encode_local_structure_w_gnn(
        self,
        coords: torch.Tensor,
        affine: Affine3D,
        attention_mask: torch.Tensor,
        sequence_id: torch.Tensor | None,
        affine_mask: torch.Tensor,
        residue_index: torch.Tensor | None = None,
        chain_index: torch.Tensor | None = None
    ):
        """This function allows for a multi-layered encoder to encode tokens with a local receptive fields. The implementation is as follows:

        1. Starting with (B, L) frames, we find the KNN in structure space. This now gives us (B, L, K) where the last dimension is the local
        neighborhood of all (B, L) residues.
        2. We reshape these frames to (B*L, K) so now we have a large batch of a bunch of local neighborhoods.
        3. Pass the (B*L, K) local neighborhoods through a stack of geometric reasoning blocks, effectively getting all to all communication between
        all frames in the local neighborhood.
        4. This gives (B*L, K, d_model) embeddings, from which we need to get a single embedding per local neighborhood. We do this by simply
        taking the embedding corresponding to the query node. This gives us (B*L, d_model) embeddings.
        5. Reshape back to (B, L, d_model) embeddings
        """
        assert coords.size(-1) == 3 and coords.size(-2) == 3, "need N, CA, C"
        
        with torch.no_grad():
            knn_edges, knn_edge_mask = self.find_knn_edges(
                coords,
                ~attention_mask,
                coord_mask=affine_mask,
                sequence_id=sequence_id,
                knn=self.knn,
            )

            B, L, E = knn_edges.shape
            knn_edge_mask_ = knn_edge_mask.view(-1, E) # (B * L, 16)

            affine_tensor = affine.tensor  # for easier manipulation # [B, L, 12]
            T_D = affine_tensor.size(-1)
            knn_affine_tensor = node_gather(affine_tensor, knn_edges) # [B, L, 16, 12]
            knn_affine_tensor = knn_affine_tensor.view(-1, E, T_D).contiguous() # [B * L, 16, 12]
            affine = Affine3D.from_tensor(knn_affine_tensor) # [B * L, 16] # 각 dimension에 affine이 들어가있음
            knn_sequence_id = (
                node_gather(sequence_id.unsqueeze(-1), knn_edges).view(-1, E)
                if sequence_id is not None
                else torch.zeros(B * L, E, dtype=torch.int64, device=coords.device)
            ) # [B * L, 16]

            knn_attention_mask = (
                node_gather(attention_mask.unsqueeze(-1), knn_edges).view(-1, E)
                if attention_mask is not None
                else torch.zeros(B * L, E, dtype=torch.int64, device=coords.device)
            ) # [B * L, 16]
            knn_attention_mask = torch.logical_and(knn_attention_mask, knn_edge_mask_)

            knn_affine_mask = node_gather(affine_mask.unsqueeze(-1), knn_edges).view(
                -1, E
            ) # [B * L, 16]
            knn_affine_mask = torch.logical_and(knn_affine_mask, knn_edge_mask_)

            if residue_index is None:
                res_idxs = knn_edges.view(-1, E)
            else:
                res_idxs = node_gather(residue_index.unsqueeze(-1), knn_edges).view(
                    -1, E
                ) # [B * L, 16]

            if chain_index is None:
                knn_chain_id = None
                z = self.relative_positional_embedding(res_idxs[:, 0], res_idxs) # [B * L, 16, d_model] # 이것도 chain반영 필요
            else:
                knn_chain_id = node_gather(chain_index.unsqueeze(-1), knn_edges).view(-1, E) # [B * L, 16]
                z = self.relative_positional_embedding(res_idxs[:, 0], res_idxs, knn_chain_id[:, 0], knn_chain_id, knn_edge_mask) # [B * L, 16, d_model]

        z, _ = self.transformer.forward(
            x=z,
            attention_mask=knn_attention_mask,
            sequence_id=knn_sequence_id,
            affine=affine,
            affine_mask=knn_affine_mask,
            chain_id=knn_chain_id,
            knn_edges=knn_edges,
            knn_edge_mask=knn_edge_mask
        ) # [B * L, 16, d_model]

        # Unflatten the output and take the query node embedding, which will always be the first one because
        # a node has distance 0 with itself and the KNN are sorted.
        z = z.view(B, L, E, -1) # [B, L, 16, d_model]
        z = z[:, :, 0, :] # [B, L, d_model]

        return z