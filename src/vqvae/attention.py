import functools
import math
import einops
import torch
import torch.nn.functional as F
from torch import nn

from esm.layers.rotary import RotaryEmbedding


def masked_softmax(x, mask, **kwargs):
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")

    return torch.softmax(x_masked, **kwargs)

class MaskedSoftmax(nn.Module):
    def __init__(self):
        super(MaskedSoftmax, self).__init__()
        self.softmax = nn.Softmax(1)

    def forward(self, x, mask=None):
        """
        Performs masked softmax, as simply masking post-softmax can be
        inaccurate
        :param x: [batch_size, num_items]
        :param mask: [batch_size, num_items]
        :return:
        """
        if mask is not None:
            mask = mask.float()
        if mask is not None:
            x_masked = x * mask + (1 - 1 / mask)
        else:
            x_masked = x
        x_max = x_masked.max(1)[0]
        x_exp = (x - x_max.unsqueeze(-1)).exp()
        if mask is not None:
            x_exp = x_exp * mask.float()
        return x_exp / x_exp.sum(1).unsqueeze(-1)

class VanillaMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        qk_layernorm: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_head = self.d_model // self.n_heads
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=bias)
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()

        self.rotary = RotaryEmbedding(d_model // n_heads)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def scaled_dot_product_attention(self, query, key, value, attn_mask):
        # https://github.com/pytorch/pytorch/issues/103963
        B = query.shape[0]
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(B, 1, L, S, dtype=query.dtype, device=query.device)

        attn_bias.masked_fill_(attn_mask.logical_not(), -1e9)
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)

        return attn_weight, attn_weight @ value

    def scaled_dot_product_attention_w_bias(self, query, key, value, attn_mask, ext_attn_bias):
        # https://github.com/pytorch/pytorch/issues/103963
        B = query.shape[0]
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))

        # 1) mask bias (-1e9)만
        mask_bias = torch.zeros(B, 1, L, S, dtype=query.dtype, device=query.device)
        mask_bias.masked_fill_(attn_mask.logical_not(), -1e9)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight = attn_weight + mask_bias

        attn_weight = attn_weight + ext_attn_bias.to(dtype=attn_weight.dtype, device=attn_weight.device)

        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight, attn_weight @ value

    # def forward(self, x, attention_mask, seq_id):
    def forward(self, x, attention_mask, chain_id, attn_bias: torch.Tensor | None = None):
        """
        x: [B, L, d_model]
        attention_mask: [B, L]
        seq_id: [B, L]
        """
        qkv_BLD3 = self.layernorm_qkv(x) # [B, L, d_model * 3]
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        query_BLD, key_BLD = self.q_ln(query_BLD), self.k_ln(key_BLD)
        query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)

        n_heads = self.n_heads
        reshaper = functools.partial(
            einops.rearrange, pattern="b s (h d) -> b h s d", h=n_heads
        )

        query_BHLD, key_BHLD, value_BHLD = map(
            reshaper, (query_BLD, key_BLD, value_BLD)
        )

        # padding에 관한 attention_mask
        # attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        # attention_mask = attention_mask.expand(-1, 1, -1, -1)  # [B, 1, L, L]

        if chain_id is not None:
            mask_BLL = chain_id.unsqueeze(-1) == chain_id.unsqueeze(-2)
            attn_mask_BLL = torch.logical_and(attention_mask.unsqueeze(-1), attention_mask.unsqueeze(1)) # [B, L, L]
            mask_BHLL = torch.logical_and(mask_BLL, attn_mask_BLL).unsqueeze(1) # [B, 1, L, L]

            if attn_bias is None:

                attn_weight, context_BHLD = self.scaled_dot_product_attention(
                    query_BHLD, key_BHLD, value_BHLD, mask_BHLL
                ) 

            else:
                attn_weight, context_BHLD = self.scaled_dot_product_attention_w_bias(
                    query_BHLD, key_BHLD, value_BHLD, mask_BHLL, attn_bias
                )

            # context_BHLD = F.scaled_dot_product_attention(
            #     query_BHLD, key_BHLD, value_BHLD, mask_BHLL
            # ) 

        else:
            assert 0
            # Shortcut, if we don't use attention biases then torch
            # will autoselect flashattention as the implementation
            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD
            )

        context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")
        # import pdb; pdb.set_trace()

        return attn_weight, self.out_proj(context_BLD)

        # if seq_id is not None:
        #     # Where True, enable participation in attention.
        #     mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
        #     attn_mask_BLL = torch.logical_and(attention_mask.unsqueeze(-1), attention_mask.unsqueeze(1)) # [B, L, L]
        #     mask_BHLL = torch.logical_and(mask_BLL, attn_mask_BLL).unsqueeze(1) # [B, 1, L, L]
            
        #     context_BHLD = F.scaled_dot_product_attention(
        #         query_BHLD, key_BHLD, value_BHLD, mask_BHLL.to(torch.float)
        #     ) # https://github.com/pytorch/pytorch/issues/103963
            # '''
            # # Efficient implementation equivalent to the following:
            # def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
            #     B = query.shape[0]
            #     L, S = query.size(-2), key.size(-2)
            #     scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
            #     attn_bias = torch.zeros(B, 1, L, S, dtype=query.dtype, device=query.device)
            #     if is_causal:
            #         assert attn_mask is None
            #         temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            #         attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            #         attn_bias.to(query.dtype)

            #     if attn_mask is not None:
            #         if attn_mask.dtype == torch.bool:
            #             attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            #         else:
            #             attn_bias += attn_mask
            #     attn_weight = query @ key.transpose(-2, -1) * scale_factor
            #     attn_weight += attn_bias
            #     attn_weight = torch.softmax(attn_weight, dim=-1)
            #     attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
            #     return attn_weight @ value
            # '''
        # else:
        #     assert 0
        #     # Shortcut, if we don't use attention biases then torch
        #     # will autoselect flashattention as the implementation
        #     context_BHLD = F.scaled_dot_product_attention(
        #         query_BHLD, key_BHLD, value_BHLD
        #     )
        # context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")
        # return self.out_proj(context_BLD)