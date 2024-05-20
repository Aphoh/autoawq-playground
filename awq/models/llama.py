import tqdm
from typing import List, Tuple
from .base import BaseAWQForCausalLM
from awq.utils.fused_utils import fuse_qkv
from awq.modules.fused.block import LlamaLikeBlock
from awq.modules.fused.model import LlamaLikeModel
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer as OldLlamaDecoderLayer,
    LlamaForCausalLM as OldLlamaForCausalLM,
    LlamaMLP
)
from awq.modules.fused.norm import FasterTransformerRMSNorm
import copy
import torch
from torch import nn

def set_skip_quant(a):
    setattr(a, "skip_quant", True)
    for m in a.modules():
        if isinstance(m, nn.Linear):
            setattr(m, "skip_quant", True)

class LlamaAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "LlamaDecoderLayer"
    max_seq_len_key = "max_position_embeddings"
    modules_to_not_convert = ["mlp.mlp1"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_mlp_split = False

    @staticmethod
    def fuse_layers(model: OldLlamaForCausalLM):
        fuser = LlamaFuser(model)
        fuser.fuse_transformer()

    @staticmethod
    def get_model_layers(model: OldLlamaForCausalLM):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module: OldLlamaDecoderLayer):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: OldLlamaForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

    def split_mlp(self, act_pcts, thresh):
        assert not self.is_mlp_split, "MLP is already split?"
        for i, module in enumerate(self.model.model.layers):
            assert isinstance(module.mlp, LlamaMLP)
            num_past = (act_pcts[i] > thresh).sum()
            #num_past = module.mlp.intermediate_size
            #num_past = module.mlp.intermediate_size // 2
            if num_past > module.mlp.intermediate_size - 64:
                print(f"Skipping quantization of mlp layer {i}")
                set_skip_quant(module.mlp)
            elif num_past < 64:
                print(f"Fully quantizing mlp layer {i}")
            else:
                print(f"Quantizing ~{num_past} channels from mlp layer {i}")
                module.mlp = SplitLlamaMLP(module.mlp, act_pcts[i], thresh)
        self.is_mlp_split = True

    @staticmethod
    def get_layers_for_scaling(module: OldLlamaDecoderLayer, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )

        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )

        if isinstance(module.mlp, LlamaMLP) and not getattr(module.mlp, "skip_quant", False):
            # linear 1
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[module.mlp.gate_proj, module.mlp.up_proj],
                    inp=input_feat["mlp.gate_proj"],
                    module2inspect=module.mlp,
                )
            )

            # linear 2
            layers.append(
                dict(
                    prev_op=module.mlp.up_proj,
                    layers=[module.mlp.down_proj],
                    inp=input_feat["mlp.down_proj"],
                )
            )
        elif isinstance(module.mlp, SplitLlamaMLP):
            # linear 1
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    unscale_ops=[module.mlp.mlp1.gate_proj, module.mlp.mlp1.up_proj],
                    layers=[module.mlp.mlp2.gate_proj, module.mlp.mlp2.up_proj],
                    inp=input_feat["mlp.mlp2.gate_proj"],
                    module2inspect=module.mlp.mlp2,
                )
            )

            # linear 2
            layers.append(
                dict(
                    prev_op=module.mlp.mlp2.up_proj,
                    layers=[module.mlp.mlp2.down_proj],
                    inp=input_feat["mlp.mlp2.down_proj"],
                )
            )

        return layers


class SplitLlamaMLP(nn.Module):
    def __init__(self, mlp: LlamaMLP, act_pcts: torch.Tensor, thresh: float = 0.5):
        super().__init__()
        self.config = mlp.config
        self.hidden_size = mlp.hidden_size
        self.intermediate_size = mlp.intermediate_size

        perm = torch.argsort(act_pcts, descending=True)        
        #perm = torch.randperm(self.intermediate_size)
        limit = torch.argwhere(act_pcts[perm] > thresh).max()
        limit = min(max(limit, 64), self.intermediate_size - 64)
        #limit = self.intermediate_size // 2
        limit = ((limit + 63) // 64) * 64

        gate1_weight = mlp.gate_proj.weight[perm[:limit], :]
        gate2_weight = mlp.gate_proj.weight[perm[limit:], :]
        up1_weight = mlp.up_proj.weight[perm[:limit], :]
        up2_weight = mlp.up_proj.weight[perm[limit:], :]
        down1_weight = mlp.down_proj.weight[:, perm[:limit]]
        down2_weight = mlp.down_proj.weight[:, perm[limit:]]

        cfg1 = copy.deepcopy(mlp.config)
        cfg1.intermediate_size = limit
        cfg2 = copy.deepcopy(mlp.config)
        cfg2.intermediate_size = mlp.intermediate_size - limit
        self.mlp1 = LlamaMLP(cfg1).to(device=gate1_weight.device, dtype=gate1_weight.dtype)
        self.mlp2 = LlamaMLP(cfg2).to(device=gate2_weight.device, dtype=gate2_weight.dtype)
        self.mlp2.down_proj.bias = None # don't duplicate bias
        print(self.mlp1, self.mlp2)

        self.mlp1.gate_proj.weight.data.copy_(gate1_weight)
        self.mlp2.gate_proj.weight.data.copy_(gate2_weight)
        self.mlp1.up_proj.weight.data.copy_(up1_weight)
        self.mlp2.up_proj.weight.data.copy_(up2_weight)
        self.mlp1.down_proj.weight.data.copy_(down1_weight)
        self.mlp2.down_proj.weight.data.copy_(down2_weight)
        if mlp.gate_proj.bias is not None:
            self.mlp1.gate_proj.bias.data.copy_(mlp.gate_proj.bias[perm[:limit]])
            self.mlp2.gate_proj.bias.data.copy_(mlp.gate_proj.bias[perm[limit:]])
            self.mlp1.up_proj.bias.data.copy_(mlp.up_proj.bias[perm[:limit]])
            self.mlp2.up_proj.bias.data.copy_(mlp.up_proj.bias[perm[limit:]])
            self.mlp1.down_proj.bias.data.copy_(mlp.down_proj.bias)

    def forward(self, x):
        return self.mlp1(x) + self.mlp2(x)

class LlamaFuser:
    def __init__(self, model: OldLlamaForCausalLM):
        self.model = model

        self.llama_blocks: List[Tuple[str, OldLlamaDecoderLayer]] = [
            (name, module)
            for name, module in self.model.named_modules()
            if "LlamaDecoderLayer".lower() in module.__class__.__name__.lower()
        ]

    def fuse_transformer(self):
        blocks = []

        module: OldLlamaDecoderLayer
        for module in tqdm.tqdm(self.model.model.layers, desc="Fusing layers..."):
            device = next(iter(module.state_dict().values())).device
            qkv = fuse_qkv(
                module,
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            )
            norm_1 = FasterTransformerRMSNorm(
                module.input_layernorm.weight, module.input_layernorm.variance_epsilon
            )
            norm_2 = FasterTransformerRMSNorm(
                module.post_attention_layernorm.weight,
                module.post_attention_layernorm.variance_epsilon,
            )
            blocks.append(
                LlamaLikeBlock(
                    hidden_size=self.model.config.hidden_size,
                    n_heads=self.model.config.num_attention_heads,
                    n_kv_heads=self.model.config.num_key_value_heads,
                    qkv_layer=qkv,
                    o_proj=module.self_attn.o_proj,
                    mlp=module.mlp,
                    norm_1=norm_1,
                    norm_2=norm_2,
                    dev=device,
                    max_seq_len=self.model.config.max_seq_len,
                    rope_theta=self.model.config.rope_theta,
                )
            )

        self.model.model = LlamaLikeModel(
            self.model.config.vocab_size,
            blocks,
            self.model.model.embed_tokens,
            self.model.model.norm,
        )
        setattr(self.model.model, "blocks", self.model.model.blocks)
