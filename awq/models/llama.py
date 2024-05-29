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
from dataclasses import dataclass
from typing import Optional

def set_skip_quant(a):
    setattr(a, "skip_quant", True)
    for m in a.modules():
        if isinstance(m, nn.Linear):
            setattr(m, "skip_quant", True)

@dataclass
class SplitConfig:
    n_split_constant: Optional[int] = None
    """Split each mlp into n_split_constant channels per layer"""
    n_split_top_thresh: Optional[float] = None
    """Split each mlp into channels with activation percentage greater than this value"""
    n_split_bottom_thresh: Optional[float] = None
    """Split each mlp into channels with activation percentage less than this value"""

    random_cols: bool = False
    """Randomly choose columns to split"""
    top_cols: bool = False
    """Pick the top n_split columns with the highest activation percentage to split"""
    bottom_cols: bool = False
    """Pick the bottom n_split columns with the highest activation percentage to split"""

    def __post_init__(self):
        assert sum([self.random_cols, self.top_cols, self.bottom_cols]) == 1, "Please specify only one of random_cols, top_cols or bottom_cols"
        assert sum(a is not None for a in [self.n_split_constant, self.n_split_top_thresh]) == 1, "Please specify only one of n_split_constant or n_split_gt0_thresh"

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

    def split_mlp(self, cfg: SplitConfig, split_metric: torch.Tensor=None):
        assert not self.is_mlp_split, "MLP is already split?"
        split_metric = split_metric.to(self.model.device)
        quantized_counts = []
        non_quant_counts = []
        for i, module in enumerate(self.model.model.layers):
            assert isinstance(module.mlp, LlamaMLP)
            if cfg.n_split_constant is not None:
                assert cfg.n_split_constant % 64 == 0, "n_split_constant must be a multiple of 64"
                n_split = cfg.n_split_constant
            elif cfg.n_split_top_thresh is not None:
                n_split = get_n_split_thresh(split_metric[i], cfg.n_split_top_thresh)
            elif cfg.n_split_bottom_thresh is not None:
                n_split = get_n_split_thresh(split_metric[i], cfg.n_split_bottom_thresh)
            else:
                raise ValueError("expected either n_split_constant or n_split_gt0_thresh")
            non_quant_counts.append(n_split)
            quantized_counts.append(module.mlp.intermediate_size - n_split)

            if n_split == 0:
                print("Full quantizing mlp layer", i)
                continue
            elif n_split == module.mlp.intermediate_size:
                print("Skipping quantization of mlp layer", i)
                set_skip_quant(module.mlp)
                continue

            if cfg.random_cols:
                s1inds, s2inds = split_inds_random(module.mlp.intermediate_size, n_split)
            elif cfg.top_cols:
                assert split_metric is not None, "split_metric must be provided for top_cols"
                s1inds, s2inds = split_inds_threshold(split_metric[i], n_split, top=True)
            elif cfg.bottom_cols:
                assert split_metric is not None, "split_metric must be provided for bottom_cols"
                s2inds, s1inds = split_inds_threshold(split_metric[i], n_split, top=False)

            print(f"Quantizing {n_split} channels from mlp layer {i}")
            module.mlp = SplitLlamaMLP(module.mlp, s1inds, s2inds)
        self.is_mlp_split = True
        return quantized_counts, non_quant_counts

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


def roundbase(x, base=64):
    if isinstance(x, torch.Tensor):
        x = x.item()
    return int(round(base*round(x/base)))

def split_inds_random(intermediate_size, n_split):
    perm = torch.randperm(intermediate_size)
    return perm[:n_split], perm[n_split:]

def split_inds_threshold(split_metric, n_split, top=True):
    perm = torch.argsort(split_metric, descending=top)
    return perm[:n_split], perm[n_split:]

def get_n_split_thresh(split_metric: torch.Tensor, thresh: float) -> int:
    n_split = torch.sum(split_metric > thresh)
    n_split = roundbase(n_split, base=64)
    return n_split

class SplitLlamaMLP(nn.Module):
    def __init__(self, mlp: LlamaMLP, s1inds: torch.Tensor, s2inds: torch.Tensor):
        super().__init__()
        self.config = mlp.config
        self.hidden_size = mlp.hidden_size
        self.intermediate_size = mlp.intermediate_size

        gate1_weight = mlp.gate_proj.weight[s1inds, :]
        gate2_weight = mlp.gate_proj.weight[s2inds, :]
        up1_weight = mlp.up_proj.weight[s1inds, :]
        up2_weight = mlp.up_proj.weight[s2inds, :]
        down1_weight = mlp.down_proj.weight[:, s1inds]
        down2_weight = mlp.down_proj.weight[:, s2inds]

        cfg1 = copy.deepcopy(mlp.config)
        cfg1.intermediate_size = len(s1inds)
        cfg2 = copy.deepcopy(mlp.config)
        cfg2.intermediate_size = len(s2inds)
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
            self.mlp1.gate_proj.bias.data.copy_(mlp.gate_proj.bias[s1inds])
            self.mlp2.gate_proj.bias.data.copy_(mlp.gate_proj.bias[s2inds])
            self.mlp1.up_proj.bias.data.copy_(mlp.up_proj.bias[s1inds])
            self.mlp2.up_proj.bias.data.copy_(mlp.up_proj.bias[s2inds])
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