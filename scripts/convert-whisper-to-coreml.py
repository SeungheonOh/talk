import argparse
import torch
import torch.nn.functional as F
import coremltools as ct

from torch import Tensor
from torch import nn
from typing import Dict
from typing import Optional
from ane_transformers.reference.layer_norm import LayerNormANE as LayerNormANEBase
from coremltools.models.neural_network.quantization_utils import quantize_weights
from whisper.model import Whisper, AudioEncoder, TextDecoder, ResidualAttentionBlock, MultiHeadAttention, ModelDimensions
from whisper import load_model

import whisper.model
whisper.model.MultiHeadAttention.use_sdpa = False


def linear_to_conv2d_map(state_dict, prefix, local_metadata, strict,
                         missing_keys, unexpected_keys, error_msgs):
    for k in state_dict:
        is_attention = all(substr in k for substr in ['attn', '.weight'])
        is_mlp = any(k.endswith(s) for s in ['mlp.0.weight', 'mlp.2.weight'])
        if (is_attention or is_mlp) and len(state_dict[k].shape) == 2:
            state_dict[k] = state_dict[k][:, :, None, None]


def correct_for_bias_scale_order_inversion(state_dict, prefix, local_metadata,
                                           strict, missing_keys,
                                           unexpected_keys, error_msgs):
    state_dict[prefix + 'bias'] = state_dict[prefix + 'bias'] / state_dict[prefix + 'weight']
    return state_dict


class LayerNormANE(LayerNormANEBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_load_state_dict_pre_hook(correct_for_bias_scale_order_inversion)


class MultiHeadAttentionANE(MultiHeadAttention):
    def __init__(self, n_state: int, n_head: int):
        super().__init__(n_state, n_head)
        self.query = nn.Conv2d(n_state, n_state, kernel_size=1)
        self.key = nn.Conv2d(n_state, n_state, kernel_size=1, bias=False)
        self.value = nn.Conv2d(n_state, n_state, kernel_size=1)
        self.out = nn.Conv2d(n_state, n_state, kernel_size=1)

    def forward(self, x, xa=None, mask=None, kv_cache=None):
        q = self.query(x)
        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]
        wv, qk = self.qkv_attention_ane(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention_ane(self, q, k, v, mask=None):
        _, dim, _, seqlen = q.size()
        dim_per_head = dim // self.n_head
        scale = float(dim_per_head) ** -0.5
        q = q * scale
        mh_q = q.split(dim_per_head, dim=1)
        mh_k = k.transpose(1, 3).split(dim_per_head, dim=3)
        mh_v = v.split(dim_per_head, dim=1)
        mh_qk = [
            torch.einsum('bchq,bkhc->bkhq', [qi, ki])
            for qi, ki in zip(mh_q, mh_k)
        ]
        if mask is not None:
            for head_idx in range(self.n_head):
                mh_qk[head_idx] = mh_qk[head_idx] + mask[:, :seqlen, :, :seqlen]
        attn_weights = [aw.softmax(dim=1) for aw in mh_qk]
        attn = [torch.einsum('bkhq,bchk->bchq', wi, vi) for wi, vi in zip(attn_weights, mh_v)]
        attn = torch.cat(attn, dim=1)
        return attn, torch.cat(mh_qk, dim=1).float().detach()


class ResidualAttentionBlockANE(ResidualAttentionBlock):
    def __init__(self, n_state, n_head, cross_attention=False):
        super().__init__(n_state, n_head, cross_attention)
        self.attn = MultiHeadAttentionANE(n_state, n_head)
        self.attn_ln = LayerNormANE(n_state)
        self.cross_attn = MultiHeadAttentionANE(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNormANE(n_state) if cross_attention else None
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Conv2d(n_state, n_mlp, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(n_mlp, n_state, kernel_size=1)
        )
        self.mlp_ln = LayerNormANE(n_state)


class AudioEncoderANE(AudioEncoder):
    def __init__(self, n_mels, n_ctx, n_state, n_head, n_layer):
        super().__init__(n_mels, n_ctx, n_state, n_head, n_layer)
        self.blocks = nn.ModuleList(
            [ResidualAttentionBlockANE(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNormANE(n_state)

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        assert x.shape[1:] == self.positional_embedding.shape[::-1], "incorrect audio shape"
        x = (x + self.positional_embedding.transpose(0, 1)).to(x.dtype).unsqueeze(2)
        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)
        x = x.squeeze(2).transpose(1, 2)
        return x


class WhisperANE(Whisper):
    def __init__(self, dims):
        super().__init__(dims)
        self.encoder = AudioEncoderANE(
            self.dims.n_mels, self.dims.n_audio_ctx,
            self.dims.n_audio_state, self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self._register_load_state_dict_pre_hook(linear_to_conv2d_map)

    def forward(self, mel, tokens):
        return self.decoder(tokens, self.encoder(mel))


def convert_encoder(hparams, model, quantize=False):
    model.eval()
    input_shape = (1, hparams.n_mels, 3000)
    input_data = torch.randn(input_shape)
    traced_model = torch.jit.trace(model, input_data)
    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="logmel_data", shape=input_shape)],
        outputs=[ct.TensorType(name="output")],
        compute_units=ct.ComputeUnit.ALL,
    )
    if quantize:
        model = quantize_weights(model, nbits=16)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--encoder-only", type=bool, default=False)
    parser.add_argument("--quantize", type=bool, default=False)
    parser.add_argument("--optimize-ane", type=bool, default=False)
    args = parser.parse_args()

    whisper = load_model(args.model).cpu()
    hparams = whisper.dims
    print(hparams)

    if args.optimize_ane:
        whisperANE = WhisperANE(hparams).eval()
        whisperANE.load_state_dict(whisper.state_dict())
        encoder = whisperANE.encoder
    else:
        encoder = whisper.encoder

    encoder = convert_encoder(hparams, encoder, quantize=args.quantize)
    encoder.save(f"models/coreml-encoder-{args.model}.mlpackage")
    print("done converting")
