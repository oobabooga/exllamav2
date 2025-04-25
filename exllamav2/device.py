from __future__ import annotations

def _torch_device(idx):
    if idx == -1: return "cpu"
    return f"cuda:{idx}"

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2

from exllamav2.architecture import RopeStyle
from exllamav2 import rope

import torch
import math


global_streams = {}


def set_device_streams():
    global global_streams
    for(k, v) in global_streams.items():
        with torch.cuda.device(torch.device(k)):
            torch.cuda.set_device(torch.device(k))
            torch.cuda.set_stream(v)


def get_device_stream(index: int):
    global global_streams
    return global_streams.get(index)


class ExLlamaV2DeviceContext:

    model: ExLlamaV2
    device_idx: int
    ready: bool

    scratch_bytes: int
    scratch_idx: int

    sin: list[torch.Tensor] | None
    cos: list[torch.Tensor] | None
    local_sin: list[torch.Tensor] | None
    local_cos: list[torch.Tensor] | None

    scratch: torch.Tensor | None

    stream: torch.cuda.Stream

    def __init__(
        self,
        model: ExLlamaV2,
        device_idx: int,
        scratch_bytes: int,
        archparams = None
    ):
        self.model = model
        self.device_idx = device_idx
        self.ready = False
        self.scratch = None
        self.scratch_bytes = scratch_bytes
        self.scratch_idx = 0
        self.archparams = archparams or model.config.arch.lm

        # Create streams (only one per device)

        if device_idx not in global_streams:
            s = torch.cuda.Stream(torch.device(device_idx), -100)
            global_streams[device_idx] = s

        self.stream = global_streams[device_idx]

        xx = 0


    def prepare(self, scratch):

        self.prepare_sincos()

        if scratch:
            self.scratch = torch.empty((self.scratch_bytes // 2,), dtype = torch.half, device = _torch_device(self.device_idx))

        self.ready = True


    def drop(self):

        self.scratch = None
        self.sin = None
        self.cos = None
        self.ready = False


    def free(self):

        self.drop()
        self.scratch_bytes = 1


    def begin_scratch_alloc(self):

        self.scratch_idx = 0


    def get_scratch_slice(self, size_bytes):

        if self.scratch is None: self.prepare(True)

        size_bytes = ((size_bytes + 63) // 64) * 64
        size_half = size_bytes // 2
        scratch_slice = self.scratch.narrow(0, self.scratch_idx, size_half)
        self.scratch_idx += size_half
        return scratch_slice


    def prepare_sincos(self):

        device = _torch_device(self.device_idx)
        cfg = self.model.config

        thetas = [cfg.rotary_embedding_base]
        scales = [cfg.scale_pos_emb]
        if cfg.rotary_embedding_base_alt:
            thetas.append(cfg.rotary_embedding_base_alt)
            scales.append(cfg.scale_pos_emb_alt)

        self.sin = []
        self.cos = []

        for theta, lscale in zip(thetas, scales):

            if self.archparams.rope_style == RopeStyle.NONE:
                sin = torch.zeros((1,), device = device, dtype = torch.half)
                cos = sin
                self.sin.append(sin)
                self.cos.append(cos)
                break

            # RoPE params

            inv_freq, scaling_factor = rope.get_rope_params(device, cfg, theta)

            # Common

            scale = lscale or 1.0
            t = torch.arange(cfg.max_seq_len, device = device, dtype = torch.float32)
            if cfg.pos_id_index != 0:
                t += cfg.pos_id_index
            if scale != 1.0: t /= scale

            freqs = torch.einsum("i,j->ij", t, inv_freq)
            if self.archparams.rope_style == RopeStyle.NEOX:
                emb = torch.cat((freqs, freqs), dim=-1)
            elif self.archparams.rope_style == RopeStyle.GPTJ:
                emb = torch.repeat_interleave(freqs, 2, dim=-1)
            else:
                raise ValueError()

            sin = emb.sin()[None, None, :, :]
            cos = emb.cos()[None, None, :, :]
            if scaling_factor != 1.0:
                sin *= scaling_factor
                cos *= scaling_factor
            sin = sin.half()
            cos = cos.half()

            self.sin.append(sin)
            self.cos.append(cos)