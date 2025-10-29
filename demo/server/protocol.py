from construct import (
    Bytes,
    CString,
    Float32b,
    Rebuild,
    Struct,
    Int32ub,
    Float32l,
    Array,
    this,
    Enum,
    Flag,
    If,
    len_,
    Default,
    Optional,
)
import typing
from typing import Generic, NewType, Sequence, Union
from abc import ABC, abstractmethod
import torch
from multiprocessing.synchronize import Lock as LockType
from PIL import Image

Int = Int32ub
Shape = Struct(
    "axes" / Rebuild(Int, len_(this.dims)), "dims" / Array(this.axes, Int)
)

Data = Struct(
    "input_len" / Rebuild(Int, len_(this.input_bytes)),
    "output_len" / Rebuild(Int, len_(this.output_bytes)),
    "input_shape" / Shape,
    "output_shape" / Shape,
    "batch_size" / Int,
    "input_bytes" / Bytes(this.input_len),
    "output_bytes" / Bytes(this.output_len),
)

SDParams = Struct(
    "prompt" / CString("utf8"),
    "neg_prompt" / CString("utf8"),
    "strength" / Float32b,
    "cfg" / Float32b,
)

Input = Struct(
    "len" / Rebuild(Int, len_(this.bytes)),
    "shape" / Shape,
    "bytes" / Bytes(this.len),
)

InferLastReq = Struct(
    "decode" / Flag,
    "apply_sd" / Flag,
    "sd_params" / Optional(If(this.apply_sd, SDParams)),
)

InferReq = Struct(
    "decode" / Flag,
    "apply_sd" / Flag,
    "input" / Optional(If(this.type == 2, Input)),
    "sd_params" / Optional(If(this.apply_sd, SDParams)),
)

StatusReq = Struct(
    "message" / CString("utf8"),
)
StatusResp = Struct(
    "training" / Flag,
    "recording" / Flag,
    "reservoir_size" / Int,
    "batches_trained" / Int
)
