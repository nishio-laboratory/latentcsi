import argparse
from pathlib import Path
import subprocess
import torch
from diffusers import AutoencoderTiny
from typing import Tuple, List, Dict

HF_MODEL: str = "hf/taesd"
ONNX_DIR: Path = Path("./onnx")
TRT_DIR: Path = Path("./trt")
TRTEXEC: str = "/usr/src/tensorrt/bin/trtexec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TensorRT engines for TAESD VAE"
    )
    parser.add_argument(
        "--min-batch",
        type=int,
        default=1,
        help="Minimum batch size",
    )
    parser.add_argument(
        "--opt-batch",
        type=int,
        default=4,
        help="Optimal batch size",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=8,
        help="Maximum batch size",
    )
    return parser.parse_args()


def export_to_onnx(
    ae: AutoencoderTiny,
    name: str,
    input_tensor: torch.Tensor,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Dict[str, Dict[int, str]],
) -> Path:
    onnx_path: Path = ONNX_DIR / f"taesd_{name}.onnx"
    torch.onnx.export(
        getattr(ae, name),
        input_tensor,
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        dynamic_axes=dynamic_axes,
    )
    print(f"Exported {name} to ONNX at {onnx_path}")
    return onnx_path


def build_trt_engine(
    onnx_path: Path,
    name: str,
    input_dims: Tuple[int, int, int],
    min_batch: int,
    opt_batch: int,
    max_batch: int,
) -> Path:
    engine_path: Path = TRT_DIR / f"taesd_{name}.trt"
    dims_str: str = f"{input_dims[0]}x{input_dims[1]}x{input_dims[2]}"
    min_shapes: str = f"input:{min_batch}x{dims_str}"
    opt_shapes: str = f"input:{opt_batch}x{dims_str}"
    max_shapes: str = f"input:{max_batch}x{dims_str}"

    cmd: List[str] = [
        TRTEXEC,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes={min_shapes}",
        f"--optShapes={opt_shapes}",
        f"--maxShapes={max_shapes}",
        "--fp16",
    ]
    subprocess.run(cmd, check=True)
    print(
        f"Built TensorRT engine for {name} at {engine_path} ({min_batch}/{opt_batch}/{max_batch})"
    )
    return engine_path


def main() -> None:
    args = parse_args()
    ONNX_DIR.mkdir(exist_ok=True)
    TRT_DIR.mkdir(exist_ok=True)

    ae: AutoencoderTiny = (
        AutoencoderTiny.from_pretrained(HF_MODEL).eval().to("cuda")
    )
    enc_in: torch.Tensor = torch.zeros((1, 3, 512, 512), device="cuda")
    dec_in: torch.Tensor = torch.zeros((1, 4, 64, 64), device="cuda")

    enc_onnx: Path = export_to_onnx(
        ae=ae,
        name="encoder",
        input_tensor=enc_in,
        input_names=["input"],
        output_names=["latent"],
        dynamic_axes={"input": {0: "batch_size"}, "latent": {0: "batch_size"}},
    )
    dec_onnx: Path = export_to_onnx(
        ae=ae,
        name="decoder",
        input_tensor=dec_in,
        input_names=["latent"],
        output_names=["reconstruction"],
        dynamic_axes={
            "latent": {0: "batch_size"},
            "reconstruction": {0: "batch_size"},
        },
    )

    build_trt_engine(
        enc_onnx,
        "encoder",
        (3, 512, 512),
        args.min_batch,
        args.opt_batch,
        args.max_batch,
    )
    build_trt_engine(
        dec_onnx,
        "decoder",
        (4, 64, 64),
        args.min_batch,
        args.opt_batch,
        args.max_batch,
    )


if __name__ == "__main__":
    main()
