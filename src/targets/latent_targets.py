# (setq python-shell-interpreter "/home/esrh/csi_to_image/activate_docker.sh")
# (setq python-shell-intepreter-args "")
from typing import cast
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
import utils
import torch
import argparse
import torch.distributed as dist
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from PIL import Image
from PIL.Image import Image as ImageType


def preprocess_resize(im: ImageType, left_offset=34) -> ImageType:
    return im.resize((640, 512), resample=Image.Resampling.BICUBIC).crop(
        (left_offset, 0, 512 + left_offset, 512)
    )


def run_inference(rank, world_size, photos, formatter, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    vae = AutoencoderKL().from_pretrained(
        args.path.parents[1] / "sd/sd-v1-5",
        subfolder="vae",
        use_safetensors=True,
    )
    vae = cast(AutoencoderKL, vae)
    image_processor = VaeImageProcessor(vae_scale_factor=8)
    vae.to(rank)
    gen = torch.Generator(rank)
    print(f"RANK {rank} loaded model")

    @utils.chunk_process
    def compute(img):
        img = Image.fromarray(img)
        if len(photos[0]) != 512:
            img = preprocess_resize(img)
        img = image_processor.preprocess(img).to(device=rank)
        if args.distribution:
            return vae._encode(img)
        else:
            encoder_output = cast(AutoencoderKLOutput, vae.encode(img))
            return encoder_output.latent_dist.sample(gen)

    out = compute(
        photos,
        rank,
        world_size,
        [
            8 if args.distribution else 4,
            len(photos[0]) // 8,
            len(photos[0][0]) // 8,
        ],
    ).to("cpu")

    torch.save(out, formatter(args.path, rank))
    dist.destroy_process_group()


if __name__ == "__main__":

    def save_func(data, args):
        if args.distribution:
            torch.save(data, args.path / "targets" / "targets_dists.pt")
        else:
            torch.save(data, args.path / "targets" / "targets_latents.pt")

    parser = argparse.ArgumentParser(
        prog="Generate latent targets from photos"
    )
    parser.add_argument(
        "-d",
        "--distribution",
        action="store_true",
        help="output vae distribution instead of sampled latents",
    )

    utils.run_dist(
        run_inference,
        lambda args: "targets_dists"
        if args.distribution
        else "targets_latents",
        parser,
    )
