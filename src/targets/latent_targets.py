# (setq python-shell-interpreter "/home/esrh/csi_to_image/activate_docker.sh")
# (setq python-shell-intepreter-args "")

import utils
import torch
import argparse
import torch.distributed as dist
import diffusers
from diffusers.image_processor import VaeImageProcessor


def run_inference(rank, world_size, photos, formatter, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    vae = diffusers.AutoencoderKL().from_pretrained(
        "/data/sd/sd-v1-5",
        subfolder="vae",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    image_processor = VaeImageProcessor(vae_scale_factor=8)
    vae.to(rank)
    gen = torch.Generator(rank)
    print(f"RANK {rank} loaded model")

    @utils.chunk_process
    def compute(img):
        img = image_processor.preprocess(img).to(device=rank, dtype=torch.half)
        if args.distribution:
            return vae._encode(img)
        else:
            return vae.encode(img).latent_dist.sample(gen)

    out = compute(
        photos,
        rank,
        world_size,
        (
            8 if args.distribution else 4,
            photos[0].height // 8,
            photos[0].width // 8,
        ),
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
        lambda args: "targets_dists" if args.distribution else "targets_latents",
        parser
    )
