# (setq python-shell-interpreter "/home/esrh/csi_to_image/activate_docker.sh")
# (setq python-shell-intepreter-args "")

import torch
import torch.distributed as dist
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import utils


def run_inference(rank, world_size, photos, formatter, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    feature_extractor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    model.to(rank)
    print(f"RANK {rank} loaded model")

    @utils.chunk_process
    def compute(img):
        inputs = feature_extractor(images=img, return_tensors="pt").to(rank)
        return model(**inputs).last_hidden_state.squeeze()

    out = compute(
        photos,
        rank,
        world_size,
        (256, photos[0].height // 32, photos[0].width // 32)
    )

    torch.save(out, formatter(args.path, rank))
    dist.destroy_process_group()


if __name__ == "__main__":
    utils.run_dist(
        run_inference,
        "targets_segmented"
    )
