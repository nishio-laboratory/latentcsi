# (setq python-shell-interpreter "/home/esrh/csi_to_image/activate_docker.sh")
# (setq python-shell-intepreter-args "-p")
from typing import cast
import gc
from src.encoder.base import MLP, CSIAutoencoderBase
from src.encoder.data_utils import CSIDataset
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput
import transformers
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import lightning as L
import argparse
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# ***


class CSIAutoencoder(CSIAutoencoderBase):
    def __init__(self, layer_sizes, data_path: Path, train_seg=False, lr=5e-4):
        super().__init__(layer_sizes, lr)
        self.train_seg = train_seg
        self.segformer = (
            transformers.SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512"
            )
        )
        self.vae = cast(
            AutoencoderKL,
            AutoencoderKL().from_pretrained(
                data_path.parents[1] / "sd/sd-v1-5",
                subfolder="vae",
            ),
        )
        self.vae.enable_tiling()

        for param in self.segformer.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False

        self.vae.eval()
        self.segformer.eval()

    def training_step(self, batch, batch_idx):
        inputs, targets, segmaps = batch
        # print(inputs.shape, targets.shape, segmaps.shape)
        outputs = self.model(inputs)
        if not self.train_seg:
            loss = torch.nn.functional.mse_loss(outputs, targets)
            self.log("tr_pixel_loss", loss)
            return loss
        else:
            decoder_output = cast(
                DecoderOutput, self.vae.decode(outputs)
            ).sample
            decoded = ((decoder_output + 1) / 2).clamp(0, 1)
            logits = self.segformer(decoded).logits
            loss = torch.nn.functional.cross_entropy(logits, segmaps)
            del decoded
            del logits
            gc.collect()
            torch.cuda.empty_cache()
            self.log("seg_loss", loss)
            return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        outputs = self.model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        self.log(
            "val_loss", loss, sync_dist=True, prog_bar=True, on_epoch=True
        )
        return loss

    def ckpt_name(self):
        return (
            "mlp_seg_"
            + "-".join(map(str, self.model.layer_sizes))
            + "{val_loss}"
        )


# ***
def main():
    if "H100" in torch.cuda.get_device_name(0):
        torch.set_float32_matmul_precision("medium")

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-e", "--max-epochs", default=1, type=int)
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    args = parser.parse_args()

    data_path = Path(args.path)
    dataset = CSIDataset(data_path, aux_data=["seg_map"])
    train, val, test = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
    train = DataLoader(train, num_workers=15, batch_size=args.batch_size)
    val = DataLoader(val, num_workers=15, batch_size=args.batch_size)
    print("Loaded data")

    model = CSIAutoencoder(
        [1992, 2000, 1000, 500, 1000, 2000, 16384], data_path
    )

    def make_trainer():
        return L.Trainer(
            max_epochs=args.max_epochs,
            logger=CSVLogger(save_dir=data_path / "logs"),
            strategy="ddp_find_unused_parameters_true",
            precision=16,
            callbacks=[EarlyStopping("tr_pixel_loss", patience=15)],
        )

    trainer = make_trainer()
    trainer.fit(model, train, val)

    model.train_seg = True
    trainer_2 = L.Trainer(
        max_epochs=50,
        logger=CSVLogger(save_dir=data_path / "logs"),
        strategy="ddp_find_unused_parameters_true",
        precision=16,
        callbacks=[
            ModelCheckpoint(
                dirpath=data_path / "ckpts", filename=model.ckpt_name()
            )
        ]
        if args.save
        else [],
    )
    trainer_2.fit(model, train, val)


if __name__ == "__main__":
    main()
