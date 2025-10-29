import multiprocessing as mp
import torch
from src.other.types import *
from demo.server.trainer_base import TrainerBase
from tqdm import tqdm


class TrainerSave(TrainerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_loop()

    def main_loop(self):
        samples = 1000
        csi_buf = torch.zeros((samples, 16, 1992))
        lat_buf = torch.zeros((samples, 16, 4, 64, 64))
        for i in tqdm(range(samples)):
            batch = self.data_queue.get()
            csi_buf[i] = batch.csi
            lat_buf[i] = batch.lat
        torch.save(csi_buf, "data_csi.pt")
        torch.save(lat_buf, "data_lat.pt")
