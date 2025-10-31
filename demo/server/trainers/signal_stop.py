import time
from demo.server.protocol import StatusResp
from demo.server.trainer_base import TrainerBase, TrainerState
from demo.server.trainers.basic import BatchReservoir, CNNDecoderTrainable
from src.other.types import *

class TrainerStoppable(TrainerBase):
    def effectful_init(self):
        self.model = CNNDecoderTrainable(
            input_dim=1992,
            base_channels=128,
            lr=1e-4,
        ).to(self.device)
        self.batch_reservoir = BatchReservoir(
            2000, replace_rate=1, uniform=False
        )
        self.state = TrainerState()
        print("Train process started!")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.effectful_init()
        self.main_loop()

    def train_new(self) -> tuple[PredLatent, float]:
        batch = self.data_queue.get()
        if self.batch_reservoir.size() == 1:
            self.batch_reservoir.buffer[0] = batch
        if self.state.recording or self.batch_reservoir.empty():
            self.batch_reservoir.add(batch)
            self.state.reservoir_size = self.batch_reservoir.size()

        inputs = BatchCSI(batch.csi.to(self.device))
        outputs = BatchTrueLatent(batch.lat.to(self.device))
        loss, preds = self.model.train_step(inputs, outputs)

        self.state.batches_trained += 1
        return (
            PredLatent(preds[-1].unsqueeze(0).detach().cpu().contiguous()),
            float(loss.item()),
        )

    def train_replay(self) -> float:
        batch = self.batch_reservoir.pick()
        inputs = BatchCSI(batch.csi.to(self.device))
        outputs = BatchTrueLatent(batch.lat.to(self.device))
        loss, _ = self.model.train_step(inputs, outputs)
        return float(loss.item())

    def infer_last(self) -> tuple[PredLatent, float]:
        batch = self.data_queue.get()
        csi = CSI(batch.csi[-1].unsqueeze(0).to(self.device))
        out = CSI(batch.lat[-1].unsqueeze(0).to(self.device))
        with self.model.as_eval(), torch.no_grad():
            p = self.model(csi)
            loss = torch.nn.functional.mse_loss(p, out).item()
        return p.cpu(), loss

    def dispatch(self, msg: Message):
        match msg:
            case "start_rec":
                self.state.recording = True
            case "stop_rec":
                self.state.recording = False
            case "start_train":
                self.state.training = True
            case "stop_train":
                self.state.training = False
            case "reset":
                self.effectful_init()
            case ("chglr", new_lr):
                for param_group in self.model.optimizer.param_groups:
                    param_group["lr"] = new_lr

    def main_loop(self):
        elapsed_new = []
        while True:
            self.shared.state = self.state
            if not self.message_queue.empty():
                msg = self.message_queue.get()
                self.dispatch(msg)

            if self.data_queue.empty():
                if not self.state.training or self.batch_reservoir.empty():
                    continue
                loss = self.train_replay()
            else:
                if self.state.training:
                    now = time.time()
                    pred, loss = self.train_new()
                    elapsed_new.append(time.time() - now)
                    if self.state.batches_trained % 10 == 0:
                        print(f"Train_new over 10 batches: {sum(elapsed_new)/10}")
                        elapsed_new = []
                else:
                    pred, loss = self.infer_last()
                self.latest_pred.update(pred)

            # print(
            #     f"Batch {self.state.batches_trained} trained! Loss: {loss:.6f}. Qsize: {self.data_queue.qsize()}"
            # )
