from demo.server.protocol import StatusResp
from demo.server.trainer_base import TrainerBase
from demo.server.trainers.basic import BatchReservoir, CNNDecoderTrainable
from src.other.types import *


class TrainerStoppable(TrainerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = CNNDecoderTrainable(
            input_dim=1992,
            base_channels=128,
            lr=1e-4,
        ).to(self.device)
        self.batch_reservoir = BatchReservoir(
            2000, replace_rate=1, uniform=False
        )
        print("Train process ready!")
        self.state.started = True
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
                print("Set recording to true")
            case "stop_rec":
                self.state.recording = False
                print("Set recording to false")
            case "start_train":
                self.state.training = True
                print("Set training to true")
            case "stop_train":
                self.state.training = False
                print("Set training to false")
            case "reset":
                raise NotImplementedError()
            case ("chglr", new_lr):
                raise NotImplementedError()

    def main_loop(self):
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
                    pred, loss = self.train_new()
                else:
                    pred, loss = self.infer_last()
                self.latest_pred.update(pred)

            # print(
            #     f"Batch {self.state.batches_trained} trained! Loss: {loss:.6f}. Qsize: {self.data_queue.qsize()}"
            # )
