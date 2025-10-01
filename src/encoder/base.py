from pathlib import Path
import time
import lightning as L
from pytorch_lightning.utilities import rank_zero_only


class TrainingTimerCallback(L.Callback):
    def on_train_start(self, trainer, pl_module):
        pl_module._train_start_time = time.time()

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        total_time = time.time() - pl_module._train_start_time
        total_epochs = trainer.current_epoch + 1
        avg_time = total_time / total_epochs if total_epochs > 0 else 0

        if trainer.logger and trainer.logger.log_dir:
            out_path = Path(trainer.logger.log_dir) / "training_time.txt"

            with open(out_path, "w") as f:
                f.write(f"Total training time (s): {total_time:.3f}\n")
                f.write(f"Average time per epoch (s): {avg_time:.3f}\n")
                f.write(f"Number of epochs: {total_epochs}")

            print(f"Training timing info written to {out_path}")
        else:
            raise Exception("log dir not specified")
