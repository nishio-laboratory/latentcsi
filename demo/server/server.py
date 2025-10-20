from __future__ import annotations
from typing import ByteString, Optional
import struct
import time
from diffusers import AutoencoderTiny
import torch
import asyncio
import torch.multiprocessing as mp
from demo.server.decoder import decode_latent_to_image
from demo.server.protocol import Data, InferLastReq
from demo.server.trainers.basic import TrainerLastReplay
from demo.server.trainers.signal_stop import TrainerStoppable
from src.inference.utils import pil_image_to_bytes
from src.other.types import *
from demo.server.trainer_base import TrainerBase, LockedTensor


class TrainingServerBase:
    def __init__(self, host: str, port: int, trainer: type[TrainerBase]):
        self.host = host
        self.port = port
        self.ctx = mp.get_context("spawn")
        self.data_queue: mp.Queue[Batch] = self.ctx.Queue(maxsize=100)
        self.message_queue: mp.Queue[Message] = self.ctx.Queue(maxsize=1000)
        self.out_tensor: LockedTensor = LockedTensor(
            torch.zeros(1, 4, 64, 64), self.ctx.Lock()
        )
        self.ae_tiny = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
            torch.device("cuda:1")
        )
        self.trainer = trainer

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        addr = writer.get_extra_info("peername")
        print(f"Client connected: {addr}")
        t, i = 0, 0
        inf_elapsed_times = []
        try:
            while True:
                header = await reader.readexactly(5)
                if header == b"train":
                    req_len = struct.unpack("!I", await reader.readexactly(4))[
                        0
                    ]
                    packet = Data.parse(await reader.readexactly(req_len))
                    inputs = BatchCSI(
                        torch.frombuffer(
                            bytearray(packet.input_bytes), dtype=torch.float32
                        ).view(packet.input_shape.dims)
                    )

                    outputs = BatchTrueLatent(
                        torch.frombuffer(
                            bytearray(packet.output_bytes), dtype=torch.float32
                        ).view(packet.output_shape.dims)
                    )
                    # img = decode_latent_to_image(outputs[0].unsqueeze(0), self.ae_tiny, scale=False)
                    # img.save("test.png")
                    if self.data_queue.full():
                        self.data_queue.get()
                    self.data_queue.put_nowait(Batch(inputs, outputs))

                elif header == b"itrai":
                    req_len = struct.unpack("!I", await reader.readexactly(4))[
                        0
                    ]
                    req = InferLastReq.parse(await reader.readexactly(req_len))
                    now = time.time()
                    latent_tensor = PredLatent(
                        (await asyncio.to_thread(self.out_tensor.get_copy)).to(
                            1
                        )
                    )
                    img = await asyncio.to_thread(
                        decode_latent_to_image,
                        latent_tensor,
                        self.ae_tiny,
                        scale=False,
                    )
                    img_bytes = pil_image_to_bytes(img)
                    writer.write(len(img_bytes).to_bytes(4, "big") + img_bytes)
                    await writer.drain()
                    elapsed = time.time() - now
                    inf_elapsed_times.append(elapsed)
                    i += 1
                    if i % 50 == 0:
                        print(
                            f"avg time to compute: {sum(inf_elapsed_times) / len(inf_elapsed_times)}"
                        )
                        inf_elapsed_times = []
                elif header == b"messa":
                    req_len = struct.unpack(
                        "!I", await reader.readexactly(4)
                    )[0]
                    msg_str = (await reader.readexactly(req_len)).decode(
                        "utf-8"
                    )
                    print(f"message recvd: {msg_str}")
                    self.message_queue.put_nowait(check_msg(msg_str))
                else:
                    out = await self.dispatch(header, reader)
                    if out:
                        writer.write(out)
                        await writer.drain()
        except asyncio.IncompleteReadError:
            pass
        finally:
            writer.close()
            await writer.wait_closed()
            print(f"Client disconnected: {addr}")

    async def dispatch(
        self, header: ByteString, reader: asyncio.StreamReader
    ) -> Optional[ByteString]:
        pass

    async def start(self):
        server = await asyncio.start_server(
            self.handle_client, self.host, self.port
        )
        print(f"TCP server listening on {self.host}:{self.port}")

        train_process = self.ctx.Process(
            target=self.trainer,
            args=(
                self.data_queue,
                self.message_queue,
                self.out_tensor,
                torch.device("cuda:0"),
            ),
        )
        train_process.start()

        async with server:
            await server.serve_forever()


async def main():
    srv = TrainingServerBase(
        host="0.0.0.0", port=9999, trainer=TrainerStoppable
    )
    await srv.start()


if __name__ == "__main__":
    asyncio.run(main())
