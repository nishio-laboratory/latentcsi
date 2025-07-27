from diffusers.utils.dummy_torch_and_transformers_objects import StableDiffusionImg2ImgPipeline
from fastapi import Request
from pydantic import BaseModel, PositiveFloat, confloat
from asyncio import StreamReader, StreamWriter, Event
from typing import Optional, Annotated
from diffusers import AutoencoderTiny

class Connection:
    def __init__(self, reader: StreamReader , writer: StreamWriter):
        self.reader = reader
        self.writer = writer

    async def close(self):
        self.writer.close()
        await self.writer.wait_closed()


class ServerState:
    def __init__(self):
        self.server_conn: Optional[Connection] = None
        self.sensor_conn: Optional[Connection] = None
        self.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to("cuda")
        self.sd = StableDiffusionImg2ImgPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
        self.interval: float = 0.33
        self.use_sd_post: bool = False
        self.running: bool = False
        self.start_event = Event()



class SliderInput(BaseModel):
    value: Annotated[float, confloat(gt=0, le=1.0)]

class LRInput(BaseModel):
    value: PositiveFloat


def get_state(request: Request) -> ServerState:
    return request.app.state.state
