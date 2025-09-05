from fastapi import Request
from pydantic import BaseModel, PositiveFloat, confloat
from asyncio import StreamReader, StreamWriter, Event
from typing import Optional, Annotated
from diffusers import AutoencoderTiny, StableDiffusionImg2ImgPipeline
import torch

class Connection:
    def __init__(self, reader: StreamReader , writer: StreamWriter):
        self.reader = reader
        self.writer = writer

    async def close(self):
        self.writer.close()
        await self.writer.wait_closed()


class DummyImageProcessor:
    def __init__(self, *args, **kwargs):
        pass
    def preprocess(self, image, *args, **kwargs):
        return image
    def postprocess(self, image, *args, **kwargs):
        return image



class Img2ImgParams(BaseModel):
    enabled: bool
    prompt: str
    negativePrompt: str
    strength: Annotated[float, confloat(gt=0, le=1.0)]
    cfg: float

class ServerState:
    def __init__(self):
        self.server_conn: Optional[Connection] = None
        self.sensor_conn: Optional[Connection] = None
        self.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.half).to("cuda")
        self.sd = StableDiffusionImg2ImgPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.half).to("cuda")
        print(self.sd.dtype)

        self.sd.safety_checker = None
        self.sd.image_processor = DummyImageProcessor()
        self.sd.vae = self.vae

        self.sd_settings = Img2ImgParams(
            enabled=False,
            prompt="",
            negativePrompt="",
            strength=0.55,
            cfg=7
        )

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
