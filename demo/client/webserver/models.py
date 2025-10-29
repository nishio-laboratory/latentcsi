import logging
from asyncio import Event, StreamReader, StreamWriter
from typing import Annotated, Any, Optional
from fastapi import Request
from pydantic import BaseModel, PositiveFloat, confloat

from diffusers import AutoencoderTiny, StableDiffusionImg2ImgPipeline
from demo.client.utils import DummyImageProcessor
import torch

from demo.server.protocol import InferLastReq, SDParams

logger = logging.getLogger(__name__)


class Connection:
    def __init__(self, reader: StreamReader, writer: StreamWriter):
        self.reader = reader
        self.writer = writer

    async def close(self):
        self.writer.close()
        try:
            await self.writer.wait_closed()
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            logger.debug("Ignoring transport close error: %s", exc)


class Img2ImgParams(BaseModel):
    enabled: bool
    prompt: str
    negativePrompt: str
    strength: Annotated[float, confloat(gt=0, le=1.0)]
    cfg: float

    @classmethod
    def from_construct(cls, s: Any) -> 'Img2ImgParams':
        return cls(
            enabled = True,
            prompt = s.prompt,
            negativePrompt = s.neg_prompt,
            strength = s.strength,
            cfg = s.cfg
        )
    def to_construct_d(self) -> dict:
        if not self.enabled or self.prompt == "":
            return {"decode": True, "apply_sd": False}
        return {"decode": True,
           "apply_sd": True,
           "sd_params": {
               "prompt": self.prompt,
               "neg_prompt": self.negativePrompt,
               "strength": self.strength,
               "cfg": self.cfg
           }}



class ServerState:
    def __init__(self):
        self.server_conn: Optional[Connection] = None
        self.sensor_conn: Optional[Connection] = None
        self.sd_settings = Img2ImgParams(
            enabled=False, prompt="", negativePrompt="", strength=0.55, cfg=7
        )
        self.interval: float = 0.33
        self.use_sd_post: bool = False
        self.running: bool = False
        self.start_event = Event()
        self.shutdown_event = Event()

class SliderInput(BaseModel):
    value: Annotated[float, confloat(gt=0, le=1.0)]

class LRInput(BaseModel):
    value: PositiveFloat

class MsgInput(BaseModel):
    value: str


def get_state(request: Request) -> ServerState:
    return request.app.state.state
