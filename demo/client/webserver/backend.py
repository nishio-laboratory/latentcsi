import asyncio
from contextlib import asynccontextmanager
import struct
import base64
from io import BytesIO
from typing import Annotated, Optional

import numpy as np
import torch
from diffusers.models.autoencoders import AutoencoderTiny
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, PositiveFloat, confloat, BaseModel

SERVER_HOST, SERVER_PORT = "192.168.1.221", 9999
LATENT_SHAPE = (1, 4, 64, 64)

class State:
    def __init__(self):
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.model = AutoencoderTiny.from_pretrained("madebyollin/taesd").to("cuda")
        self.interval: float = 0.33
        self.use_sd_post: bool = False
        self.running: bool = False
        self.start_event = asyncio.Event()


def state(app: FastAPI) -> State:
    return app.state.state


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.state = State()
    yield

app = FastAPI(lifespan=lifespan)
app.mount(
    "/static", StaticFiles(directory="demo/client/webserver/static"), name="static"
)
clients: set[WebSocket] = set()

class SliderInput(BaseModel):
    value: Annotated[float, confloat(gt=0, le=1.0)]

class LRInput(BaseModel):
    value: PositiveFloat

async def read_latent(reader: asyncio.StreamReader) -> torch.Tensor:
    length = struct.unpack("!I", await reader.readexactly(4))[0]
    data = await reader.readexactly(length)
    arr = np.frombuffer(data, dtype=np.float32)
    if state(app).use_sd_post:
        arr *= 0.18215
    tensor = torch.tensor(arr).reshape(LATENT_SHAPE).to("cuda")
    return tensor


def encode_img(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            await state(app).start_event.wait()
            while state(app).running and state(app).reader:
                state(app).writer.write(b"ilast")
                latent = await read_latent(state(app).reader)
                with torch.no_grad():
                    img_tensor = state(app).model.decode(latent).sample.squeeze().cpu()
                if state(app).use_sd_post:
                    img_tensor = (img_tensor + 1) / 2
                pil_img = to_pil_image(img_tensor.clip(0, 1))
                await ws.send_text(encode_img(pil_img))
                await asyncio.sleep(state(app).interval)
            state(app).start_event.clear()
    finally:
        clients.remove(ws)

@app.post("/control/start")
async def start():
    if state(app).running:
        return {"status": "already running"}
    try:
        reader, writer = await asyncio.open_connection(SERVER_HOST, SERVER_PORT)
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    state(app).reader = reader
    state(app).writer = writer
    state(app).running = True
    state(app).start_event.set()
    return {"status": "started"}

@app.post("/control/stop")
async def stop():
    if not state(app).running:
        return {"status": "not running"}
    state(app).running = False
    w = state(app).writer
    if w:
        w.close()
        await w.wait_closed()
    state(app).reader = None
    state(app).writer = None
    return {"status": "stopped"}

@app.post("/control/slider")
async def update_slider(input: SliderInput):
    state(app).interval = input.value
    return {"status": "interval updated"}

@app.post("/control/lr")
async def update_lr(input: LRInput):
    w = state(app).writer
    if state(app).running and w:
        w.write(b"chglr" + struct.pack("!f", input.value))
        await w.drain()
        return {"status": "lr sent"}
    return {"status": "not running"}

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("demo/client/webserver/static/index.html") as f:
        return f.read()
