from demo.client.webserver.models import *
from demo.client.webserver import control
from fastapi.staticfiles import StaticFiles
import struct
import json
import time
import numpy as np
from PIL import Image
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
import base64
from io import BytesIO
import asyncio
import torch
from torchvision.transforms.functional import to_pil_image

router = APIRouter()
clients: set[WebSocket] = set()

LATENT_SHAPE = (1, 4, 64, 64)

def latent_to_tensor(latent_bytes, use_sd_post):
    arr = np.frombuffer(latent_bytes, dtype=np.float32)
    if use_sd_post:
        arr *= 0.18215
    return torch.tensor(arr).reshape((1, 4, 64, 64)).to("cuda")

async def ilast(server_conn: Connection, use_sd_post: bool) -> torch.Tensor:
    reader, writer = server_conn.reader, server_conn.writer
    writer.write(b"ilast")
    l_i = struct.unpack("!I", await reader.readexactly(4))[0]
    data = await reader.readexactly(l_i)
    return latent_to_tensor(data, use_sd_post)

async def infer(server_conn: Connection, csi: np.ndarray, use_sd_post: bool) -> torch.Tensor:
    reader, writer = server_conn.reader, server_conn.writer
    writer.write(b"infer")
    csi_bytes = csi.tobytes()
    writer.write(struct.pack("!I", len(csi_bytes)) + csi_bytes)
    await writer.drain()
    l_i = struct.unpack("!I", await reader.readexactly(4))[0]
    data = await reader.readexactly(l_i)
    return latent_to_tensor(data, use_sd_post)


async def get_jpg_csi(sensor_conn: Connection):
    print("querying sensor...")
    sensor_conn.writer.write(b"jpg")
    await sensor_conn.writer.drain()
    l_i, l_c = struct.unpack("!II", await sensor_conn.reader.readexactly(8))
    buf = BytesIO(await sensor_conn.reader.readexactly(l_i))
    img = Image.open(buf)
    csi = np.frombuffer(await sensor_conn.reader.readexactly(l_c), dtype=np.float32)
    return img, csi


def encode_img(img) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    st: ServerState = ws.app.state.state
    await ws.accept()
    clients.add(ws)
    i = 0
    try:
        while True:
            await st.start_event.wait()
            while st.running and st.server_conn and st.sensor_conn:
                jpg, csi = await get_jpg_csi(st.sensor_conn)
                pred = await infer(st.server_conn, csi, st.use_sd_post)
                pred = pred.half().cuda()

                with torch.no_grad():
                    if st.sd_settings.enabled and st.sd_settings.prompt != "":
                        img_tensor = st.sd(
                            prompt=st.sd_settings.prompt,
                            image=pred,
                            strength=st.sd_settings.strength,
                            guidance_scale=st.sd_settings.cfg
                        ).images[0].cpu()
                    else:
                        img_tensor = st.vae.decode(pred).sample.squeeze().cpu()

                if st.use_sd_post:
                    img_tensor = (img_tensor + 1) / 2
                pil_img = to_pil_image(img_tensor.clip(0, 1))

                # pil_img.save(f"example_images/predicted_{i}.png")
                # jpg.save(f"example_images/real_{i}.png")
                i += 1

                try:
                    await ws.send_text(
                        json.dumps(
                            {
                                "stream": "pred",
                                "img": encode_img(pil_img)
                            }
                        )
                    )
                    await ws.send_text(
                        json.dumps(
                            {
                                "stream": "true",
                                "img": encode_img(jpg)
                            }
                        )
                    )
                except WebSocketDisconnect:
                    await control.stop(st)
                    return
                await asyncio.sleep(st.interval)
            st.start_event.clear()
    finally:
        await control.stop(st)
        clients.remove(ws)
