import asyncio
import base64
import json
import logging
import struct
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Literal, Union

import numpy as np
import torch
from PIL import Image
from fastapi import APIRouter, WebSocket
from starlette.endpoints import WebSocketEndpoint

from demo.client.webserver.models import *
from demo.server.protocol import InferLastReq, SDParams, StatusResp
from demo.server.trainer_base import TrainerState

router = APIRouter()
LATENT_SHAPE = (1, 4, 64, 64)
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ImageMessage:
    type: Literal["image"]
    channel: Literal["pred", "true"]
    img: str


@dataclass(slots=True)
class TrainerStatusMessage:
    type: Literal["trainer_status"]
    status: TrainerState


ConnectionScope = Literal["frontend", "server"]


@dataclass(slots=True)
class ConnectionStateMessage:
    type: Literal["connection_state"]
    scope: ConnectionScope
    state: Literal["connecting", "connected", "disconnected", "error"]
    detail: str | None = None


@dataclass(slots=True)
class ErrorMessage:
    type: Literal["error"]
    message: str
    detail: str | None = None


WebsocketPayload = Union[
    ImageMessage, TrainerStatusMessage, ConnectionStateMessage, ErrorMessage
]


async def send_payload(ws: WebSocket, payload: WebsocketPayload) -> None:
    await ws.send_text(json.dumps(asdict(payload)))


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


async def infer(
    server_conn: Connection, csi: np.ndarray, use_sd_post: bool
) -> torch.Tensor:
    reader, writer = server_conn.reader, server_conn.writer
    writer.write(b"infer")
    csi_bytes = csi.tobytes()
    writer.write(struct.pack("!I", len(csi_bytes)) + csi_bytes)
    await writer.drain()
    l_i = struct.unpack("!I", await reader.readexactly(4))[0]
    data = await reader.readexactly(l_i)
    return latent_to_tensor(data, use_sd_post)


async def get_jpg_csi(sensor_conn: Connection):
    sensor_conn.writer.write(b"jpg")
    await sensor_conn.writer.drain()
    l_i, l_c = struct.unpack("!II", await sensor_conn.reader.readexactly(8))
    buf = BytesIO(await sensor_conn.reader.readexactly(l_i))
    img = Image.open(buf)
    csi = np.frombuffer(
        await sensor_conn.reader.readexactly(l_c), dtype=np.float32
    )
    return img, csi


async def get_jpg(server_conn: Connection, sd_settings: Img2ImgParams):
    packet = InferLastReq.build(sd_settings.to_construct_d())
    server_conn.writer.write(b"itrai")
    server_conn.writer.write(len(packet).to_bytes(4, "big") + packet)
    await server_conn.writer.drain()

    l_i = struct.unpack("!I", await server_conn.reader.readexactly(4))[0]
    buf = BytesIO(await server_conn.reader.readexactly(l_i))
    img = Image.open(buf)
    return img


async def get_status(server_conn: Connection) -> TrainerState:
    server_conn.writer.write(b"state")
    await server_conn.writer.drain()
    l_o = struct.unpack("!I", await server_conn.reader.readexactly(4))[0]
    packet = StatusResp.parse(await server_conn.reader.readexactly(l_o))
    return TrainerState(**{k: v for k, v in packet.items() if k != "_io"})


def encode_img(img) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


@router.websocket_route("/ws")
class ImageEndpoint(WebSocketEndpoint):
    async def on_connect(self, websocket: WebSocket) -> None:
        self.st: ServerState = websocket.app.state.state
        await websocket.accept()
        await send_payload(
            websocket,
            ConnectionStateMessage(
                type="connection_state", scope="frontend", state="connected"
            ),
        )
        await self.send_task(websocket)

    async def send_task(self, ws: WebSocket):
        while True:
            if self.st.shutdown_event.is_set():
                return

            await self.st.start_event.wait()

            if self.st.shutdown_event.is_set():
                return

            await send_payload(
                ws,
                ConnectionStateMessage(
                    type="connection_state", scope="server", state="connecting"
                ),
            )

            try:
                await self._stream_predictions(ws)
            finally:
                self.st.start_event.clear()

    async def on_disconnect(
        self, websocket: WebSocket, close_code: int
    ) -> None:
        pass

    async def _stream_predictions(self, ws: WebSocket) -> None:
        state = self.st
        conn = state.server_conn

        if conn is None:
            await send_payload(
                ws,
                ConnectionStateMessage(
                    type="connection_state",
                    scope="server",
                    state="error",
                    detail="Server connection unavailable.",
                ),
            )
            return

        await send_payload(
            ws,
            ConnectionStateMessage(
                type="connection_state", scope="server", state="connected"
            ),
        )

        final_state = ConnectionStateMessage(
            type="connection_state", scope="server", state="disconnected"
        )

        try:
            while state.running and state.server_conn is conn:
                try:
                    pil_img = await get_jpg(conn, state.sd_settings)
                    status = await get_status(conn)
                except asyncio.CancelledError:
                    raise
                except (
                    ConnectionError,
                    asyncio.IncompleteReadError,
                    OSError,
                ) as exc:
                    logger.warning("Connection dropped: %s", exc)
                    final_state = ConnectionStateMessage(
                        type="connection_state",
                        scope="server",
                        state="disconnected",
                        detail=str(exc),
                    )
                    await self._handle_connection_drop()
                    break
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("Unexpected streaming failure")
                    await send_payload(
                        ws,
                        ErrorMessage(
                            type="error",
                            message="Unexpected streaming failure.",
                            detail=str(exc),
                        ),
                    )
                    final_state = ConnectionStateMessage(
                        type="connection_state",
                        scope="server",
                        state="error",
                        detail="Unexpected streaming failure.",
                    )
                    await self._handle_connection_drop()
                    break
                else:
                    await send_payload(
                        ws,
                        ImageMessage(
                            type="image",
                            channel="pred",
                            img=encode_img(pil_img),
                        ),
                    )
                    await send_payload(
                        ws,
                        TrainerStatusMessage(
                            type="trainer_status",
                            status=status,
                        ),
                    )
                await asyncio.sleep(state.interval)
        finally:
            await send_payload(ws, final_state)

    async def _handle_connection_drop(self) -> None:
        state = self.st
        state.running = False

        conn = state.server_conn
        if conn is not None:
            try:
                await conn.close()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Ignoring server connection close error: %s", exc)
        state.server_conn = None

        sensor_conn = state.sensor_conn
        if sensor_conn is not None:
            try:
                await sensor_conn.close()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Ignoring sensor connection close error: %s", exc)
        state.sensor_conn = None
