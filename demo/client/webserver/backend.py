import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
import numpy as np
from demo.client.webserver.models import *
from demo.client.webserver.ws import router as ws_router
from demo.client.webserver.ws import clients
from demo.client.webserver.control import router as ctrl_router
from starlette.websockets import WebSocketState
import signal

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.state = ServerState()
    yield
    app.state.state.shutdown_event.set()

app = FastAPI(lifespan=lifespan)

app.include_router(ws_router)
app.include_router(ctrl_router)
app.mount("/", StaticFiles(directory="demo/client/webserver/frontend/dist"), name="static")
