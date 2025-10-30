import uvicorn
import argparse
from demo.client.webserver.ws import *
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from demo.client.webserver.models import *
from demo.client.webserver.ws import router as ws_router
from demo.client.webserver.control import router as ctrl_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.state = ServerState()
    yield
    app.state.state.shutdown_event.set()


app = FastAPI(lifespan=lifespan)
app.include_router(ws_router)
app.include_router(ctrl_router)
app.mount(
    "/",
    StaticFiles(directory="demo/client/webserver/frontend/dist"),
    name="static",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
