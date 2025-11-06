from fastapi import APIRouter, Depends
from demo.client.webserver.models import *
import asyncio
import struct

SENSOR_HOST, SENSOR_PORT = "localhost", 10000
router = APIRouter(prefix="/control")


@router.post("/start")
async def start(st: ServerState = Depends(get_state)):
    print("started")
    if st.running:
        return {"status": "already running"}
    try:
        st.server_conn = Connection(
            *await asyncio.open_connection(st.server_addr, 9000)
        )
    except Exception as e:
        return {"status": "error", "detail": str(e)}

    print("server_conn opened")
    # try:
    #     st.sensor_conn = Connection(
    #         *await asyncio.open_connection(SENSOR_HOST, SENSOR_PORT, ssl_handshake_timeout=2)
    #     )
    # except Exception as e:
    #     return {"status": "error", "detail": str(e)}
    # print("sensor_conn opened")
    st.running = True
    st.start_event.set()
    return {"status": "started"}


@router.post("/stop")
async def stop(st: ServerState = Depends(get_state)):
    if not st.running:
        return {"status": "not running"}
    st.running = False
    server_conn = st.server_conn
    if server_conn:
        await server_conn.close()
    st.server_conn = None
    return {"status": "stopped"}


@router.post("/slider")
async def update_slider(
    input: SliderInput, st: ServerState = Depends(get_state)
):
    st.interval = input.value
    return {"status": "interval updated"}


@router.post("/lr")
async def update_lr(input: LRInput, st: ServerState = Depends(get_state)):
    c = st.server_conn
    if st.running and c:
        async with c.lock:
            c.writer.write(b"chglr" + struct.pack("!f", input.value))
            await c.writer.drain()
        return {"status": "lr sent"}
    return {"status": "not running"}


@router.post("/msg")
async def send_msg(input: MsgInput, st: ServerState = Depends(get_state)):
    c = st.server_conn
    if c:
        async with c.lock:
            c.writer.write(
                b"messa"
                + struct.pack("!I", len(input.value))
                + input.value.encode("utf-8")
            )
            await c.writer.drain()
        print("sent")
        return {"status": "msg sent"}
    return {"status": "not running"}


@router.post("/sdsettings")
async def set_img2img(p: Img2ImgParams, st: ServerState = Depends(get_state)):
    st.sd_settings = p
    return {"status": "ok"}
