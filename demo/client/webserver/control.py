from fastapi import APIRouter, Depends
from demo.client.webserver.models import *
import asyncio
import struct

SERVER_HOST, SERVER_PORT = "192.168.1.221", 9999
SENSOR_HOST, SENSOR_PORT = "192.168.1.32", 10000
router = APIRouter(prefix="/control")

@router.post("/start")
async def start(st: ServerState = Depends(get_state)):
    print("started")
    if st.running:
        return {"status": "already running"}
    try:
        st.server_conn = Connection(*await asyncio.open_connection(SERVER_HOST, SERVER_PORT))
    except Exception as e:
        return {"status": "error", "detail": str(e)}

    print("server_conn opened")
    try:
        st.sensor_conn = Connection(*await asyncio.open_connection(SENSOR_HOST, SENSOR_PORT))
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    print("sensor_conn opened")
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
async def update_slider(input: SliderInput, st: ServerState = Depends(get_state)):
    st.interval = input.value
    return {"status": "interval updated"}

@router.post("/lr")
async def update_lr(input: LRInput, st: ServerState = Depends(get_state)):
    c = st.server_conn
    if st.running and c:
        c.writer.write(b"chglr" + struct.pack("!f", input.value))
        return {"status": "lr sent"}
    return {"status": "not running"}



@router.post("/sdsettings")
async def set_img2img(p: Img2ImgParams, st: ServerState = Depends(get_state)):
    st.sd_settings = p
    return {"status": "ok"}
