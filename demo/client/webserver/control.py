from fastapi import APIRouter, Depends
from demo.client.webserver.models import *
import asyncio
import struct

SERVER_HOST, SERVER_PORT = "192.168.1.221", 9999
SENSOR_HOST, SENSOR_PORT = "192.168.2.4", 10000
router = APIRouter(prefix="/control")

@router.post("/start")
async def start(st: State = Depends(get_state)):
    if st.running:
        return {"status": "already running"}

    try:
        st.server_conn = Connection(*await asyncio.open_connection(SERVER_HOST, SERVER_PORT))
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    try:
        st.sensor_conn = Connection(*await asyncio.open_connection(SENSOR_HOST, SENSOR_PORT))
    except Exception as e:
        return {"status": "error", "detail": str(e)}

    st.running = True
    st.start_event.set()
    return {"status": "started"}

@router.post("/stop")
async def stop(st: State = Depends(get_state)):
    if not st.running:
        return {"status": "not running"}
    st.running = False
    server_conn = st.server_conn
    if server_conn:
        await server_conn.close()
    st.server_conn = None
    return {"status": "stopped"}

@router.post("/slider")
async def update_slider(input: SliderInput, st: State = Depends(get_state)):
    st.interval = input.value
    return {"status": "interval updated"}

@router.post("/lr")
async def update_lr(input: LRInput, st: State = Depends(get_state)):
    c = st.server_conn
    if st.running and c:
        c.writer.write(b"chglr" + struct.pack("!f", input.value))
        return {"status": "lr sent"}
    return {"status": "not running"}
