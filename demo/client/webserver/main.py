import uvicorn
from demo.client.webserver.backend import app
from demo.client.webserver.ws import *

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8123)
