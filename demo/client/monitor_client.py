import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import socket
import threading
import time
import struct
import torch
from torchvision.transforms.functional import to_pil_image
from diffusers.models.autoencoders import AutoencoderTiny

taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd")

HOST, PORT = "192.168.1.221", 9999
LATENT_SHAPE = (1, 4, 64, 64)
LATENT_SIZE = int(np.prod(LATENT_SHAPE) * 4)


class ImageClientApp:
    def __init__(self, root, use_sd_post: bool = False):
        self.root = root
        self.root.title("live viewer")

        self.interval = 0.33
        self.use_sd_post = use_sd_post
        self.running = False
        self.sock = None  # persistent socket

        # Canvas for image display
        self.canvas = tk.Canvas(root, width=512, height=512)
        self.canvas.pack()

        # Slider to adjust update interval
        self.slider = ttk.Scale(
            root,
            from_=0.1,
            to=2.0,
            value=self.interval,
            command=self.update_interval,
            orient="horizontal",
            length=300,
        )
        self.slider.pack()
        self.label = ttk.Label(
            root, text=f"Update interval: {self.interval:.2f}s"
        )
        self.label.pack()

        # Entry for learning rate
        self.lr_label = ttk.Label(root, text="Learning rate:")
        self.lr_label.pack()
        self.lr_entry = ttk.Entry(root)
        self.lr_entry.pack()
        self.btn_lr = ttk.Button(root, text="Set LR", command=self.send_lr)
        self.btn_lr.pack(pady=5)

        # Start/Stop buttons
        self.btn_start = ttk.Button(root, text="Start", command=self.start)
        self.btn_start.pack(side=tk.LEFT, padx=10)
        self.btn_stop = ttk.Button(root, text="Stop", command=self.stop)
        self.btn_stop.pack(side=tk.LEFT, padx=10)

        self.thread = None
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_interval(self, val):
        self.interval = float(val)
        self.label.config(text=f"Update interval: {self.interval:.2f}s")

    def start(self):
        if self.running:
            return
        try:
            self.sock = socket.create_connection((HOST, PORT), timeout=5)
            self.sock.settimeout(2.0)
        except Exception as e:
            print("Could not connect to server:", e)
            self.sock = None
            return

        self.running = True
        self.thread = threading.Thread(target=self.fetch_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        if self.thread:
            self.thread.join(timeout=1)
            self.thread = None

    def send_lr(self):
        val = self.lr_entry.get()
        try:
            lr = float(val)
        except ValueError:
            print(f"Invalid learning rate: {val}")
            return
        if lr <= 0:
            print(f"Learning rate must be positive: {lr}")
            return
        if not self.sock:
            print("Not connected to server")
            return
        try:
            msg = b"chglr" + struct.pack("!f", lr)
            self.sock.sendall(msg)
            print(f"Sent learning rate: {lr}")
        except Exception as e:
            print("Error sending learning rate:", e)

    def fetch_loop(self):
        while self.running:
            try:
                self.sock.sendall(b"ilast")
                data = b""
                while len(data) < LATENT_SIZE:
                    chunk = self.sock.recv(LATENT_SIZE - len(data))
                    if not chunk:
                        raise ConnectionError("Socket closed by server")
                    data += chunk

                latent = np.frombuffer(data, dtype=np.float32).reshape(
                    LATENT_SHAPE
                )
                latent_tensor = torch.tensor(latent.copy())
                if self.use_sd_post:
                    latent_tensor *= 0.18215
                with torch.no_grad():
                    img = (
                        taesd.decode(latent_tensor)
                        .sample.detach()
                        .cpu()
                        .squeeze()
                    )
                if self.use_sd_post:
                    img = (img + 1) / 2
                pil_img = to_pil_image(img.clip(0, 1))
                self.root.after(0, self.update_image, pil_img)

            except Exception as e:
                print("Socket error, stopping fetch loop:", e)
                self.running = False
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = None
                break

            time.sleep(self.interval)

    def update_image(self, pil_img):
        pil_img = pil_img.resize((512, 512))
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    def on_close(self):
        self.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClientApp(root)
    root.mainloop()
