import numpy as np
import cv2
from PIL import Image, ImageTk

# Import UI
try:
    import Tkinter as tk
except ImportError:  # Python 3
    import tkinter as tk


class ResizableImageCanvas(tk.Canvas):
    def __init__(self, preserve_aspect_ratio=True, *args, **kwargs):
        super(ResizableImageCanvas, self).__init__(*args, **kwargs)
        self.preserve_aspect_ratio = preserve_aspect_ratio

        self.bind("<Configure>", self.on_resize)
        self.img = None
        self.tk_img = None
        self.tk_img_size = (0, 0)
        self.canvas_img = None
        self.canvas_size = np.array((self.winfo_width(), self.winfo_height()), dtype=float)

    def _fit(self, dims):  # Fit an image inside the canvas and return its dimensions
        dims = np.array(dims, dtype=float)

        if self.preserve_aspect_ratio:
            scale_wh = self.canvas_size/dims
            scale = scale_wh.min()
            return scale * dims
        else:
            return self.canvas_size

    def update_image(self, cv2_img):
        # Resize cv2_img and create a PIL.Image from it
        img_dims = np.array(cv2_img.shape[1::-1])
        cv2_img_resized = cv2.resize(cv2_img, tuple(self._fit(img_dims).astype(int)))
        cv2.cvtColor(cv2_img_resized, cv2.COLOR_BGR2RGB, cv2_img_resized)
        self.img = Image.fromarray(cv2_img_resized)

        # If the rescaled image has different size than self.tk_img, create a new self.tk_img with correct size (pastes the image too), otherwise just update the image
        if np.any(self.img.size != self.tk_img_size):
            self.resize_canvas_img(self.img.size)
        else:
            self.tk_img.paste(self.img)

    def resize_canvas_img(self, img_size):
        # Delete old image if needed
        if self.canvas_img:
            self.delete(self.canvas_img)

        self.tk_img_size = np.array(img_size, dtype=int)
        self.tk_img = ImageTk.PhotoImage(master=self, width=self.tk_img_size[0], height=self.tk_img_size[1], image=self.img.resize(self.tk_img_size) if self.img is not None else "RGB")
        self.canvas_img = self.create_image(self.canvas_size[0]//2, self.canvas_size[1]//2, image=self.tk_img)

    def on_resize(self, event):
        self.canvas_size = np.array((event.width, event.height), dtype=float)  # Update new canvas size
        img_size = self._fit(self.img.size) if self.img is not None else self.canvas_size  # Find the image size that fits inside
        self.resize_canvas_img(img_size)  # Resize image