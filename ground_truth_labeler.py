from threading import Event
from generate_video import generate_multicam_video
from aux_tools import str2bool, str_to_datetime, date_range, time_to_float, format_axis_as_timedelta, ExperimentTraverser, EXPERIMENT_DATETIME_STR_FORMAT
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from queue import Queue
import os
import cv2
import numpy as np
import h5py
import json
import argparse

# Import UI
try:
    import Tkinter as tk
    import tkFont
    import ttk
    import tkMessageBox as messagebox
except ImportError:  # Python 3
    import tkinter as tk
    import tkinter.font as tkFont
    import tkinter.ttk as ttk
    from tkinter import messagebox
from MultiColumnListbox import MultiColumnListbox


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


class VideoAndWeightHandler:
    TIME_INCREMENT = timedelta(seconds=0.1)  # How much to shift the time offset cameras-weights from keyboard input (ASDW)
    FRAME_INCREMENT = 8  # How many frames to skip forward/backward on keyboard input (arrow keys)
    LEFT_RIGHT_MULTIPLIER = 10  # How much larger the skip is when using left-right (A-D) vs up-down (or W-S)

    def __init__(self, experiment_base_folder, cb_event_start_or_end, user_wants_to_exit, update_xaxis=False):
        self.cb_event_start_or_end = cb_event_start_or_end
        self.user_wants_to_exit = user_wants_to_exit
        self.update_xaxis = update_xaxis  # For faster plot update, set this to False and the weight's xaxis will be static (-0:03 -0:02 ... 0:03)
        self.n = -1  # Frame number
        self.is_paused = False
        self.refresh_weight = True
        self.do_skip_frames = False
        self.t_lims = 3  # How many seconds of weight to show on either side of curr_t
        self.initial_scale = 0.5  # Rescale video_img before converting to Tkinter image (~3X faster to render)
        self.keys_pressed = Queue()
        self.video_canvas = None
        self.video_tk_img = None
        self.weight_canvas = None
        self.bg_cache = None

        # Load video info
        video_in_filename = generate_multicam_video(experiment_base_folder)
        self.video_in = cv2.VideoCapture(video_in_filename)
        with h5py.File(os.path.splitext(video_in_filename)[0] + ".h5", 'r') as h5_cam:
            self.t_cam = np.array(list(date_range(str_to_datetime(h5_cam.attrs['t_start']), str_to_datetime(h5_cam.attrs['t_end']), timedelta(seconds=1.0/h5_cam.attrs['fps']))))
        self.video_dims = np.array([self.video_in.get(cv2.CAP_PROP_FRAME_HEIGHT), self.video_in.get(cv2.CAP_PROP_FRAME_WIDTH)]).astype(int)
        self.video_initial_dims = (self.initial_scale * self.video_dims).astype(int)
        self.weight_dims = np.array([self.video_initial_dims[0], 350]).astype(int)

        # Read all weight sensors for the full experiment duration at once
        t_experiment_start = experiment_base_folder.rsplit('/', 1)[-1]  # Last folder in the path should indicate time at which experiment started
        with h5py.File(os.path.join(experiment_base_folder, "weights_{}.h5".format(t_experiment_start)), 'r') as h5_weights:
            self.weight_t = np.array([str_to_datetime(t) for t in h5_weights['t_str']])
            weight_data = h5_weights['w'][:]
            w = np.sum(weight_data, axis=1)
        t_w = time_to_float(self.weight_t, self.weight_t[0])

        # Manually align weight and cam timestamps (not synced because OSX and Linux use different NTP servers)
        self.weight_to_cam_t_offset = self.weight_t[0] + timedelta(seconds=13)  # Initialize the offset to ~13s (empirical)

        # Set up matplotlib figure
        self.fig = plt.figure(figsize=self.weight_dims[::-1]/100.0)
        num_subplots = len(w)
        ax = self.fig.subplots(num_subplots, 1, sharex=True, squeeze=False)
        self.curr_t_lines = []
        for i in range(num_subplots):
            shelf_i = num_subplots - (i+1)  # Shelf 1 is at the bottom
            # Plot weight and a vertical line at currT. Draw invisible: we'll copy the canvas bgnd, then make it visible
            ax[i,0].plot(t_w, w[shelf_i])
            self.curr_t_lines.append(ax[i,0].axvline(0, linestyle='--', color='black', linewidth=1))
            ax[i,0].set_title('Shelf {}'.format(shelf_i+1))
            ax[i,0].set_xlim(-self.t_lims, self.t_lims)
            format_axis_as_timedelta(ax[i,0].xaxis)

            if self.update_xaxis:
                ax[i,0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)

        # Render the figure and save background so updating the plot can be much faster (using blit instead of draw)
        self.update_bg_cache()

        # Allocate memory space for a video frame and a downsampled copy
        self.video_img = np.zeros((self.video_dims[0], self.video_dims[1], 3), dtype=np.uint8)

    def update_bg_cache(self, resize_event=None):
        if resize_event is not None:
            self.weight_canvas.resize(resize_event)  # Forward event to figure canvas so it resizes the figure
        axes = self.fig.get_axes()

        def set_visibility(is_visible):
            for i, ax in enumerate(axes):
                for l in ax.lines:
                    l.set_visible(is_visible)
                    if is_visible: ax.draw_artist(l)  # Will need to rerender

                # update_xaxis=False means weight's xaxis will be static (always show: -0:03 -0:02 ... 0:03)
                # update_xaxis=True means we'll rerender the xaxis on every replot -> Need to hide the labels before copying the canvas bgnd
                if self.update_xaxis:
                    is_last = (i == len(axes)-1)
                    ax.tick_params(axis='x', which='both', bottom=is_visible, labelbottom=is_last and is_visible)
                    if is_visible: ax.draw_artist(ax.xaxis)

        set_visibility(False)  # Make axes and lines invisible
        self.fig.canvas.draw()  # Rerender full figure
        self.bg_cache = self.fig.canvas.copy_from_bbox(self.fig.bbox)  # Copy the whole canvas
        set_visibility(True)  # Make everything visible again
        self.fig.canvas.blit()  # Rerender only necessary parts


    def update(self):
        # Update video frame (if needed)
        if not self.is_paused or self.do_skip_frames:
            # Grab next frame
            self.n += 1
            ok = self.video_in.read(self.video_img[:, :self.video_dims[1], :])
            assert ok, "Couldn't read frame {}!".format(self.n)
            print("Read frame {} out of {} frames ({:6.2f}%)".format(self.n+1, len(self.t_cam), 100.0*(self.n+1)/len(self.t_cam)))

            # Render the frame
            self.video_canvas.update_image(self.video_img)

        # Update weight plot (if needed)
        if self.refresh_weight:
            # Update current time and redraw whatever needed
            curr_t = (self.t_cam[self.n]-self.weight_to_cam_t_offset).total_seconds()
            self.fig.canvas.restore_region(self.bg_cache)  # We'll render on top of our cached bgnd (contains subplot frames, shelf number [title], ylabels, etc)
            for l in self.curr_t_lines: l.set_xdata(curr_t)  # Update time cursor (dashed black lines)
            for ax in self.fig.get_axes():
                ax.set_xlim(curr_t-self.t_lims, curr_t+self.t_lims)  # Update xlims to be centered on current time
                for l in ax.lines: ax.draw_artist(l)  # Redraw all lines
                if self.update_xaxis:  # Redraw xlabels if needed
                    ax.draw_artist(ax.xaxis)

            # Refresh weight plot (using blitting for a ~5X speedup vs canvas.draw())
            self.fig.canvas.blit()

        # Process key presses
        self.handle_kb_input()
        if self.do_skip_frames:
            self.video_in.set(cv2.CAP_PROP_POS_FRAMES, self.n)
            self.n = int(self.video_in.get(cv2.CAP_PROP_POS_FRAMES))  # Don't let it go over the length of the video

    def handle_kb_input(self):
        self.do_skip_frames = False
        self.refresh_weight = False

        while not self.keys_pressed.empty():
            key_info = self.keys_pressed.get()
            k = key_info.keysym.lower()

            if k == 'left':  # Left arrow (at least on my Mac)
                self.n -= self.LEFT_RIGHT_MULTIPLIER*self.FRAME_INCREMENT
                self.do_skip_frames = True
                self.refresh_weight = True
            elif k == 'right':  # Right arrow
                self.n += self.LEFT_RIGHT_MULTIPLIER*self.FRAME_INCREMENT
                self.do_skip_frames = True
                self.refresh_weight = True
            elif k == 'up':  # Up arrow
                self.n += self.FRAME_INCREMENT
                self.do_skip_frames = True
                self.refresh_weight = True
            elif k == 'down':  # Down arrow
                self.n -= self.FRAME_INCREMENT
                self.do_skip_frames = True
                self.refresh_weight = True
            elif k == 'a':
                self.weight_to_cam_t_offset -= self.LEFT_RIGHT_MULTIPLIER*self.TIME_INCREMENT
                self.refresh_weight = True
            elif k == 'd':
                self.weight_to_cam_t_offset += self.LEFT_RIGHT_MULTIPLIER*self.TIME_INCREMENT
                self.refresh_weight = True
            elif k == 'w':
                self.weight_to_cam_t_offset -= self.TIME_INCREMENT
                self.refresh_weight = True
            elif k == 's':
                self.weight_to_cam_t_offset += self.TIME_INCREMENT
                self.refresh_weight = True
            elif k == 'b':
                self.cb_event_start_or_end(True, self.t_cam[self.n])
            elif k == 'n':
                self.cb_event_start_or_end(False, self.t_cam[self.n])
            elif k == 'space':
                self.is_paused = not self.is_paused
            elif k == 'escape':  # Don't exit on unrecognized keys if labeling ground truth
                print('Esc pressed, exiting!')
                self.user_wants_to_exit.set()
        self.refresh_weight = self.refresh_weight or not self.is_paused


class GroundTruthLabelerWindow(tk.Tk):
    VIDEO_AND_WEIGHT_UPDATE_PERIOD = 10  # msec
    WIN_PAD = 10
    GRID_PAD = 3  # 3px between consecutive items in a hor/vert grid (e.g. between video feed and weight plot)

    def __init__(self, experiment_base_folder):
        super(GroundTruthLabelerWindow, self).__init__()
        self.weight_to_cam_t_offset = None
        self.t_offset_float = 0
        self.user_wants_to_exit = Event()

        self.video_and_weight = VideoAndWeightHandler(experiment_base_folder, self.on_set_event_time_start_or_end, self.user_wants_to_exit)
        video_canvas_size = self.video_and_weight.video_initial_dims
        weight_canvas_size = self.video_and_weight.weight_dims

        # Load product info
        with open("Dataset/product_info.json", 'r') as f:
            self.product_info = json.load(f)['products']
        options = tuple((p['id'], p['name']) for p in self.product_info)
        column_headers = ("Time start", "Time end", "Pickup?", "Item ID", "Item name", "Quantity")
        column_widths = (186, 186, 50, 48, -1, 55)

        # Setup ui
        self.title("Ground truth labeler")
        win_size = np.array((video_canvas_size[1] + weight_canvas_size[1] + 2*self.WIN_PAD + 2*self.GRID_PAD, video_canvas_size[0]+300))
        win_offs = (np.array((self.winfo_screenwidth(), self.winfo_screenheight())) - win_size)/2
        self.geometry("{s[0]}x{s[1]}+{o[0]}+{o[1]}".format(s=win_size.astype(int), o=win_offs.astype(int)))
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.ui_container = tk.Frame(self)
        self.ui_container.pack(fill='both', expand=True, padx=self.WIN_PAD, pady=self.WIN_PAD)

        # Variables
        self.quantity = tk.IntVar(self, 1)
        self.selected_product = tk.Variable(self, options[0])
        self.is_pickup = tk.BooleanVar(self, True)
        self.t_start = None
        self.t_end = None
        self.events = []

        # Widgets
        self.video_and_weight_container = tk.Frame(self)
        self.video_and_weight_container.grid(column=0, columnspan=6, row=0, sticky='nesw', ipady=self.GRID_PAD/2, in_=self.ui_container)
        self.video_canvas = ResizableImageCanvas(master=self, width=video_canvas_size[1], height=video_canvas_size[0], highlightthickness=0)
        self.video_canvas.grid(column=0, row=0, sticky='nesw', in_=self.video_and_weight_container)
        self.weight_canvas = FigureCanvasTkAgg(self.video_and_weight.fig, master=self)
        self.weight_canvas.draw()
        self.weight_canvas.get_tk_widget().grid(column=1, row=0, sticky='ns', padx=(self.WIN_PAD, 0), in_=self.video_and_weight_container)
        self.weight_canvas.get_tk_widget().bind("<Configure>", self.video_and_weight.update_bg_cache)
        self.video_and_weight.video_canvas = self.video_canvas
        self.video_and_weight.weight_canvas = self.weight_canvas
        self.lst_events = MultiColumnListbox(column_headers, master=self)
        for i,w in enumerate(column_widths):
            if w > 0:
                self.lst_events.tree.column(i, width=w, stretch=False)
        self.lst_events.tree.grid(column=0, columnspan=6, row=1, pady=self.GRID_PAD, sticky='nesw', in_=self.ui_container)
        num_quantity = tk.Spinbox(self, from_=1, to_=5, width=1, borderwidth=0, textvariable=self.quantity)
        num_quantity.grid(column=0, row=2, rowspan=2, in_=self.ui_container)
        drp_product = tk.OptionMenu(self, self.selected_product, *options)
        drp_product.grid(column=1, row=2, rowspan=2, sticky='ew', ipadx=10, in_=self.ui_container)
        opt_pickup = tk.Radiobutton(self, text="Pick up", variable=self.is_pickup, value=True)
        opt_pickup.grid(column=2, row=2, sticky='ew', ipadx=10, in_=self.ui_container)
        opt_pickup = tk.Radiobutton(self, text="Put back", variable=self.is_pickup, value=False)
        opt_pickup.grid(column=2, row=3, sticky='ew', ipadx=0, in_=self.ui_container)
        tk.Label(self, text="Start:").grid(column=3, row=2, sticky='nsew', in_=self.ui_container)
        tk.Label(self, text="End:").grid(column=3, row=3, sticky='nsew', in_=self.ui_container)
        self.txt_t_start = tk.Text(self, state=tk.DISABLED, height=1, width=26)
        self.txt_t_start.grid(column=4, row=2, sticky='nsew', in_=self.ui_container)
        self.txt_t_end = tk.Text(self, state=tk.DISABLED, height=1, width=26)
        self.txt_t_end.grid(column=4, row=3, sticky='nsew', in_=self.ui_container)
        self._update_time()  # Initialize their text
        btn_add_event = tk.Button(self, text="Add event", command=self.add_event)
        btn_add_event.grid(column=5, row=2, rowspan=2, in_=self.ui_container)

        # Event handling
        self.bind('<KeyPress>', self.video_and_weight.keys_pressed.put)
        self.lst_events.tree.bind('<KeyPress>', self.remove_event)

        # Make grids expandable on window resize
        self.ui_container.grid_rowconfigure(0, weight=1)
        self.ui_container.grid_rowconfigure(1, weight=1)
        self.ui_container.grid_columnconfigure(1, weight=1)
        self.video_and_weight_container.grid_columnconfigure(0, weight=1)
        self.video_and_weight_container.grid_rowconfigure(0, weight=1)

        # Load the first image
        self.update_canvas()

    def run(self):
        # Run main loop
        self.mainloop()

        # Save final offset values
        self.weight_to_cam_t_offset = self.video_and_weight.weight_to_cam_t_offset
        self.t_offset_float = (self.video_and_weight.weight_t[0]-self.weight_to_cam_t_offset).total_seconds()

    def on_set_event_time_start_or_end(self, is_start, t):
        if is_start:
            self.t_start = t
        else:
            self.t_end = t
        self._update_time()

    def update_canvas(self):
        # Update video frame and weight plot
        self.video_and_weight.update()

        # Check if user pressed Escape
        if self.user_wants_to_exit.is_set():
            self.on_closing()
        else:
            self.after(self.VIDEO_AND_WEIGHT_UPDATE_PERIOD, self.update_canvas)

    def on_closing(self):
        # Save state of the events list before destroying
        events_info = (self.lst_events.tree.item(child, values=None) for child in self.lst_events.tree.get_children(''))
        self.events = [{
            "t_start": event[0],
            "t_end": event[1],
            "is_pickup": str2bool(event[2]),
            "item_id": int(event[3]),
            "item_name": event[4],
            "quantity": int(event[5])
        } for event in events_info]

        if messagebox.askokcancel("Exit?", "Are you sure you're done annotating this experiment's ground truth?\nWe've registered {} event{}".format(len(self.events), '' if len(self.events)==1 else 's')):
            self.destroy()

    def add_event(self):
        prod_id, prod_name = self.selected_product.get()
        new_item = (self.t_start, self.t_end, self.is_pickup.get(), prod_id, prod_name, self.quantity.get())
        print("Adding event: {}".format(new_item))
        self.lst_events.add_item(new_item)

        # Reset state
        self.quantity.set(1)
        self.t_start = None
        self.t_end = None
        self._update_time()

    def remove_event(self, k):
        if k.keysym == 'BackSpace' or k.keysym == 'Delete':
            selected_items = self.lst_events.tree.selection()
            if len(selected_items) > 0 and messagebox.askokcancel("Are you sure?", "Are you sure you want to remove {} item{}?".format(len(selected_items), 's' if len(selected_items)>1 else '')):
                self.lst_events.tree.delete(*selected_items)

    def _set_time(self, is_t_start):
        if is_t_start:
            txt_box = self.txt_t_start
            text = self.t_start if self.t_start is not None else "Press 'b' to set t_start"
        else:
            txt_box = self.txt_t_end
            text = self.t_end if self.t_end is not None else "Press 'n' to set t_end"

        # Update text (need to set state to normal, change text, then disable the widget again)
        txt_box.config(state='normal')
        txt_box.delete(1.0, 'end')
        txt_box.insert('end', text)
        txt_box.config(state='disabled')

    def _update_time(self):
        self._set_time(True)
        self._set_time(False)


class GroundTruthLabeler(ExperimentTraverser):
    def process_subfolder(self, f):
        experiment_folder = os.path.join(self.main_folder, f)
        ground_truth_file = os.path.join(experiment_folder, "ground_truth.json")
        if os.path.exists(ground_truth_file):
            print("Video already annotated!! Skipping (delete '{}' and run this tool again if you want to overwrite)".format(ground_truth_file))
            return

        # Open ground truth labeling windows
        gt_labeler = GroundTruthLabelerWindow(experiment_folder)
        gt_labeler.run()
        print("Generate_video finished! Weight-camera time offset manually set as {} ({}s wrt weight's own timestamps)".format(gt_labeler.weight_to_cam_t_offset, gt_labeler.t_offset_float))
        annotated_events = gt_labeler.events
        print("Received annotated events: {}".format(annotated_events))
        with open(ground_truth_file, 'w') as f_gt:
            json.dump({
                'ground_truth': annotated_events,
                'weight_to_cam_t_offset': str(gt_labeler.weight_to_cam_t_offset),
                'weight_to_cam_t_offset_float': gt_labeler.t_offset_float,
            }, f_gt, indent=2)
        print("Ground truth annotation saved as '{}'!".format(ground_truth_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", default="Dataset/Evaluation", help="Folder containing the experiment to visualize")
    parser.add_argument("-s", "--start-datetime", default="", help="Only preprocess experiments collected later than this datetime (format: {}; empty for no limit)".format(EXPERIMENT_DATETIME_STR_FORMAT))
    parser.add_argument("-e", "--end-datetime", default="", help="Only preprocess experiments collected before this datetime (format: {}; empty for no limit)".format(EXPERIMENT_DATETIME_STR_FORMAT))
    args = parser.parse_args()

    t_start = datetime.strptime(args.start_datetime, EXPERIMENT_DATETIME_STR_FORMAT) if len(args.start_datetime) > 0 else datetime.min
    t_end = datetime.strptime(args.end_datetime, EXPERIMENT_DATETIME_STR_FORMAT) if len(args.end_datetime) > 0 else datetime.max

    GroundTruthLabeler(args.folder, t_start, t_end).run()
