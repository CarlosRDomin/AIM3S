import matplotlib
matplotlib.use('TkAgg')

from threading import Event
from generate_video import generate_multicam_video
from preprocess_experiments import DATETIME_FORMAT
from aux_tools import list_subfolders, str2bool, str_to_datetime, date_range, time_to_float, format_axis_as_timedelta, plt_fig_to_cv2_img
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
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


class VideoAndWeightHandler:
    TIME_INCREMENT = timedelta(seconds=0.1)  # How much to shift the time offset cameras-weights from keyboard input (ASDW)
    FRAME_INCREMENT = 8  # How many frames to skip forward/backward on keyboard input (arrow keys)
    LEFT_RIGHT_MULTIPLIER = 10  # How much larger the skip is when using left-right (A-D) vs up-down (or W-S)

    def __init__(self, experiment_base_folder, cb_event_start_or_end, new_video_and_weight_frame_ready, user_wants_to_exit):
        self.cb_event_start_or_end = cb_event_start_or_end
        self.new_video_and_weight_frame_ready = new_video_and_weight_frame_ready
        self.user_wants_to_exit = user_wants_to_exit
        self.n = -1  # Frame number
        self.is_paused = False
        self.refresh_weight = True
        self.do_skip_frames = False
        self.t_lims = 3  # How many seconds of weight to show on either side of curr_t
        self.out_scale = 0.5  # Rescale rgb_data before converting to Tkinter image (~3X faster to render)
        self.keys_pressed = Queue()
        self.tk_img = None

        # Load video info
        video_in_filename = generate_multicam_video(experiment_base_folder)
        self.video_in = cv2.VideoCapture(video_in_filename)
        with h5py.File(os.path.splitext(video_in_filename)[0] + ".h5", 'r') as h5_cam:
            self.t_cam = np.array(list(date_range(str_to_datetime(h5_cam.attrs['t_start']), str_to_datetime(h5_cam.attrs['t_end']), timedelta(seconds=1.0/h5_cam.attrs['fps']))))
        self.video_dims = np.array([self.video_in.get(cv2.CAP_PROP_FRAME_HEIGHT), self.video_in.get(cv2.CAP_PROP_FRAME_WIDTH)]).astype(int)

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
        self.fig = plt.figure(figsize=(3.5,5))
        num_subplots = len(w)
        ax = self.fig.subplots(num_subplots, 1, sharex=True, squeeze=False)
        self.curr_t_lines = []
        for i in range(num_subplots):
            shelf_i = num_subplots - (i+1)  # Shelf 1 is at the bottom
            ax[i,0].plot(t_w, w[shelf_i])
            ax[i,0].set_title('Shelf {}'.format(shelf_i+1))
            ax[i,0].set_ylabel('Weight (g)')
            format_axis_as_timedelta(ax[i,0].xaxis)
            self.curr_t_lines.append(ax[i,0].axvline(0, linestyle='--', color='black', linewidth=1))
        # Render the figure once to get its dimensions
        self.fig.canvas.draw()
        weight_img = plt_fig_to_cv2_img(self.fig)
        self.weight_scale = float(self.video_dims[0])/weight_img.shape[0]
        self.weight_fig_dims = (self.weight_scale * np.array(weight_img.shape[:2])).astype(int)
        # Allocate extra space when the figure is going to be plotted to the right of the video
        self.rgb_data = np.zeros((self.video_dims[0], self.video_dims[1]+self.weight_fig_dims[1], 3), dtype=np.uint8)
        self.downsampled_img = np.zeros((int(self.out_scale*self.rgb_data.shape[0]), int(self.out_scale*self.rgb_data.shape[1]), 3), dtype=np.uint8) if self.out_scale != 1 else None
        self.downsampled_dims = self.downsampled_img.shape[1::-1]

    def grab_next_frame(self):
        t = [datetime.now()]
        if not self.is_paused or self.do_skip_frames:
            self.n += 1
            ok = self.video_in.read(self.rgb_data[:, :self.video_dims[1], :])
            assert ok, "Couldn't read frame {}!".format(self.n)
            print("Read frame {} out of {} frames ({:6.2f}%)".format(self.n+1, len(self.t_cam), 100.0*(self.n+1)/len(self.t_cam)))

        if self.refresh_weight:
            # Update current time
            curr_t = (self.t_cam[self.n]-self.weight_to_cam_t_offset).total_seconds()
            self.fig.get_axes()[-1].set_xlim(curr_t-self.t_lims, curr_t+self.t_lims)

            # Update weight plot and convert to image
            for l in self.curr_t_lines: l.set_xdata(curr_t)
            t.append(datetime.now())
            self.fig.canvas.draw()
            t.append(datetime.now())
            weight_img = plt_fig_to_cv2_img(self.fig)

            # Place the weight plot to the right of the camera frames
            cv2.resize(weight_img, None, self.rgb_data[:,self.video_dims[1]:,:], fx=self.weight_scale, fy=self.weight_scale)

        if not self.is_paused or self.refresh_weight:
            if self.downsampled_img is not None:
                img = cv2.resize(self.rgb_data, self.downsampled_dims, self.downsampled_img)
            else:
                img = self.rgb_data
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.new_video_and_weight_frame_ready.set()

        # Process key presses
        self.handle_kb_input()
        if self.do_skip_frames:
            self.video_in.set(cv2.CAP_PROP_POS_FRAMES, self.n)
            self.n = int(self.video_in.get(cv2.CAP_PROP_POS_FRAMES))  # Don't let it go over the length of the video
        print("Process frame: {}".format(", ".join(["{:.2f}ms".format(1000*(t[i]-t[i-1]).total_seconds()) for i in range(1,len(t))])))

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

    def run(self):
        while not self.user_wants_to_exit.is_set():
            self.grab_next_frame()
        self.video_in.release()


class GroundTruthLabelerWindow:
    DELAY_FPS = 10  # msec

    def __init__(self, experiment_base_folder):
        self.weight_to_cam_t_offset = None
        self.t_offset_float = 0
        self.new_video_and_weight_frame_ready = Event()
        self.user_wants_to_exit = Event()

        self.video_and_weight = VideoAndWeightHandler(experiment_base_folder, self.on_set_event_time_start_or_end, self.new_video_and_weight_frame_ready, self.user_wants_to_exit)
        canvas_size = tuple( int(self.video_and_weight.out_scale*d) for d in self.video_and_weight.rgb_data.shape[:2] )

        # Load product info
        with open("Dataset/product_info.json", 'r') as f:
            self.product_info = json.load(f)['products']
        options = tuple((p['id'], p['name']) for p in self.product_info)
        column_headers = ("Time start", "Time end", "Pickup?", "Item ID", "Item name", "Quantity")
        column_widths = (186, 186, 50, 48, -1, 55)

        # Setup ui
        self.ui = tk.Tk()
        self.ui.title("Ground truth labeler")
        self.ui.geometry("{}x{}".format(canvas_size[1], canvas_size[0]+300))  # WxH
        self.ui.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.ui_container = ttk.Frame(self.ui)
        self.ui_container.pack(fill='both', expand=True, padx=10, pady=10, ipadx=20)

        # Variables
        self.quantity = tk.IntVar(self.ui, 1)
        self.selected_product = tk.Variable(self.ui, options[0])
        self.is_pickup = tk.BooleanVar(self.ui, True)
        self.t_start = None
        self.t_end = None
        self.events = []

        # Widgets
        self.canvas = tk.Canvas(self.ui, width=canvas_size[1], height=canvas_size[0])
        self.canvas_image = self.canvas.create_image(0, 0, image=self.video_and_weight.tk_img, anchor='nw')
        self.canvas.grid(column=0, columnspan=6, row=0, in_=self.ui_container)
        self.lst_events = MultiColumnListbox(column_headers, master=self.ui)
        for i,w in enumerate(column_widths):
            if w > 0:
                self.lst_events.tree.column(i, width=w, stretch=False)
        self.lst_events.tree.grid(column=0, columnspan=6, row=1, sticky='nesw', in_=self.ui_container)
        num_quantity = tk.Spinbox(self.ui, from_=1, to_=5, width=3, textvariable=self.quantity)
        num_quantity.grid(column=0, row=2, rowspan=2, in_=self.ui_container)
        drp_product = tk.OptionMenu(self.ui, self.selected_product, *options)
        drp_product.grid(column=1, row=2, rowspan=2, sticky='ew', in_=self.ui_container)
        opt_pickup = tk.Radiobutton(self.ui, text="Pick up", variable=self.is_pickup, value=True)
        opt_pickup.grid(column=2, row=2, sticky='ew', ipadx=10, in_=self.ui_container)
        opt_pickup = tk.Radiobutton(self.ui, text="Put back", variable=self.is_pickup, value=False)
        opt_pickup.grid(column=2, row=3, sticky='ew', ipadx=0, in_=self.ui_container)
        tk.Label(self.ui, text="Start:").grid(column=3, row=2, sticky='nsew', in_=self.ui_container)
        tk.Label(self.ui, text="End:").grid(column=3, row=3, sticky='nsew', in_=self.ui_container)
        self.txt_t_start = tk.Text(self.ui, state=tk.DISABLED, height=1, width=26)
        self.txt_t_start.grid(column=4, row=2, sticky='nsew', in_=self.ui_container)
        self.txt_t_end = tk.Text(self.ui, state=tk.DISABLED, height=1, width=26)
        self.txt_t_end.grid(column=4, row=3, sticky='nsew', in_=self.ui_container)
        self._update_time()  # Initialize their text
        btn_add_event = tk.Button(self.ui, text="Add event", command=self.add_event)
        btn_add_event.grid(column=5, row=2, rowspan=2, in_=self.ui_container)

        # Event handling
        self.ui.bind('<KeyPress>', self.video_and_weight.keys_pressed.put)
        self.lst_events.tree.bind('<KeyPress>', self.remove_event)

        # Make grids expandable on window resize
        self.ui_container.grid_rowconfigure(1, weight=1)
        self.ui_container.grid_columnconfigure(1, weight=1)

        # Load the first image
        self.update_canvas()

    def run(self):
        # Run main loop
        self.ui.mainloop()

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
        self.video_and_weight.grab_next_frame()
        if self.new_video_and_weight_frame_ready.is_set():
            self.new_video_and_weight_frame_ready.clear()
            self.canvas.itemconfig(self.canvas_image, image=self.video_and_weight.tk_img)

        # Check if user pressed Escape
        if self.user_wants_to_exit.is_set():
            self.on_closing()
        else:
            self.ui.after(self.DELAY_FPS, self.update_canvas)

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
            self.ui.destroy()

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


class GroundTruthLabeler:
    def __init__(self, main_folder, start_datetime=datetime.min, end_datetime=datetime.max):
        self.main_folder = main_folder
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime

    def run(self):
        for f in list_subfolders(self.main_folder, True):
            if f.endswith("_ignore"): continue
            t = datetime.strptime(f, DATETIME_FORMAT)  # Folder name specifies the date -> Convert to datetime

            # Filter by experiment date (only consider experiments within t_start and t_end)
            if self.start_datetime <= t <= self.end_datetime:
                experiment_folder = os.path.join(self.main_folder, f)
                ground_truth_file = os.path.join(experiment_folder, "ground_truth.json")
                if os.path.exists(ground_truth_file):
                    print("Video already annotated!! Skipping (delete '{}' and run this tool again if you want to overwrite)".format(ground_truth_file))
                    continue

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
    parser.add_argument("-s", "--start-datetime", default="", help="Only preprocess experiments collected later than this datetime (format: {}; empty for no limit)".format(DATETIME_FORMAT))
    parser.add_argument("-e", "--end-datetime", default="", help="Only preprocess experiments collected before this datetime (format: {}; empty for no limit)".format(DATETIME_FORMAT))
    args = parser.parse_args()

    t_start = datetime.strptime(args.start_datetime, DATETIME_FORMAT) if len(args.start_datetime) > 0 else datetime.min
    t_end = datetime.strptime(args.end_datetime, DATETIME_FORMAT) if len(args.end_datetime) > 0 else datetime.max

    GroundTruthLabeler(args.folder, t_start, t_end).run()
