from threading import Event
from generate_video import generate_multicam_video
from preprocess_experiments import DATETIME_FORMAT
from aux_tools import list_subfolders, str2bool, str_to_datetime, date_range, time_to_float, format_axis_as_timedelta, plt_fig_to_cv2_img
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
        self.out_scale = 0.5  # Rescale video_img before converting to Tkinter image (~3X faster to render)
        self.keys_pressed = Queue()
        self.tk_img = None

        # Load video info
        video_in_filename = generate_multicam_video(experiment_base_folder)
        self.video_in = cv2.VideoCapture(video_in_filename)
        with h5py.File(os.path.splitext(video_in_filename)[0] + ".h5", 'r') as h5_cam:
            self.t_cam = np.array(list(date_range(str_to_datetime(h5_cam.attrs['t_start']), str_to_datetime(h5_cam.attrs['t_end']), timedelta(seconds=1.0/h5_cam.attrs['fps']))))
        self.video_dims = np.array([self.video_in.get(cv2.CAP_PROP_FRAME_HEIGHT), self.video_in.get(cv2.CAP_PROP_FRAME_WIDTH)]).astype(int)
        self.video_downsampled_dims = (self.out_scale * self.video_dims).astype(int)
        self.weight_dims = np.array([self.video_downsampled_dims[0], 350]).astype(int)

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
            ax[i,0].plot(t_w, w[shelf_i], visible=False)
            self.curr_t_lines.append(ax[i,0].axvline(0, linestyle='--', color='black', linewidth=1, visible=False))
            ax[i,0].set_title('Shelf {}'.format(shelf_i+1))
            ax[i,0].set_xlim(-self.t_lims, self.t_lims)
            format_axis_as_timedelta(ax[i,0].xaxis)

            # update_xaxis=False means weight's xaxis will be static (always show: -0:03 -0:02 ... 0:03)
            # update_xaxis=True means we'll rerender the xaxis on every replot -> Need to hide the labels before copying the canvas bgnd
            ax[i,0].tick_params(axis='both', which='both', bottom=not self.update_xaxis, labelbottom=not self.update_xaxis)

        # Render the figure and save background so updating the plot can be much faster (using blit instead of draw)
        self.fig.canvas.draw()
        self.bg_cache = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        for i in range(num_subplots):  # Make everything visible again
            for l in ax[i,0].lines: l.set_visible(True)
            is_last = (i == num_subplots-1)
            ax[i,0].tick_params(axis='both', which='both', bottom=True, labelbottom=is_last)

        # Allocate memory space for a video frame and a downsampled copy
        self.video_img = np.zeros((self.video_dims[0], self.video_dims[1], 3), dtype=np.uint8)
        self.video_downsampled_img = np.zeros((self.video_downsampled_dims[0], self.video_downsampled_dims[1], 3), dtype=np.uint8) if self.out_scale != 1 else None

    def init_tk_img(self, master=None):
        self.tk_img = ImageTk.PhotoImage(master=master, width=self.video_downsampled_dims[1], height=self.video_downsampled_dims[0], image="RGB")
        return self.tk_img

    def update(self):
        # Update video frame (if needed)
        if not self.is_paused or self.do_skip_frames:
            # Grab next frame
            self.n += 1
            ok = self.video_in.read(self.video_img[:, :self.video_dims[1], :])
            assert ok, "Couldn't read frame {}!".format(self.n)
            print("Read frame {} out of {} frames ({:6.2f}%)".format(self.n+1, len(self.t_cam), 100.0*(self.n+1)/len(self.t_cam)))

            # Render the frame
            if self.video_downsampled_img is not None:
                img = cv2.resize(self.video_img, tuple(self.video_downsampled_dims[::-1]), self.video_downsampled_img)
            else:
                img = self.video_img
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            self.tk_img.paste(Image.fromarray(img))

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


class GroundTruthLabelerWindow:
    DELAY_FPS = 10  # msec

    def __init__(self, experiment_base_folder):
        self.weight_to_cam_t_offset = None
        self.t_offset_float = 0
        self.user_wants_to_exit = Event()

        self.video_and_weight = VideoAndWeightHandler(experiment_base_folder, self.on_set_event_time_start_or_end, self.user_wants_to_exit)
        video_canvas_size = self.video_and_weight.video_downsampled_dims
        weight_canvas_size = self.video_and_weight.weight_dims

        # Load product info
        with open("Dataset/product_info.json", 'r') as f:
            self.product_info = json.load(f)['products']
        options = tuple((p['id'], p['name']) for p in self.product_info)
        column_headers = ("Time start", "Time end", "Pickup?", "Item ID", "Item name", "Quantity")
        column_widths = (186, 186, 50, 48, -1, 55)

        # Setup ui
        self.ui = tk.Tk()
        self.ui.title("Ground truth labeler")
        self.ui.geometry("{}x{}".format(video_canvas_size[1]+weight_canvas_size[1], video_canvas_size[0]+300))  # WxH
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
        self.canvas_container = ttk.Frame(self.ui)
        self.canvas_container.grid(column=0, columnspan=6, row=0, in_=self.ui_container)
        self.video_canvas = tk.Canvas(self.ui, width=video_canvas_size[1], height=video_canvas_size[0])
        self.video_canvas.grid(column=0, row=0, in_=self.canvas_container)
        self.canvas_image = self.video_canvas.create_image(0, 0, image=self.video_and_weight.init_tk_img(self.video_canvas), anchor='nw')
        self.weight_canvas = FigureCanvasTkAgg(self.video_and_weight.fig, master=self.ui)
        self.weight_canvas.draw()
        self.weight_canvas.get_tk_widget().grid(column=1, row=0, in_=self.canvas_container)
        self.lst_events = MultiColumnListbox(column_headers, master=self.ui)
        for i,w in enumerate(column_widths):
            if w > 0:
                self.lst_events.tree.column(i, width=w, stretch=False)
        self.lst_events.tree.grid(column=0, columnspan=6, row=1, sticky='nesw', in_=self.ui_container)
        num_quantity = tk.Spinbox(self.ui, from_=1, to_=5, width=1, borderwidth=0, textvariable=self.quantity)
        num_quantity.grid(column=0, row=2, rowspan=2, in_=self.ui_container)
        drp_product = tk.OptionMenu(self.ui, self.selected_product, *options)
        drp_product.grid(column=1, row=2, rowspan=2, sticky='ew', ipadx=10, in_=self.ui_container)
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
        self.video_and_weight.update()

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
