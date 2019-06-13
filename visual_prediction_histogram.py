from generate_video import generate_multicam_video
from aux_tools import ExperimentTraverser, EXPERIMENT_DATETIME_STR_FORMAT
from read_dataset import parse_product_info
from preprocess_experiments import HDF5_FRAME_NAME_FORMAT
from queue import Queue
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os
import cv2
import numpy as np
import h5py
import re
import argparse

# Import UI
try:
    import Tkinter as tk
except ImportError:  # Python 3
    import tkinter as tk
from ResizableImageCanvas import ResizableImageCanvas


class VisuallyPredictedItem:
    THICKNESS_NORMAL   = 1
    THICKNESS_HOVER    = 3
    THICKNESS_SELECTED = 3
    COLOR_NORMAL   = (200,   0,   0)
    COLOR_HOVER    = (  0, 200, 200)
    COLOR_SELECTED = (  0, 200,   0)

    def __init__(self, product_info, color=None):
        self.product_info = product_info
        self.color = color
        self.is_hovered = False
        self.is_selected = False

        # Create views of product_info for quick access
        self.xy_min = self.product_info[0:2]
        self.xy_max = self.product_info[2:4]
        self.center = (self.xy_max + self.xy_min)/2
        self.class_prob = self.product_info[4:]

    def get_state(self):
        if self.is_hovered:
            return "HOVER"
        elif self.is_selected:
            return "SELECTED"
        else:
            return "NORMAL"

    def dist_to_center(self, x, y):
        if (self.xy_min[0] <= x <= self.xy_max[0]) and (self.xy_min[1] <= y <= self.xy_max[1]):
            return abs(x - self.center[0]) + abs(y - self.center[1])
        else:
            return float("inf")

    def render(self, img):
        color = getattr(self, "COLOR_{}".format(self.get_state())) if self.color is None else self.color
        thickness = getattr(self, "THICKNESS_{}".format(self.get_state()))

        cv2.rectangle(img, tuple(self.xy_min.astype(int)), tuple(self.xy_max.astype(int)), color, thickness)


class VisuallyPredictedItemsManager:
    HOVER_MARGIN = 5

    def __init__(self):
        self.items_in_frame = []
        self.items_coords = np.empty((0,4))  # Keep a copy of all items coords so we can find closest item to mouse click in parallel
        self.selected_item = None
        self.mouse_x, self.mouse_y = 0, 0

    def clear(self):
        self.items_in_frame = []
        self.items_coords = np.empty((0,4))
        self.selected_item = None

    def add(self, products_found):
        if len(products_found) == 0:
            return

        self.items_coords = np.vstack((self.items_coords, products_found[:, :4]))
        for product_info in products_found:
            self.items_in_frame.append(VisuallyPredictedItem(product_info))

    def on_mouse_event(self, x, y, bool_click):
        self.mouse_x, self.mouse_y = x, y
        self.update_hover(bool_click)

    def update_hover(self, bool_select=False):
        if len(self.items_in_frame) == 0:
            return  # Nothing to do
        for i in self.items_in_frame:
            i.is_hovered = False

        inds_mouse_is_inside = np.argwhere(np.all((
            self.items_coords[:,0]-self.HOVER_MARGIN <= self.mouse_x,
            self.items_coords[:,2]+self.HOVER_MARGIN >= self.mouse_x,
            self.items_coords[:,1]-self.HOVER_MARGIN <= self.mouse_y,
            self.items_coords[:,3]+self.HOVER_MARGIN >= self.mouse_y)
            , axis=0)).ravel()
        if len(inds_mouse_is_inside) > 0:
            dist_to_bbox_edges = np.abs(self.items_coords[inds_mouse_is_inside,:] - np.tile((self.mouse_x, self.mouse_y), 2))
            ind_closest = inds_mouse_is_inside[np.argmin(dist_to_bbox_edges)//dist_to_bbox_edges.shape[1]]
            if bool_select:
                if self.selected_item is not None:
                    self.selected_item.is_selected = False
                if self.selected_item is not self.items_in_frame[ind_closest]:  # Clicking on an already selected item should de-select it (so skip selecting it here). In any other case, select the new item
                    self.selected_item = self.items_in_frame[ind_closest]
                    self.selected_item.is_selected = True
            else:
                self.items_in_frame[ind_closest].is_hovered = True

    def render(self, img, is_new_frame=True):
        # Need to create a copy of the original img so we don't need to rerender all bboxes if we later select another item
        new_img = img.copy()

        # Update hover state (new frame has new self.items_in_frame -> Hover one based on last saved mouse position)
        if is_new_frame:
            self.update_hover()

        # Render all item bboxes
        for item in self.items_in_frame:
            item.render(new_img)

        return new_img


class ProductPredictionVisualizer(tk.Tk):
    CANVAS_UPDATE_PERIOD = 10  # msec
    WIN_PAD = 10
    FIG_W = 550

    def __init__(self, video_filename, visual_predictions, frame_nums=None, is_multicam=True, with_hist_navbar=False):
        super(ProductPredictionVisualizer, self).__init__()

        self.video_filename = video_filename
        self.visual_predictions = visual_predictions
        self.cams = visual_predictions.keys()
        self.is_multicam = is_multicam
        self.items_in_frame_manager = VisuallyPredictedItemsManager()
        self.products_info = sorted(parse_product_info(), key=lambda product_info: product_info.get("training_id", float("inf")))  # Load info about products: name, id, barcode, etc
        self.products_names = [product_info.get("name", "Unkwown name!") for product_info in self.products_info]

        # Load video
        self.v = cv2.VideoCapture(self.video_filename)
        W, H = int(self.v.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.v.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.halfW, self.halfH = W//2, H//2
        self.img = np.zeros((H, W, 3), dtype=np.uint8)
        self.N_frames = int(self.v.get(cv2.CAP_PROP_FRAME_COUNT))
        self.n_frame = int(self.v.get(cv2.CAP_PROP_POS_FRAMES))
        self.frame_nums = frame_nums if frame_nums is not None else np.tile(np.arange(1, self.N_frames+1), (4,1))
        self.do_skip_frames = False
        self.is_paused = False

        # Preprocess all coords if multicam to adjust position
        for cam, pred_frames in self.visual_predictions.items():
            # Compute horiz. and vert. offset for this camera [e.g. cam 2 is in the bottom left corner -> (0, halfH)]
            xy_offset = (self.halfW if int(cam)>2 else 0, self.halfH if (int(cam) % 2)==0 else 0)

            # Iterate all frames where products where found and modify the coordinates of the bounding boxes
            for products_found in pred_frames.values():
                products_found[:, :4] = products_found[:, :4]/2 + np.tile(np.tile(xy_offset, 2), (products_found.shape[0], 1))

        # Setup ui
        self.title("Visual product prediction for video {}".format(os.path.basename(video_filename)))
        win_size = np.array((W + self.FIG_W + 2*self.WIN_PAD, H + 2*self.WIN_PAD))
        win_offs = (np.array((self.winfo_screenwidth(), self.winfo_screenheight())) - win_size)/2
        self.geometry("{s[0]}x{s[1]}+{o[0]}+{o[1]}".format(s=win_size.astype(int), o=win_offs.astype(int)))
        self.keys_pressed = Queue()
        self.bind('<KeyPress>', self.keys_pressed.put)
        self.ui_container = tk.Frame(self)
        self.ui_container.pack(fill="both", expand=True, padx=self.WIN_PAD, pady=self.WIN_PAD)

        # Setup video
        self.video_canvas = ResizableImageCanvas(master=self, width=W, height=H, highlightthickness=0)
        self.video_canvas.grid(row=0, column=0, rowspan=2, padx=(0, self.WIN_PAD), in_=self.ui_container)
        self.video_canvas.bind("<Motion>", self.on_mouse_event)
        self.video_canvas.bind("<Button-1>", self.on_mouse_event)

        # Setup histogram
        self.hist_container = tk.Frame(self)
        self.hist_container.grid(row=0, column=1, rowspan=1+(not with_hist_navbar), in_=self.ui_container)
        self.hist_fig, self.hist_ax = plt.subplots(figsize=(self.FIG_W/100., self.winfo_screenheight()/100.), dpi=100)
        self.hist_bars = []
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=self)
        self.hist_canvas.get_tk_widget().pack(fill="both", expand=True, in_=self.hist_container)
        self.hist_canvas.mpl_connect('resize_event', self._on_hist_canvas_resize)
        self.plot_prob_hist(create_ax=True)
        if with_hist_navbar:
            self.hist_navbar_container = tk.Frame(self)
            self.hist_navbar_container.grid(row=1, column=1, sticky='ew', padx=5, in_=self.ui_container)
            self.hist_navbar = NavigationToolbar2Tk(self.hist_canvas, self.hist_navbar_container)

        self.ui_container.grid_columnconfigure(0, weight=1)
        self.ui_container.grid_rowconfigure(0, weight=1)
        self.update()

    def _on_hist_canvas_resize(self, event):
        plt.tight_layout(0)

    def update_canvas(self):
        if not self.is_paused or self.do_skip_frames:
            self.items_in_frame_manager.clear()

            # Read frames until we reach one where at least one object/bbox was found
            while len(self.items_in_frame_manager.items_in_frame) == 0 and self.n_frame < self.N_frames:
                assert self.v.read(self.img)[0], "Couldn't read frame from {}!".format(self.video_filename)
                self.n_frame += 1
                print("Read frame {:4d}/{} ({:.2f}%)".format(self.n_frame, self.N_frames, 100.*self.n_frame/self.N_frames))

                # Process products found in this frame
                for cam, pred_frames in self.visual_predictions.items():
                    products_found = pred_frames.get(HDF5_FRAME_NAME_FORMAT.format(self.frame_nums[cam-1, self.n_frame-1]), ())
                    self.items_in_frame_manager.add(products_found)

            # Visualize results
            self._render()

        # Process key presses
        self.process_kb()

        self.after(self.CANVAS_UPDATE_PERIOD, self.update_canvas)

    def on_mouse_event(self, event):
        bool_click = (event.type == tk.EventType.ButtonPress)
        x, y = self.video_canvas.canvas_to_img_coords((event.x, event.y))
        self.items_in_frame_manager.on_mouse_event(x, y, bool_click)
        if bool_click:
            self.plot_prob_hist(self.items_in_frame_manager.selected_item.class_prob if self.items_in_frame_manager.selected_item is not None else None)

        self._render(is_new_frame=False)

    def process_kb(self):
        self.do_skip_frames = False
        while not self.keys_pressed.empty():
            key_info = self.keys_pressed.get()
            k = key_info.keysym.lower()
            if k == 'space':
                self.is_paused = not self.is_paused
            elif k == 'a' or k == 'left':
                self.n_frame -= 10
                self.do_skip_frames = True
            elif k == 'd' or k == 'right':
                self.n_frame += 10
                self.do_skip_frames = True
            elif k == 'w' or k == 'up':
                self.n_frame += 1
                self.do_skip_frames = True
            elif k == 's' or k == 'down':
                self.n_frame -= 1
                self.do_skip_frames = True
            elif k == 'q' or k == 'escape':
                self.destroy()

        if self.do_skip_frames:
            self.v.set(cv2.CAP_PROP_POS_FRAMES, self.n_frame-1)
            self.n_frame = int(self.v.get(cv2.CAP_PROP_POS_FRAMES))  # Make sure we don't go negative or over N_frames

    def plot_prob_hist(self, class_prob=None, create_ax=False):
        # Prepare results (x and y axes)
        y = np.arange(len(self.products_info))
        x = np.zeros(len(self.products_info))
        if class_prob is not None:
            for product_info in self.products_info:
                i = product_info.get("training_id", -1)
                if i >= 0:
                    x[i] = class_prob[i]

        if create_ax:
            self.hist_bars = self.hist_ax.barh(y, x, tick_label=self.products_names)
            self.hist_ax.set_xlim(0, 1)
            self.hist_ax.set_ylim(-1, len(y))
            plt.tight_layout(0)
        else:
            for i,bar in enumerate(self.hist_bars):
                bar.set_width(x[i])
        self.hist_canvas.draw()

    def _render(self, is_new_frame=True):
        self.video_canvas.update_image(self.items_in_frame_manager.render(self.img, is_new_frame))

    def run(self):
        self.update()  # Initialize UI
        self.update_canvas()  # Start the periodic canvas update
        self.mainloop()  # Run Tk's main loop


class ProductPredictionExperimentsVisualizer(ExperimentTraverser):
    def __init__(self, main_folder, start_datetime, end_datetime, cams):
        super(ProductPredictionExperimentsVisualizer, self).__init__(main_folder, start_datetime, end_datetime)
        self.cams = cams
        self.is_multicam = (self.cams[0] == "--all")  # Boolean flag indicating whether to visualize all cams at once

    def process_subfolder(self, f):
        experiment_folder = os.path.join(self.main_folder, f)
        cams_folder = os.path.join("Dataset/Evaluation full contents", f)
        if self.is_multicam:
            all_files_in_folder = sorted(next(os.walk(experiment_folder))[2])
            cams = []
            for file in all_files_in_folder:
                s = re.search(r"product_prediction_cam(\d+)_{}.h5".format(f), file)
                if s is not None:
                    cams.append(s.group(1))
            cam_video_filenames = [generate_multicam_video(experiment_folder)]
        else:
            cams = self.cams
            cam_video_filenames = [os.path.join(cams_folder, f, "cam{}_{}.mp4".format(c, f)) for c in cams]

        # Load resampled timing (-> makes fps ~constant)
        with h5py.File(os.path.join(experiment_folder, "multicam_{}.h5".format(f)), 'r') as f_hdf5:
            frame_nums = f_hdf5['frame_nums'][:]

        # Load predictions for each cam
        visual_predictions = {}
        for cam in cams:
            with h5py.File(os.path.join(experiment_folder, "product_prediction_cam{}_{}.h5".format(cam, f)), 'r') as f_hdf5:
                visual_predictions[int(cam)] = dict([(frame_num, prods[:]) for frame_num, prods in f_hdf5.items()])
            print("Loaded visual product predictions from cam {}...".format(cam))

        # Visualize video
        for video_filename in cam_video_filenames:
            ProductPredictionVisualizer(video_filename, visual_predictions, frame_nums, self.is_multicam).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", default="Dataset/Evaluation", help="Folder containing the experiment(s) to preprocess")
    parser.add_argument("-s", "--start-datetime", default="", help="Only preprocess experiments collected later than this datetime (format: {}; empty for no limit)".format(EXPERIMENT_DATETIME_STR_FORMAT))
    parser.add_argument("-e", "--end-datetime", default="", help="Only preprocess experiments collected before this datetime (format: {}; empty for no limit)".format(EXPERIMENT_DATETIME_STR_FORMAT))
    parser.add_argument("-c", "--cams", nargs='+', default=["--all"], help="Cameras to visualize (e.g. '1 3' to visualize cam1_ and cam3_).")
    args = parser.parse_args()

    t_start = datetime.strptime(args.start_datetime, EXPERIMENT_DATETIME_STR_FORMAT) if len(args.start_datetime) > 0 else datetime.min
    t_end = datetime.strptime(args.end_datetime, EXPERIMENT_DATETIME_STR_FORMAT) if len(args.end_datetime) > 0 else datetime.max

    ProductPredictionExperimentsVisualizer(args.folder, t_start, t_end, args.cams).run()