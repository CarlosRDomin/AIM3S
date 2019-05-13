from multiprocessing import Process, Pipe
from threading import Thread, Event
from generate_video import generate_video
from preprocess_experiments import DATETIME_FORMAT
from aux_tools import list_subfolders, str2bool
from datetime import datetime
import os
import json
import argparse


class GroundTruthLabelerWindow:
    def __init__(self, pipe):
        self.pipe = pipe
        self.done = Event()
        self.t_pipe = Thread(target=self.handle_pipe)
        self.t_pipe.setDaemon(False)
        self.t_pipe.start()

        # Load product info
        with open("Dataset/product_info.json", 'r') as f:
            self.product_info = json.load(f)['products']
        options = tuple((p['id'], p['name']) for p in self.product_info)
        column_headers = ("Time start", "Time end", "Pickup?", "Item ID", "Item name", "Quantity")
        column_widths = (186, 186, 50, 48, -1, 55)

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
        self.msg_box = messagebox

        # Setup ui
        self.ui = tk.Tk()
        self.ui.title("Ground truth labeler")
        self.ui.geometry("850x300")
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
        self.lst_events = MultiColumnListbox(column_headers, master=self.ui)
        for i,w in enumerate(column_widths):
            if w > 0:
                self.lst_events.tree.column(i, width=w, stretch=False)
        self.lst_events.tree.grid(column=0, columnspan=6, row=0, sticky='nesw', in_=self.ui_container)
        num_quantity = tk.Spinbox(self.ui, from_=1, to_=5, width=2, textvariable=self.quantity)
        num_quantity.grid(column=0, row=1, rowspan=2, in_=self.ui_container)
        drp_product = tk.OptionMenu(self.ui, self.selected_product, *options)
        drp_product.grid(column=1, row=1, rowspan=2, sticky='ew', in_=self.ui_container)
        opt_pickup = tk.Radiobutton(self.ui, text="Pick up", variable=self.is_pickup, value=True)
        opt_pickup.grid(column=2, row=1, sticky='ew', ipadx=10, in_=self.ui_container)
        opt_pickup = tk.Radiobutton(self.ui, text="Put back", variable=self.is_pickup, value=False)
        opt_pickup.grid(column=2, row=2, sticky='ew', ipadx=0, in_=self.ui_container)
        tk.Label(self.ui, text="Start:").grid(column=3, row=1, sticky='nsew', in_=self.ui_container)
        tk.Label(self.ui, text="End:").grid(column=3, row=2, sticky='nsew', in_=self.ui_container)
        self.txt_t_start = tk.Text(self.ui, state=tk.DISABLED, height=1, width=26)
        self.txt_t_start.grid(column=4, row=1, sticky='nsew', in_=self.ui_container)
        self.txt_t_end = tk.Text(self.ui, state=tk.DISABLED, height=1, width=26)
        self.txt_t_end.grid(column=4, row=2, sticky='nsew', in_=self.ui_container)
        self._update_time()  # Initialize their text
        btn_add_event = tk.Button(self.ui, text="Add event", command=self.add_event)
        btn_add_event.grid(column=5, row=1, rowspan=2, in_=self.ui_container)

        # Event handling
        self.lst_events.tree.bind('<KeyPress>', self.remove_event)

        # Make grids expandable on window resize
        self.ui_container.grid_rowconfigure(0, weight=1)
        self.ui_container.grid_columnconfigure(1, weight=1)

        # Run main loop
        self.ui.mainloop()
        self.done.set()  # Tell the thread t_pipe to send results (it's non-daemonic so the process won't die until it's done sending)
        self.t_pipe.join()  # Wait for the pipe thread to be done

    def handle_pipe(self):
        while not self.done.is_set():
            while self.pipe.poll(1):
                rx = self.pipe.recv()
                print("Received: {}".format(rx))
                if rx[0]:
                    self.t_start = rx[1]
                else:
                    self.t_end = rx[1]
                self._update_time()

        # Done, send all events before leaving
        self.pipe.send(self.events)

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

        if self.msg_box.askokcancel("Exit?", "Are you sure you're done annotating this experiment's ground truth?\nWe've registered {} event{}".format(len(self.events), '' if len(self.events)==1 else 's')):
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
            if len(selected_items) > 0 and self.msg_box.askokcancel("Are you sure?", "Are you sure you want to remove {} item{}?".format(len(selected_items), 's' if len(selected_items)>1 else '')):
                self.lst_events.tree.delete(*selected_items)

    def _set_time(self, is_t_start):
        if is_t_start:
            txt_box = self.txt_t_start
            text = self.t_start if self.t_start is not None else "Press 'b' on cv window"
        else:
            txt_box = self.txt_t_end
            text = self.t_end if self.t_end is not None else "Press 'n' on cv window"

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

        # Initialize UI process, pipes and cv thread
        self.t_video = None
        self.pipe_local, pipe_remote = Pipe()
        self.ui_process = Process(target=GroundTruthLabelerWindow, args=(pipe_remote,))

    def on_set_event_time_start_or_end(self, is_start, t):
        self.pipe_local.send((is_start, t))

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
                self.ui_process.start()
                weight_to_cam_t_offset, t_offset_float = generate_video(experiment_folder, -1, -1, cb_event_start_or_end=self.on_set_event_time_start_or_end)
                print("Generate_video finished! Weight-camera time offset manually set as {} ({}s wrt weight's own timestamps)".format(weight_to_cam_t_offset, t_offset_float))
                annotated_events = self.pipe_local.recv()
                print("Received annotated events: {}".format(annotated_events))
                self.ui_process.join()
                with open(ground_truth_file, 'w') as f_gt:
                    json.dump({
                        'ground_truth': annotated_events,
                        'weight_to_cam_t_offset': str(weight_to_cam_t_offset),
                        'weight_to_cam_t_offset_float': t_offset_float,
                    }, f_gt, indent='\t')
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
