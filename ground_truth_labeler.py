from multiprocessing import Process, Pipe
from threading import Thread, Event
import json


class GroundTruthLabeler:
    def __init__(self, pipe):
        self.pipe = pipe
        self.done = Event()
        self.t_pipe = Thread(target=self.handle_pipe, daemon=False)
        self.t_pipe.start()

        # Load product info
        with open("Dataset/product_info.json", 'r') as f:
            self.product_info = json.load(f)['products']

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
        self.ui.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.ui_container = ttk.Frame(self.ui)
        self.ui_container.pack(fill='both', expand=True, padx=10, pady=10, ipadx=20)

        # Variables
        self.selected_product = tk.Variable(self.ui)
        self.is_pickup = tk.BooleanVar(self.ui, True)
        self.t_start = None
        self.t_end = None
        self.events = []

        # Widgets
        self.lst_events = MultiColumnListbox(("Time start", "Time end", "Pickup?", "Item ID", "Item name", "Quantity"), master=self.ui)
        self.lst_events.tree.grid(column=0, columnspan=5, row=0, sticky='nesw', in_=self.ui_container)
        drp_product = tk.OptionMenu(self.ui, self.selected_product, *((p['id'], p['name']) for p in self.product_info))
        drp_product.grid(column=0, row=1, rowspan=2, sticky='ew', in_=self.ui_container)
        opt_pickup = tk.Radiobutton(self.ui, text="Pick up", variable=self.is_pickup, value=True)
        opt_pickup.grid(column=1, row=1, sticky='ew', ipadx=10, in_=self.ui_container)
        opt_pickup = tk.Radiobutton(self.ui, text="Put back", variable=self.is_pickup, value=False)
        opt_pickup.grid(column=1, row=2, sticky='ew', ipadx=0, in_=self.ui_container)
        tk.Label(self.ui, text="Start:").grid(column=2, row=1, sticky='nsew', in_=self.ui_container)
        tk.Label(self.ui, text="End:").grid(column=2, row=2, sticky='nsew', in_=self.ui_container)
        self.txt_t_start = tk.Text(self.ui, state=tk.DISABLED, height=1, width=10)
        self.txt_t_start.grid(column=3, row=1, sticky='nsew', in_=self.ui_container)
        self.txt_t_end = tk.Text(self.ui, state=tk.DISABLED, height=1, width=10)
        self.txt_t_end.grid(column=3, row=2, sticky='nsew', in_=self.ui_container)
        self._update_time()  # Initialize their text
        btn_add_event = tk.Button(self.ui, text="Add event", command=self.add_event)
        btn_add_event.grid(column=4, row=1, rowspan=2, in_=self.ui_container)

        # Event handling
        self.lst_events.tree.bind('<KeyPress>', self.remove_event)

        # Make grids expandable on window resize
        self.ui_container.grid_columnconfigure(0, weight=1)
        self.ui_container.grid_rowconfigure(0, weight=1)

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
        self.events = [self.lst_events.tree.item(child, values=None) for child in self.lst_events.tree.get_children('')]

        if self.msg_box.askokcancel("Exit?", "Are you sure you're done annotating this experiment's ground truth?\nWe've registered these events:\n\t{}".format(
            '\n\t'.join('{' + ', '.join(e) + '}' for e in self.events)
        )):
            self.ui.destroy()

    def add_event(self):
        new_item = (self.t_start, self.t_end, self.is_pickup.get(), *self.selected_product.get())
        print("Adding event: {}".format(new_item))
        self.lst_events.add_item(new_item)

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


if __name__ == '__main__':
    pipe_local, pipe_remote = Pipe()
    ui_process = Process(target=GroundTruthLabeler, args=(pipe_remote,))
    ui_process.start()
    import time
    time.sleep(3)
    pipe_local.send((True,"Hooolaa"))
    time.sleep(2)
    pipe_local.send((False,"123"))
    print("Received: {}".format(pipe_local.recv()))
    ui_process.join()
