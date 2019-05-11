from multiprocessing import Process, Pipe
import json

class GroundTruthLabeler:
    def __init__(self, pipe):
        self.pipe = pipe

        # Load product info
        with open("Dataset/product_info.json", 'r') as f:
            self.prooduct_info = json.load(f)['products']

        # Import UI
        try:
            import Tkinter as Tk
        except ModuleNotFoundError:
            import tkinter as Tk

        # Setup ui
        self.ui = Tk.Tk()
        self.ui.title("Ground truth labeler")
        self.selected_product = Tk.Variable(self.ui)
        self.drp_product = Tk.OptionMenu(self.ui, self.selected_product, *((p['id'], p['name']) for p in self.prooduct_info), command=self.on_change)
        self.drp_product.pack(fill=Tk.X)
        self.btn_accept = Tk.Button(self.ui, text="Add item", command=self.terminate)
        self.btn_accept.pack(fill=Tk.X)
        Tk.Listbox(self.ui).pack()
        Tk.Radiobutton(self.ui, text="Option 1").pack()
        Tk.Radiobutton(self.ui, text="Option 2").pack()

        self.ui.mainloop()

    def terminate(self):
        print("Value is: {}".format(self.selected_product.get()))
        self.ui.destroy()

    def on_change(self, value):
        self.pipe.send(value)

    def test(self, p):
        print(p)


if __name__ == '__main__':
    pipe_local, pipe_remote = Pipe()
    ui_process = Process(target=GroundTruthLabeler, args=(pipe_remote,))
    ui_process.start()
    # print("Waited for: {}".format(pipe_local.recv()))
    ui_process.join()
