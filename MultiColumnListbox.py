try:
    import Tkinter as tk
    import tkFont
    import ttk
except ImportError:  # Python 3
    import tkinter as tk
    import tkinter.font as tkFont
    import tkinter.ttk as ttk


class MultiColumnListbox(object):
    """Use a ttk.TreeView as a multicolumn ListBox"""

    def __init__(self, headers, data=None, sortable=True, scrollbars_on_overflow=False, autowidth_on_add=False, master=None, **kw):
        self.container = tk.Frame(master)
        self.container.pack(fill='both', expand=True)

        self.tree = ttk.Treeview(master, columns=headers, show="headings", **kw)

        if not scrollbars_on_overflow:
            self.tree.pack(fill='x', in_=self.container)
        else:
            vsb = ttk.Scrollbar(orient="vertical", command=self.tree.yview)
            hsb = ttk.Scrollbar(orient="horizontal", command=self.tree.xview)
            self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            self.tree.grid(column=0, row=0, sticky='nsew', in_=self.container)
            vsb.grid(column=1, row=0, sticky='ns', in_=self.container)
            hsb.grid(column=0, row=1, sticky='ew', in_=self.container)
            self.container.grid_columnconfigure(0, weight=1)
            self.container.grid_rowconfigure(0, weight=1)

        for iCol,col in enumerate(headers):
            on_sort = (lambda c=iCol: self.sortby(c)) if sortable else ''
            self.tree.heading(iCol, text=col, command=on_sort)
            self.tree.column(iCol, width=tkFont.Font().measure(col))  # Adjust the column's width to the header string

        self.autowidth_on_add = autowidth_on_add
        if data is not None:
            for item in data:
                self.add_item(item)

    def autowidth(self, new_item):
        for iCol, val in enumerate(new_item):
            if val is None: continue
            col_w = tkFont.Font().measure(val)
            if self.tree.column(iCol, width=None) < col_w:
                self.tree.column(iCol, width=col_w)

    def add_item(self, new_item):
        self.tree.insert('', 'end', values=new_item)
        if self.autowidth_on_add:
            self.autowidth(new_item)

    def sortby(self, iCol, descending=False):
        """sort tree contents when a column header is clicked on"""
        # Grab values to sort
        data = [(self.tree.set(child, iCol), child) for child in self.tree.get_children('')]
        # Sort the data in place
        data.sort(reverse=descending)
        for iRow, item in enumerate(data):
            self.tree.move(item[1], '', iRow)
        # Switch the heading so it will sort in the opposite direction
        self.tree.heading(iCol, command=lambda c=iCol: self.sortby(c, not descending))
