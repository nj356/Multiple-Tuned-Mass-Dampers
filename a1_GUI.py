#! /usr/bin/env python3

import tkinter
import matplotlib
import matplotlib.backends.backend_tkagg as tkagg
import numpy as np
import pathlib
import a1


class FloatEntry(tkinter.Entry):

    """Specialised Entry widget which should contain a float."""

    def __init__(self, root, width, on_changed=lambda: None):

        """Construct a FloatEntry object.

        Args:
            root:
                The Tk root window in which to create the widget.
            width:
                The width (in characters) of the widget.
            on_changed:
                A callable to be notified of any change to the value of the
                widget. No arguments are passed when this call is made.
        """

        self._on_changed = on_changed

        self._text = tkinter.StringVar()
        self._text.trace_add("write", self._text_change)

        super().__init__(root, width=width, textvariable=self._text)

        self._validate()

    @property
    def value(self):

        """Return the value of the widget.

        Returns:
            The value in the Entry widget as a float, or None if the widget
            text is not a valid float.
        """

        return self._value

    def _text_change(self, *args):

        # Recieve notification that the value of the Entry widget has changed.
        # Performs validation and then calls any on_changed callable supplied
        # when the object was constructed.

        self._validate()
        self._on_changed()

    def _validate(self):

        # Called to validate the current contents of the Entry widget. If the
        # contents are a valid float the widget background is set to white and
        # the value property to that float value. Otherwise the background is
        # set to pink and the value property to None.

        try:
            self._value = float(self.get())
            self.config({"background": "White"})
        except ValueError:
            self._value = None
            self.config({"background": "Pink"})


class Params:

    """Class representing a collection of named label/entry pairs."""

    def __init__(self, on_changed=lambda: None):

        """Construct a Params object.

        Args:
            on_changed:
                A callable to be notified of any change to the value of the
                any of the widgets in the collection. No arguments are passed
                when this call is made.
        """

        self._labels1 = {}
        self._entries = {}
        self._labels2 = {}
        self._on_changed = on_changed

    def create_param(self, root, label, key, row, column):

        """Create a Label/Entry pair and add it to the collection.

        Args:
            root:
                The Tk root window in which to create the widgets.
            label:
                The text to be displayed by the Label widget.
            key:
                A string which identifies the Label/Entry pair.
            row:
                The row of the parent grid in which to place the Label/Entry
                pair.
            column:
                The column of the parent grid in which to place the Label
                widget. The Entry widget is placed in column+1 and a units
                label in column+2.
        """

        text1, text2 = label.split(':')

        label1 = tkinter.Label(root, text=text1)
        entry = FloatEntry(root, width=10, on_changed=self._on_changed)
        label2 = tkinter.Label(root, text=text2)

        label1.grid(row=row, column=column, padx=5, pady=3, sticky='se')
        entry.grid(row=row, column=column+1, padx=5, pady=3, sticky='sew')
        label2.grid(row=row, column=column+2, padx=5, pady=3, sticky='sw')

        self._labels1[key] = label1
        self._entries[key] = entry
        self._labels2[key] = label2

    def state(self, keys):

        """Check whether a subset of the Entry widgets are all valid.

        Args:
            keys:
                An enumerable of the keys of the widgets to be checked.

        Returns:
            tkinter.NORMAL if all the checked widgets are valid, otherwise
            tkinter.DISABLED.
        """

        keys = keys.split(',')
        if all(self._entries[key].value is not None for key in keys):
            return tkinter.NORMAL
        else:
            return tkinter.DISABLED

    def set_state(self, key, state):

        """Set the enabled (greyed-out) state of a named Label/Entry pair.

        Args:
            key:
                The key of the Label/Entry pair whose state is to be set.
            state:
                The new state, typically tkinter.NORMAL or tkinter.DISABLED, to
                be applied to the Label/Entry pair.
        """

        self._labels1[key]['state'] = state
        self._entries[key]['state'] = state
        self._labels2[key]['state'] = state

    def __getitem__(self, key):

        """Return the value of a named Entry widget.

        Args:
            key:
                The name of the Entry widget whose value is to be returned.

        Returns:
            The value of the Entry widget.
        """
        return self._entries[key].value


class GUI:

    """Class representing the GUI."""

    # The keys are both names which will be used to refer to the corresponding
    # parameter in this code, and the parameter name in the call to a1.plot().
    # The labels are just that: labels displayed in the GUI for the parameter.

    keys_1dof = 'm1,k1,l1,f1'   # Params needed for 1 D.O.F calculations
    labels_1dof = 'm₁ =:kg,k₁ =:N/m,λ₁ =:Ns/m,f₁ =:N'

    keys_2dof = 'm2,k2,l2'      # Extra params needed for 2 D.O.F calculations
    labels_2dof = 'm₂ =:kg,k₂ =:N/m,λ₂ =:Ns/m'

    def __init__(self):

        """Construct the GUI and run it."""

        self.params = Params(on_changed=self.enable_disable)

        # Initialise Tk and set window title

        self.root = tkinter.Tk()
        self.root.title('A1')

        # Create matplotlib widgets

        self.fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)

        self.canvas = tkagg.FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()

        self.canvas.get_tk_widget().grid(
            row=1, column=3, rowspan=102,
            padx=5, pady=3, sticky='nsew'
        )

        self.canvas.mpl_connect(
            'resize_event',
            lambda x: self.fig.tight_layout(pad=2.5)
        )

        nav_tbar = tkagg.NavigationToolbar2Tk(
            self.canvas,
            self.root,
            pack_toolbar=False
        )
        nav_tbar.update()

        nav_tbar.grid(row=0, column=3, sticky='nsew')

        # Create all the parameter labels & entry boxes

        self.create_params(
            labels=self.labels_1dof,
            keys=self.keys_1dof,
            first_row=1
        )

        self.enabled_2dof = tkinter.IntVar()
        self.enabled_2dof.trace_add("write", self.enable_disable)

        check_dof2 = tkinter.Checkbutton(
            master=self.root,
            text="2 D.O.F.",
            variable=self.enabled_2dof
        )

        check_dof2.grid(
            row=50, column=0, columnspan=3,
            padx=5, pady=3, sticky='nsw'
        )

        self.create_params(
            labels=self.labels_2dof,
            keys=self.keys_2dof,
            first_row=51
        )

        # Create show phase checkbox

        self.enabled_phase = tkinter.IntVar()

        check_phase = tkinter.Checkbutton(
            master=self.root,
            text="Show phase",
            variable=self.enabled_phase
        )

        check_phase.grid(
            row=99, column=0, columnspan=3,
            padx=5, pady=3, sticky='nsw'
        )

        # Create plot button

        self.button_plot = tkinter.Button(
            master=self.root,
            text="Plot",
            command=self.plot
        )

        self.button_plot.grid(
            row=100, column=0, columnspan=3,
            padx=5, pady=3, sticky='nsew'
        )

        # Create diagram label

        prog_dir = pathlib.PurePath(__file__).parent

        self.diagram_1dof = tkinter.PhotoImage(
            file=prog_dir / 'a1_diagram_1dof.png'
        )
        self.diagram_2dof = tkinter.PhotoImage(
            file=prog_dir / 'a1_diagram_2dof.png'
        )

        self.label_diagram = tkinter.Label(
            master=self.root,
            image=self.diagram_1dof
        )

        self.label_diagram.grid(
            row=101, column=0, columnspan=3,
            padx=5, pady=3, sticky='nsew'
        )

        # Create quit button

        button_quit = tkinter.Button(
            master=self.root,
            text="Quit",
            command=self.root.destroy
        )

        button_quit.grid(
            row=102, column=0, columnspan=3,
            padx=5, pady=3, sticky='nsew'
        )

        # Configure the row/column where resizing applies

        self.root.rowconfigure(101, weight=1)
        self.root.columnconfigure(3, weight=1)

        # And off we go!

        self.enable_disable()
        self.root.mainloop()

    def create_params(self, labels, keys, first_row):

        """Create a set of label/entry pairs for parameters."""

        for row, (label, key) in enumerate(
            zip(
                labels.split(','),
                keys.split(','),
                # strict=True (removed for Python 3.9 compatibility)
            ),
            start=first_row
        ):
            self.params.create_param(self.root, label, key, row, 0)

    def enable_disable(self, *_):

        """Enable or disable (grey-out) widgets as appropriate."""

        if self.enabled_2dof.get() == 0:
            state = self.params.state(self.keys_1dof)
            twostate = tkinter.DISABLED
            self.label_diagram.configure(image=self.diagram_1dof)
        else:
            state = self.params.state(f'{self.keys_1dof},{self.keys_2dof}')
            twostate = tkinter.NORMAL
            self.label_diagram.configure(image=self.diagram_2dof)

        self.button_plot['state'] = state

        for key in self.keys_2dof.split(','):
            self.params.set_state(key, twostate)

    def plot(self):

        """Plot the graphs."""

        if self.enabled_2dof.get() == 0:
            args = self.keys_1dof
            MKLF = a1.MLKF_1dof
            kwargs = {}
        else:
            args = f'{self.keys_1dof},{self.keys_2dof}'
            MKLF = a1.MLKF_2dof
            kwargs = {'f2': 0}

        kwargs |= {arg: self.params[arg] for arg in args.split(',')}
        M, L, K, F = MKLF(**kwargs)

        hz = np.linspace(0, 5, 10001)
        sec = np.linspace(0, 30, 10001)

        if self.enabled_phase.get() == 0:
            phase = None
        elif self.enabled_2dof.get() == 0:
            phase = 0
        else:
            phase = 1

        a1.plot(
            self.fig,
            hz, sec,
            M, L, K, F,
            phase
        )

        self.canvas.draw()


if __name__ == '__main__':
    GUI()
