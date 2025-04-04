import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ttkthemes import ThemedStyle
from PIL import Image, ImageTk
import numpy as np
import logging
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

matplotlib.use('TkAgg')


class PulseVisualizer:
    def __init__(self, root, show_plots=True):
        self.root = root
        self._root_initialized = True  # Track if root is valid
        self.peak_marker = None
        self.tk_image = None
        self.bpm_text = None
        self.root = root
        self.show_plots = show_plots
        self.settings_window = None

        # Default parameters
        self.params = {
            'min_bpm': 40,
            'max_bpm': 180,
            'smoothing': 0.7,
            'theme': 'equilux',
            'regions': {
                'forehead': {'enabled': True, 'weight': 0.8, 'offset_y': -0.35, 'offset_x': 0, 'scale_w': 0.35, 'scale_h': 0.25},
                'left_cheek': {'enabled': True, 'weight': 0.1, 'offset_y': 0, 'offset_x': -0.2, 'scale_w': 0.2, 'scale_h': 0.15},
                'right_cheek': {'enabled': True, 'weight': 0.1, 'offset_y': 0, 'offset_x': 0.2, 'scale_w': 0.2, 'scale_h': 0.15}
            },
            'buffer_size': 250,
            'bandpass_low': 45,
            'bandpass_high': 240,
            'detection_interval': 5
        }

        # Apply the theme
        self.style = ThemedStyle(self.root)
        self.style.set_theme(self.params['theme'])

        # Configure custom colors
        self._configure_colors()
        self._setup_display()
        self._add_settings_button()

        self.save_file = None  # Track the current save file
        self.last_save_time = 0  # For throttling saves
        self.save_interval = 1.0  # Save every 1 second
        self.save_data_var = tk.BooleanVar(value=False)

    def _add_settings_button(self):
        """Add a settings button to the window"""
        settings_frame = ttk.Frame(self.root)
        settings_frame.place(relx=1.0, rely=0.0, anchor='ne', x=-10, y=10)

        self.settings_btn = ttk.Button(
                settings_frame,
                text="⚙",
                command=self._open_settings,
                style='Toolbutton')

        self.settings_btn.pack()

    def _open_settings(self):
        """Open the settings window"""
        if self.settings_window and self.settings_window.winfo_exists():
            self.settings_window.lift()
            return

        self.settings_window = tk.Toplevel(self.root)
        self.settings_window.title("Settings")
        self.settings_window.geometry("760x580")
        self.settings_window.resizable(False, False)

        # Prevent multiple settings windows
        self.settings_window.protocol("WM_DELETE_WINDOW", self._close_settings)

        # Add settings controls
        self._create_settings_controls()

    def _close_settings(self):
        """Close the settings window"""
        if self.settings_window:
            self.settings_window.destroy()
            self.settings_window = None

    def _create_settings_controls(self):
        """Create the settings controls"""
        if not self.settings_window:
            return

        # Create main container with scrollbar
        container = ttk.Frame(self.settings_window)
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        container.pack(fill="both", expand=True)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Main content frame with two columns
        main_frame = ttk.Frame(scrollable_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Column 1:
        col1_frame = ttk.Frame(main_frame)
        col1_frame.grid(row=0, column=0, padx=10, pady=5, sticky='nsew')

        # Column 2:
        col2_frame = ttk.Frame(main_frame)
        col2_frame.grid(row=0, column=1, padx=10, pady=5, sticky='nsew')

        # Column 3:
        col3_frame = ttk.Frame(main_frame)
        col3_frame.grid(row=0, column=2, padx=10, pady=5, sticky='nsew')

        # Processing & Visualization section
        proc_frame = ttk.LabelFrame(col1_frame, text="Processing & Visualization", padding=10)
        proc_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # BPM Range
        ttk.Label(proc_frame, text="BPM Range:").grid(row=0, column=0, sticky='w', pady=5)
        range_frame = ttk.Frame(proc_frame)
        range_frame.grid(row=0, column=1, sticky='ew', pady=5)

        ttk.Label(range_frame, text="Min:").pack(side=tk.LEFT)
        self.min_bpm_var = tk.IntVar(value=self.params['min_bpm'])
        ttk.Spinbox(
            range_frame,
            from_=30,
            to=100,
            textvariable=self.min_bpm_var,
            width=5
        ).pack(side=tk.LEFT, padx=5)

        ttk.Label(range_frame, text="Max:").pack(side=tk.LEFT)
        self.max_bpm_var = tk.IntVar(value=self.params['max_bpm'])
        ttk.Spinbox(
            range_frame,
            from_=60,
            to=250,
            textvariable=self.max_bpm_var,
            width=5
        ).pack(side=tk.LEFT, padx=5)

        # Smoothing Factor
        ttk.Label(proc_frame, text="Smoothing Factor:").grid(row=1, column=0, sticky='w', pady=5)
        self.smoothing_var = tk.DoubleVar(value=self.params['smoothing'])
        ttk.Scale(
            proc_frame,
            from_=0.1,
            to=1.0,
            variable=self.smoothing_var,
            orient=tk.HORIZONTAL
        ).grid(row=1, column=1, sticky='ew', pady=5)

        # Buffer Size
        ttk.Label(proc_frame, text="Buffer Size:").grid(row=2, column=0, sticky='w', pady=5)
        self.buffer_size_var = tk.IntVar(value=self.params['buffer_size'])
        ttk.Spinbox(
            proc_frame,
            from_=50,
            to=1000,
            textvariable=self.buffer_size_var,
            width=7
        ).grid(row=2, column=1, sticky='w', pady=5)

        # Bandpass Filter
        ttk.Label(proc_frame, text="Bandpass Filter (Hz):").grid(row=3, column=0, sticky='w', pady=5)
        bandpass_frame = ttk.Frame(proc_frame)
        bandpass_frame.grid(row=3, column=1, sticky='ew', pady=5)

        ttk.Label(bandpass_frame, text="Low:").pack(side=tk.LEFT)
        self.bandpass_low_var = tk.DoubleVar(value=self.params['bandpass_low'])
        ttk.Spinbox(
            bandpass_frame,
            from_=1,
            to=100,
            increment=1,
            textvariable=self.bandpass_low_var,
            width=5
        ).pack(side=tk.LEFT, padx=5)

        ttk.Label(bandpass_frame, text="High:").pack(side=tk.LEFT)
        self.bandpass_high_var = tk.DoubleVar(value=self.params['bandpass_high'])
        ttk.Spinbox(
            bandpass_frame,
            from_=100,
            to=300,
            increment=1,
            textvariable=self.bandpass_high_var,
            width=5
        ).pack(side=tk.LEFT, padx=5)

        # Theme Selection
        theme_frame = ttk.LabelFrame(col2_frame, text="Theme", padding=10)
        theme_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.theme_var = tk.StringVar(value=self.params['theme'])
        ttk.Radiobutton(
            theme_frame,
            text="Dark",
            variable=self.theme_var,
            value='equilux'
        ).pack(anchor='w', pady=2)
        ttk.Radiobutton(
            theme_frame,
            text="Light",
            variable=self.theme_var,
            value='arc'
        ).pack(anchor='w', pady=2)

        # Detection Interval
        face_frame = ttk.LabelFrame(col2_frame, text="Face Detection Interval (Frames)", padding=10)
        face_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.detection_interval_var = tk.IntVar(value=self.params['detection_interval'])

        ttk.Spinbox(
            face_frame,
            from_=1,
            to=20,
            textvariable=self.detection_interval_var,
            width=7
        ).pack(anchor='w', pady=2)

        # Data Saving
        save_frame = ttk.LabelFrame(col3_frame, text="Data Saving", padding=10)
        save_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        ttk.Checkbutton(
            save_frame,
            text="Enable data saving to TXT",
            variable=self.save_data_var
        ).pack(anchor='w', pady=2)

        # Forehead Settings
        self.forehead_enabled_var = tk.BooleanVar(value=self.params['regions']['forehead']['enabled'])
        forehead_frame = ttk.LabelFrame(col1_frame, text="Forehead Region", padding=10)
        forehead_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        ttk.Checkbutton(
            forehead_frame,
            text="Enable",
            variable=self.forehead_enabled_var
        ).pack(anchor='w', pady=2)

        settings_grid = ttk.Frame(forehead_frame)
        settings_grid.pack(fill=tk.BOTH, expand=True)

        ttk.Label(settings_grid, text="Weight:").grid(row=0, column=0, sticky='e', padx=5, pady=2)
        self.forehead_weight_var = tk.DoubleVar(value=self.params['regions']['forehead']['weight'])
        ttk.Spinbox(
            settings_grid,
            from_=0.0,
            to=1.0,
            increment=0.05,
            textvariable=self.forehead_weight_var,
            width=5
        ).grid(row=0, column=1, sticky='w', pady=2)

        ttk.Label(settings_grid, text="X Offset:").grid(row=1, column=0, sticky='e', padx=5, pady=2)
        self.forehead_offset_x_var = tk.DoubleVar(value=self.params['regions']['forehead']['offset_x'])
        ttk.Spinbox(
            settings_grid,
            from_=-0.5,
            to=0.5,
            increment=0.05,
            textvariable=self.forehead_offset_x_var,
            width=5
        ).grid(row=1, column=1, sticky='w', pady=2)

        ttk.Label(settings_grid, text="Y Offset:").grid(row=2, column=0, sticky='e', padx=5, pady=2)
        self.forehead_offset_y_var = tk.DoubleVar(value=self.params['regions']['forehead']['offset_y'])
        ttk.Spinbox(
            settings_grid,
            from_=-1.0,
            to=1.0,
            increment=0.05,
            textvariable=self.forehead_offset_y_var,
            width=5
        ).grid(row=2, column=1, sticky='w', pady=2)

        ttk.Label(settings_grid, text="Width Scale:").grid(row=3, column=0, sticky='e', padx=5, pady=2)
        self.forehead_scale_w_var = tk.DoubleVar(value=self.params['regions']['forehead']['scale_w'])
        ttk.Spinbox(
            settings_grid,
            from_=-1.0,
            to=1.0,
            increment=0.05,
            textvariable=self.forehead_scale_w_var,
            width=5
        ).grid(row=3, column=1, sticky='w', pady=2)

        ttk.Label(settings_grid, text="Height Scale:").grid(row=4, column=0, sticky='e', padx=5, pady=2)
        self.forehead_scale_h_var = tk.DoubleVar(value=self.params['regions']['forehead']['scale_h'])
        ttk.Spinbox(
            settings_grid,
            from_=0.1,
            to=1.0,
            increment=0.05,
            textvariable=self.forehead_scale_h_var,
            width=5
        ).grid(row=4, column=1, sticky='w', pady=2)

        # Left Cheek Settings
        self.left_cheek_enabled_var = tk.BooleanVar(value=self.params['regions']['left_cheek']['enabled'])
        left_cheek_frame = ttk.LabelFrame(col2_frame, text="Left Cheek Region", padding=10)
        left_cheek_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        ttk.Checkbutton(
            left_cheek_frame,
            text="Enable",
            variable=self.left_cheek_enabled_var
        ).pack(anchor='w', pady=2)

        settings_grid = ttk.Frame(left_cheek_frame)
        settings_grid.pack(fill=tk.BOTH, expand=True)

        ttk.Label(settings_grid, text="Weight:").grid(row=0, column=0, sticky='e', padx=5, pady=2)
        self.left_cheek_weight_var = tk.DoubleVar(value=self.params['regions']['left_cheek']['weight'])
        ttk.Spinbox(
            settings_grid,
            from_=0.0,
            to=1.0,
            increment=0.05,
            textvariable=self.left_cheek_weight_var,
            width=5
        ).grid(row=0, column=1, sticky='w', pady=2)

        ttk.Label(settings_grid, text="X Offset:").grid(row=1, column=0, sticky='e', padx=5, pady=2)
        self.left_cheek_offset_x_var = tk.DoubleVar(value=self.params['regions']['left_cheek']['offset_x'])
        ttk.Spinbox(
            settings_grid,
            from_=-0.5,
            to=0.5,
            increment=0.05,
            textvariable=self.left_cheek_offset_x_var,
            width=5
        ).grid(row=1, column=1, sticky='w', pady=2)

        ttk.Label(settings_grid, text="Y Offset:").grid(row=2, column=0, sticky='e', padx=5, pady=2)
        self.left_cheek_offset_y_var = tk.DoubleVar(value=self.params['regions']['left_cheek']['offset_y'])
        ttk.Spinbox(
            settings_grid,
            from_=-1.0,
            to=1.0,
            increment=0.05,
            textvariable=self.left_cheek_offset_y_var,
            width=5
        ).grid(row=2, column=1, sticky='w', pady=2)

        ttk.Label(settings_grid, text="Width Scale:").grid(row=3, column=0, sticky='e', padx=5, pady=2)
        self.left_cheek_scale_w_var = tk.DoubleVar(value=self.params['regions']['left_cheek']['scale_w'])
        ttk.Spinbox(
            settings_grid,
            from_=0.1,
            to=1.0,
            increment=0.05,
            textvariable=self.left_cheek_scale_w_var,
            width=5
        ).grid(row=3, column=1, sticky='w', pady=2)

        ttk.Label(settings_grid, text="Height Scale:").grid(row=4, column=0, sticky='e', padx=5, pady=2)
        self.left_cheek_scale_h_var = tk.DoubleVar(value=self.params['regions']['left_cheek']['scale_h'])
        ttk.Spinbox(
            settings_grid,
            from_=0.1,
            to=1.0,
            increment=0.05,
            textvariable=self.left_cheek_scale_h_var,
            width=5
        ).grid(row=4, column=1, sticky='w', pady=2)

        # Right Cheek Settings
        self.right_cheek_enabled_var = tk.BooleanVar(value=self.params['regions']['right_cheek']['enabled'])
        right_cheek_frame = ttk.LabelFrame(col3_frame, text="Right Cheek Region", padding=10)
        right_cheek_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        ttk.Checkbutton(
            right_cheek_frame,
            text="Enable",
            variable=self.right_cheek_enabled_var
        ).pack(anchor='w', pady=2)

        settings_grid = ttk.Frame(right_cheek_frame)
        settings_grid.pack(fill=tk.BOTH, expand=True)

        ttk.Label(settings_grid, text="Weight:").grid(row=0, column=0, sticky='e', padx=5, pady=2)
        self.right_cheek_weight_var = tk.DoubleVar(value=self.params['regions']['right_cheek']['weight'])
        ttk.Spinbox(
            settings_grid,
            from_=0.0,
            to=1.0,
            increment=0.05,
            textvariable=self.right_cheek_weight_var,
            width=5
        ).grid(row=0, column=1, sticky='w', pady=2)

        ttk.Label(settings_grid, text="X Offset:").grid(row=1, column=0, sticky='e', padx=5, pady=2)
        self.right_cheek_offset_x_var = tk.DoubleVar(value=self.params['regions']['right_cheek']['offset_x'])
        ttk.Spinbox(
            settings_grid,
            from_=-0.5,
            to=0.5,
            increment=0.05,
            textvariable=self.right_cheek_offset_x_var,
            width=5
        ).grid(row=1, column=1, sticky='w', pady=2)

        ttk.Label(settings_grid, text="Y Offset:").grid(row=2, column=0, sticky='e', padx=5, pady=2)
        self.right_cheek_offset_y_var = tk.DoubleVar(value=self.params['regions']['right_cheek']['offset_y'])
        ttk.Spinbox(
            settings_grid,
            from_=-1.0,
            to=1.0,
            increment=0.05,
            textvariable=self.right_cheek_offset_y_var,
            width=5
        ).grid(row=2, column=1, sticky='w', pady=2)

        ttk.Label(settings_grid, text="Width Scale:").grid(row=3, column=0, sticky='e', padx=5, pady=2)
        self.right_cheek_scale_w_var = tk.DoubleVar(value=self.params['regions']['right_cheek']['scale_w'])
        ttk.Spinbox(
            settings_grid,
            from_=0.1,
            to=1.0,
            increment=0.05,
            textvariable=self.right_cheek_scale_w_var,
            width=5
        ).grid(row=3, column=1, sticky='w', pady=2)

        ttk.Label(settings_grid, text="Height Scale:").grid(row=4, column=0, sticky='e', padx=5, pady=2)
        self.right_cheek_scale_h_var = tk.DoubleVar(value=self.params['regions']['right_cheek']['scale_h'])
        ttk.Spinbox(
            settings_grid,
            from_=0.1,
            to=1.0,
            increment=0.05,
            textvariable=self.right_cheek_scale_h_var,
            width=5
        ).grid(row=4, column=1, sticky='w', pady=2)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=3, pady=10)

        ttk.Button(
            button_frame,
            text="Apply",
            command=self._apply_settings
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Cancel",
            command=self._close_settings
        ).pack(side=tk.LEFT, padx=5)

        # Configure column weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

    def _apply_settings(self):
        """Apply the selected settings"""
        bandpass_low = self.bandpass_low_var.get()
        bandpass_high = self.bandpass_high_var.get()

        if bandpass_low >= bandpass_high:
            messagebox.showerror("Invalid Settings", "Bandpass low must be less than high")
            return

        self.params = {
            'min_bpm': self.min_bpm_var.get(),
            'max_bpm': self.max_bpm_var.get(),
            'smoothing': self.smoothing_var.get(),
            'buffer_size': self.buffer_size_var.get(),
            'bandpass_low': self.bandpass_low_var.get(),
            'bandpass_high': self.bandpass_high_var.get(),
            'detection_interval': self.detection_interval_var.get(),
            'theme': self.theme_var.get(),
            'regions': {
                'forehead': {
                    'enabled': self.forehead_enabled_var.get(),
                    'weight': self.forehead_weight_var.get(),
                    'offset_y': self.forehead_offset_y_var.get(),
                    'offset_x': self.forehead_offset_x_var.get(),
                    'scale_w': self.forehead_scale_w_var.get(),
                    'scale_h': self.forehead_scale_h_var.get()
                },
                'left_cheek': {
                    'enabled': self.left_cheek_enabled_var.get(),
                    'weight': self.left_cheek_weight_var.get(),
                    'offset_y': self.left_cheek_offset_y_var.get(),
                    'offset_x': self.left_cheek_offset_x_var.get(),
                    'scale_w': self.left_cheek_scale_w_var.get(),
                    'scale_h': self.left_cheek_scale_h_var.get()
                },
                'right_cheek': {
                    'enabled': self.right_cheek_enabled_var.get(),
                    'weight': self.right_cheek_weight_var.get(),
                    'offset_y': self.right_cheek_offset_y_var.get(),
                    'offset_x': self.right_cheek_offset_x_var.get(),
                    'scale_w': self.right_cheek_scale_w_var.get(),
                    'scale_h': self.right_cheek_scale_h_var.get()
                }
            }
        }

        # Update FFT plot range
        if hasattr(self, 'fft_ax'):
            self.fft_ax.set_xlim(self.params['min_bpm'], self.params['max_bpm'])
            self.fft_canvas.draw()

        # Update theme if changed
        if self.style.theme_use() != self.params['theme']:
            self.style.set_theme(self.params['theme'])
            self._configure_colors()

            # Handle data saving if enabled
        if self.save_data_var.get() and not self.save_file:
            default_filename = f"pulse_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.save_file = filedialog.asksaveasfilename(
                defaultextension=".csv",
                initialfile=default_filename,
                filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
            )
            if self.save_file:
                # Write header if file is new
                if not os.path.exists(self.save_file):
                    with open(self.save_file, 'w') as f:
                        f.write("timestamp,bpm\n")
        elif not self.save_data_var.get() and self.save_file:
            self.save_file = None

        self._close_settings()

    def _configure_colors(self):
        """Configure colors that work well with the Equilux theme"""
        # Get colors from the theme
        bg_color = self.style.lookup('TFrame', 'background')
        fg_color = self.style.lookup('TLabel', 'foreground')

        # Matplotlib colors that match the theme
        self.plot_bg = '#2e2e2e'  # Slightly darker than Equilux default
        self.plot_fg = '#e0e0e0'  # Light text
        self.plot_grid = '#424242'  # Grid color
        self.signal_color = '#4e9af5'  # Blue signal line
        self.fft_color = '#4ef5a2'  # Green FFT line

        # Configure matplotlib style
        plt.style.use('dark_background')
        matplotlib.rcParams.update({
            'axes.facecolor': self.plot_bg,
            'figure.facecolor': bg_color,
            'axes.edgecolor': self.plot_fg,
            'axes.labelcolor': self.plot_fg,
            'text.color': self.plot_fg,
            'xtick.color': self.plot_fg,
            'ytick.color': self.plot_fg,
            'grid.color': self.plot_grid
        })

    def _setup_display(self):
        """Initialize the Tkinter display layout with Equilux theme"""
        self.root.title("Pulse Detector")
        self.root.geometry("1200x750")

        # Configure root window background
        self.root.configure(bg=self.style.lookup('TFrame', 'background'))

        # Create main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Webcam view (left side)
        self.cam_frame = ttk.Frame(self.main_frame)
        self.cam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.cam_label = ttk.Label(self.cam_frame)
        self.cam_label.pack(fill=tk.BOTH, expand=True)

        # Initialize with blank image
        blank_image = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
        self.tk_image = ImageTk.PhotoImage(image=blank_image)
        self.cam_label.config(image=self.tk_image)

        if self.show_plots:
            # Right side container for plots
            self.plots_frame = ttk.Frame(self.main_frame)
            self.plots_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Signal plot (top)
            self.signal_frame = ttk.Frame(self.plots_frame)
            self.signal_frame.pack(fill=tk.BOTH, expand=True)

            self.signal_fig = Figure(figsize=(6, 3), dpi=100, facecolor=self.plot_bg)
            self.signal_ax = self.signal_fig.add_subplot(111)
            self.signal_ax.set_title('Pulse Signal', pad=10, fontsize=12, color=self.plot_fg)
            self.signal_ax.grid(True, linestyle='--', alpha=0.6, color=self.plot_grid)
            self.signal_ax.set_facecolor(self.plot_bg)
            self.signal_line, = self.signal_ax.plot([], [], color=self.signal_color, linewidth=1.5)
            self.signal_ax.set_xlabel('Time (s)', fontsize=10, color=self.plot_fg)
            self.signal_ax.set_ylabel('Intensity', fontsize=10, color=self.plot_fg)
            self.signal_ax.tick_params(colors=self.plot_fg, labelsize=9)

            self.signal_canvas = FigureCanvasTkAgg(self.signal_fig, master=self.signal_frame)
            self.signal_canvas.draw()
            self.signal_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # FFT plot (bottom)
            self.fft_frame = ttk.Frame(self.plots_frame)
            self.fft_frame.pack(fill=tk.BOTH, expand=True)

            self.fft_fig = Figure(figsize=(6, 3), dpi=100, facecolor=self.plot_bg)
            self.fft_ax = self.fft_fig.add_subplot(111)
            self.fft_ax.set_title('Frequency Spectrum (BPM)', pad=10, fontsize=12, color=self.plot_fg)
            self.fft_ax.grid(True, which='both', linestyle='--', alpha=0.6, color=self.plot_grid)
            self.fft_ax.set_facecolor(self.plot_bg)
            self.fft_line, = self.fft_ax.plot([], [], color=self.fft_color, linewidth=1.5)
            self.fft_ax.set_xlim(self.params['min_bpm'], self.params['max_bpm'])
            self.fft_ax.set_xlabel('BPM', fontsize=10, color=self.plot_fg)
            self.fft_ax.set_ylabel('Amplitude', fontsize=10, color=self.plot_fg)
            self.fft_ax.tick_params(colors=self.plot_fg, labelsize=9)
            self.fft_ax.grid(True, which='minor', linestyle=':', alpha=0.4, color=self.plot_grid)
            self.fft_ax.minorticks_on()

            self.fft_canvas = FigureCanvasTkAgg(self.fft_fig, master=self.fft_frame)
            self.fft_canvas.draw()
            self.fft_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Status bar
            self.status_var = tk.StringVar()
            self.status_var.set("Ready")
            self.status_bar = ttk.Label(self.root, textvariable=self.status_var,
                                        relief=tk.SUNKEN, padding=(5, 5))
            self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_plots(self, processor):
        """Update all visualizations"""
        try:
            # Update webcam view
            img = Image.fromarray(processor.frame_out)
            self.tk_image = ImageTk.PhotoImage(image=img)
            self.cam_label.config(image=self.tk_image)

            if self.show_plots and hasattr(processor, 'regions'):
                # Update signal plot
                if (processor.regions['forehead']['values'] and
                        hasattr(processor, 'times') and
                        len(processor.times) >= len(processor.regions['forehead']['values'])):
                    times = np.array(processor.times)
                    signal = np.array(processor.regions['forehead']['values'])
                    signal = signal - np.mean(signal)  # Remove DC

                    self.signal_line.set_data(times[-len(signal):] - times[0], signal)
                    self.signal_ax.relim()
                    self.signal_ax.autoscale_view()
                    self.signal_canvas.draw()

                # Update FFT plot
                if (hasattr(processor, 'freqs') and
                        hasattr(processor, 'fft') and
                        len(processor.freqs) > 0 and
                        len(processor.fft) > 0):

                    bpm = processor.freqs * 60
                    mask = (bpm >= self.params['min_bpm']) & (bpm <= self.params['max_bpm'])

                    if np.any(mask):
                        self.fft_line.set_data(bpm[mask], processor.fft[mask])
                        self.fft_ax.relim()
                        self.fft_ax.autoscale_view()

                        if (processor.bpm > self.params['min_bpm'] and
                                hasattr(processor, 'snr') and
                                len(bpm[mask]) > 0 and
                                len(processor.fft[mask]) > 0):

                            peak_idx = np.argmin(np.abs(bpm[mask] - processor.bpm))
                            if peak_idx < len(bpm[mask]) and peak_idx < len(processor.fft[mask]):
                                # Remove previous peak marker if it exists
                                if hasattr(self, 'peak_marker') and self.peak_marker is not None:
                                    self.peak_marker.remove()
                                self.peak_marker, = self.fft_ax.plot(
                                    bpm[mask][peak_idx],
                                    processor.fft[mask][peak_idx],
                                    'ro', markersize=8)

                                # Remove previous text if it exists
                                if hasattr(self, 'bpm_text') and self.bpm_text is not None:
                                    self.bpm_text.remove()

                                # Get current axis limits
                                xlim = self.fft_ax.get_xlim()
                                ylim = self.fft_ax.get_ylim()

                                # Calculate text position with boundary checking
                                text_x = processor.bpm + 5
                                text_y = processor.fft[mask][peak_idx]

                                # Adjust position if it would go beyond x-axis limits
                                if text_x > xlim[1] - 15:  # Leave some margin (15 units)
                                    text_x = processor.bpm - 15  # Move to left side of peak

                                    # If still beyond left limit, place at edge
                                    if text_x < xlim[0] + 5:
                                        text_x = xlim[0] + 5

                                # Adjust position if it would go beyond y-axis limits
                                if text_y > ylim[1] * 0.9:  # If near top, move below peak
                                    text_y = processor.fft[mask][peak_idx] * 0.8

                                    # If still beyond bottom limit, place at edge
                                    if text_y < ylim[0] + (ylim[1] - ylim[0]) * 0.1:
                                        text_y = ylim[0] + (ylim[1] - ylim[0]) * 0.1

                                # Create the text with adjusted position
                                self.bpm_text = self.fft_ax.text(
                                    text_x,
                                    text_y,
                                    f'{processor.bpm:.1f} BPM\nSNR: {processor.snr:.1f}',
                                    color='#3a3a3a',
                                    fontsize=10,
                                    bbox=dict(facecolor='white', alpha=0.7)
                                )
                        self.fft_canvas.draw()

            # Update status
            if hasattr(processor, 'bpm'):
                status = f"Current BPM: {processor.bpm:.1f}" if processor.bpm > self.params['min_bpm'] else "Finding pulse..."
                self.status_var.set(status)

            if self.save_data_var.get() and hasattr(processor, 'bpm'):
                self._save_data_to_file(processor)

            self.root.update()

        except Exception as e:
            logging.error(f"Visualization error: {e}")

    def _save_data_to_file(self, processor):
        """Save the current data to a text file"""
        if not self.save_file or not hasattr(processor, 'bpm'):
            return

        current_time = time.time()
        if current_time - self.last_save_time < self.save_interval:
            return

        try:
            with open(self.save_file, 'a') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                f.write(f"{timestamp},{processor.bpm:.1f}\n")
            self.last_save_time = current_time
        except Exception as e:
            logging.error(f"Error saving data: {e}")

    def close(self):
        """Clean up resources"""
        try:
            if self.root and self.root.winfo_exists():  # Check if window still exists
                self.root.destroy()
        except tk.TclError:
            pass  # Window already destroyed
        finally:
            self.root = None  # Ensure reference is cleared
