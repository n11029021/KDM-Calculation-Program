import tkinter as tk
from collections import defaultdict  
from tkinter import filedialog, simpledialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
import csv
import time
from scipy.signal import find_peaks, savgol_filter
import pandas as pd
import re
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading

class FolderMonitorHandler(FileSystemEventHandler):
    def __init__(self, process_file_callback):
        self.process_file_callback = process_file_callback

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.txt'):
            self.process_file_callback(event.src_path)

class WaveHeightExtractionTool:
    def __init__(self, root):
        self.root = root
        self.sampling_rate = 1000
        self.cutoff_freq = 30
        self.signal_data = defaultdict(lambda: defaultdict(list))  # Store wave heights for Signal-ON and Signal-OFF
        self.average_data = defaultdict(lambda: defaultdict(dict))  # Store averaged wave heights for KDM calculations
        self.kdm_results = {
            "Baseline": [],
            "Lower Limit": [],
            "Upper Limit": [],
            "Monitor": []
        }
        self.upper_limit = float(str(400))
        self.lower_limit = float(str(151))
        self.num_trials = 3  # Default number of trials, will be updated by user input
        self.file_name_label = tk.Label(root, text="No file selected.")
        self.file_name_label.pack(pady=10)
        self.stage = 0  # 0: Baseline, 1: Upper/Lower Limit, 2: Monitor
        self.baseline_data = defaultdict(lambda: defaultdict(list))
        self.upplow_data = defaultdict(lambda: defaultdict(list))
        self.monitor_data = defaultdict(lambda: defaultdict(list))
        self.channel_calibration = {}  # Store calibration data for channels
        self.setup_gui()

    def setup_gui(self):
        open_file_button = tk.Button(self.root, text="Open File", command=self.open_file)
        open_file_button.pack(pady=5)

        extract_heights_button = tk.Button(self.root, text="Extract Wave Heights", command=self.extract_wave_heights_from_files)
        extract_heights_button.pack(pady=5)

        calculate_kdm_button = tk.Button(self.root, text="Calculate KDM", command=self.calculate_kdm)
        calculate_kdm_button.pack(pady=5)

        calculate_signal_change_button = tk.Button(
            self.root, text="Calculate % Signal Change from Files", command=self.calculate_signal_change_from_files
        )
        calculate_signal_change_button.pack(pady=5)

        monitor_folder_button = tk.Button(self.root, text="Monitor Folder", command=self.monitor_folder)
        monitor_folder_button.pack(pady=5)

        quit_button = tk.Button(self.root, text="Quit", command=self.root.quit)
        quit_button.pack(pady=5)

    def extract_time(self, file_name):
        """Extract numeric value before the first '-' in the file name."""
        match = re.match(r'^(\d+)', file_name)  # Match all digits before the first '-'
        return match.group(1).zfill(6) if match else "999999"  # Return zero-padded time or a large default value

    def get_baseline_i_min(self, channel, signal_type):
        """Get the minimum i_min value for the Baseline stage."""
        baseline_data = self.baseline_data[f"Channel_{channel}"][signal_type]
        if not baseline_data:
            return None
        # Sort by time and get the average height for the smallest time
        sorted_data = sorted(baseline_data, key=lambda x: self.extract_time(x[0]))  # Sort by time
        smallest_time = self.extract_time(sorted_data[0][0])  # Get the smallest time
        return np.mean([height for file_name, height in sorted_data if self.extract_time(file_name) == smallest_time])

    def get_latest_baseline_i_min(self, channel, signal_type):
        """Get the latest i_min value for the Baseline stage."""
        baseline_data = self.baseline_data[f"Channel_{channel}"][signal_type]
        if not baseline_data:
            return None
        # Sort by time and get the average height for the latest time
        sorted_data = sorted(baseline_data, key=lambda x: self.extract_time(x[0]))  # Sort by time
        latest_time = self.extract_time(sorted_data[-1][0])  # Get the latest time
        return np.mean([height for file_name, height in sorted_data if self.extract_time(file_name) == latest_time])
    
    def savgol_smoothing(self, data, window_length=10, polyorder=1):
        return savgol_filter(data, window_length, polyorder)

    def calculate_wave_heights(self, diff, is_smoothed):
        def find_wave_heights(diff, peaks, troughs):
            if not peaks.size or not troughs.size:
                return None, None, None, None, None, None
            largest_height = None
            largest_type = None
            peak_before_trough = None
            peak_after_trough = None
            trough_x = None
            for trough in troughs:
                peaks_before_trough = peaks[peaks < trough]
                peaks_after_trough = peaks[peaks > trough]
                valid_peaks_exist = (
                    peaks_before_trough.size > 0 and abs(peaks_before_trough[-1] - trough) > 20 and
                    peaks_after_trough.size > 0 and abs(peaks_after_trough[0] - trough) > 20
                )

                # Only run the "too close" check if no valid peaks exist
                if not valid_peaks_exist:
                    x = 10  # Number of points to consider for calculating the mean
                    start_avg = np.mean(diff[:x])
                    end_avg = np.mean(diff[-x:])
                    start_peak_index = np.argmin(np.abs(diff[:x] - start_avg))
                    end_peak_index = len(diff) - x + np.argmin(np.abs(diff[-x:] - end_avg))
                    peaks2 = np.array([start_peak_index, end_peak_index])
                    peaks_before_trough = peaks2[peaks2 < trough]
                    peaks_after_trough = peaks2[peaks2 > trough]

                if peaks_before_trough.size > 0 and peaks_after_trough.size > 0:
                    peak_before_trough_candidate = peaks_before_trough[-1]
                    peak_after_trough_candidate = peaks_after_trough[0]

                    if peak_before_trough_candidate < trough < peak_after_trough_candidate:
                        average_peak_height = (diff[peak_before_trough_candidate] + diff[peak_after_trough_candidate]) / 2
                        height_difference = abs(average_peak_height - diff[trough])
                        if largest_height is None or height_difference > largest_height:
                            largest_height = height_difference
                            largest_type = 'trough' if diff[peak_before_trough_candidate] - diff[trough] > 0 else 'peak'
                            peak_before_trough = peak_before_trough_candidate
                            peak_after_trough = peak_after_trough_candidate
                            trough_x = trough
            if largest_height is not None:
                if largest_type == 'peak':
                    max_wave_height = largest_height
                else:
                    max_wave_height = -largest_height
                return max_wave_height, peaks, troughs, peak_before_trough, peak_after_trough, trough_x
            else:
                return None, None, None, None, None, None

        def local_maxima(xval, yval):
            xval = np.asarray(xval)
            yval = np.asarray(yval)

            sort_idx = np.argsort(xval)
            yval = yval[sort_idx]
            gradient = np.diff(yval)
            maxima = np.diff((gradient > 0).view(np.int8))
            return np.concatenate((([0],) if gradient[0] < 0 else ()) +
                                (np.where(maxima == -1)[0] + 1,) +
                                (([len(yval)-1],) if gradient[-1] > 0 else ()))

        # Use local_maxima to identify peaks
        peaks = local_maxima(range(len(diff)), diff)
        
        # Identify troughs
        troughs, _ = find_peaks(-diff)

        # Calculate wave heights
        largest_height, peaks, troughs, peak_before_trough, peak_after_trough, trough_x = find_wave_heights(diff, peaks, troughs)

        if largest_height is None:
            print("No significant peaks or troughs detected.")
            return None, None, None, None

        return largest_height, peak_before_trough, peak_after_trough, trough_x

    def plot_data(self, potential, diff, smoothed_diff=None, wave_height=None, peak_before_trough=None, peak_after_trough=None, trough_x=None, title="", is_smoothed=False):
        fig, ax = plt.subplots()
        # Plot only the relevant data
        if is_smoothed and smoothed_diff is not None:
            ax.plot(potential, smoothed_diff, linestyle='--', color='red', label='Smoothed Data')
        else:
            ax.plot(potential, diff, linestyle='-', label='Original Data', color='blue')
        # Add wave height markers if available
        if peak_before_trough is not None and peak_after_trough is not None and trough_x is not None:
            data_to_plot = smoothed_diff if is_smoothed else diff
            ax.plot(potential[peak_before_trough], data_to_plot[peak_before_trough], 'rx', markersize=10, label='Peak Before Trough')
            ax.plot(potential[peak_after_trough], data_to_plot[peak_after_trough], 'rx', markersize=10, label='Peak After Trough')
            ax.plot(potential[trough_x], data_to_plot[trough_x], 'bx', markersize=10, label='Trough')
        # Display wave height information on the side
        if wave_height is not None:
            wave_heights_info = f'Ip = {wave_height:.4e} A'
            ax.text(1.05, 0.5, wave_heights_info, transform=ax.transAxes, fontsize=10,
                    verticalalignment='center', bbox=dict(boxstyle='round', alpha=0.5))
        ax.set_xlabel('Potential/V')
        ax.set_ylabel('Diff(i/A)')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.set_xlim(max(potential), min(potential))  # Reverse the x-axis direction
        plt.subplots_adjust(right=0.75)
        return fig

    def plot_raw_data(self, potential, diff):
        def smooth_data():
            smoothed_diff = self.savgol_smoothing(np.array(diff), window_length=10, polyorder=1)
            wave_height, peak_before_trough, peak_after_trough, trough_x = self.calculate_wave_heights(smoothed_diff, True)
            raw_window.destroy()
            smoothed_window = tk.Toplevel(self.root)
            smoothed_window.title("Smoothed Data Plot")
            # Plot only smoothed data
            fig = self.plot_data(
                np.array(potential), 
                None,  # Do not pass unsmoothed data
                smoothed_diff,  # Smoothed data
                wave_height, 
                peak_before_trough, 
                peak_after_trough, 
                trough_x, 
                "Smoothed Data Plot", 
                is_smoothed=True
            )
            canvas = FigureCanvasTkAgg(fig, master=smoothed_window)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)
            quit_button = tk.Button(smoothed_window, text="Quit", command=self.root.quit)
            quit_button.pack(side=tk.BOTTOM, padx=5, pady=5)
            canvas.draw()

        def quit_app():
            self.root.quit()

        global raw_window
        raw_window = tk.Toplevel(self.root)
        raw_window.title("Raw Data Plot")
        # Plot only raw data
        fig = self.plot_data(np.array(potential), np.array(diff), title="Raw Data Plot", is_smoothed=False)
        canvas = FigureCanvasTkAgg(fig, master=raw_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        smooth_button = tk.Button(raw_window, text="Smooth Data", command=smooth_data)
        smooth_button.pack(side=tk.LEFT, padx=5, pady=5)
        quit_button = tk.Button(raw_window, text="Quit", command=quit_app)
        quit_button.pack(side=tk.RIGHT, padx=5, pady=5)
        canvas.draw()

    def open_file(self):
        file_path = filedialog.askopenfilename(
            initialdir=".",
            title="Select the Text File",
            filetypes=(("Text Files", "*.txt"), ("All Files", "*.*"))
        )
        if file_path:
            file_name = os.path.basename(file_path)
            self.file_name_label.config(text=f"File selected: {file_name}")
            channel_input = simpledialog.askstring("Input", "Please enter the channel number (starting from 1):")
            try:
                channel_number = int(channel_input)
            except ValueError:
                messagebox.showerror("Error", "Invalid channel number. Please enter a valid number.")
                return
            with open(file_path, 'r') as file:
                lines = file.readlines()
            try:
                start_idx = next(i for i, line in enumerate(lines) if line.startswith('Potential/V'))
            except StopIteration:
                messagebox.showerror("Error", f"File {file_name} has unexpected axis titles.")
                return
            data_lines = lines[start_idx + 1:]
            potential, diff = [], []
            for line in data_lines:
                parts = line.split(',')
                if len(parts) >= 2 + (3 * (channel_number - 1)):
                    try:
                        potential.append(float(parts[0].strip()))
                        diff.append(float(parts[1 + (3 * (channel_number - 1))].strip()))
                    except ValueError:
                        continue
            if potential and diff:
                self.plot_raw_data(np.array(potential), np.array(diff))

    def extract_wave_heights_from_files(self):
        def process_files(folder_path, signal_type, num_channels, raw_data, combined_data):
            # Identify all .txt files in the selected folder
            all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.txt')]

            if not all_files:
                print(f"No .txt files found in the selected folder for {signal_type}. Exiting.")
                return

            wave_height_data = defaultdict(lambda: defaultdict(list))
            processed_files = set()  # Track processed files to avoid reprocessing

            # Process all files
            for file_path in all_files:
                if file_path in processed_files:
                    continue  # Skip already processed files
                processed_files.add(file_path)

                file_name = os.path.basename(file_path)
                base_name, _ = os.path.splitext(file_name)
                concentration = "-".join(base_name.split('-')[:-1]).strip().lower()  # Normalize concentration
                try:
                    with open(file_path, 'r') as file:
                        lines = file.readlines()
                    try:
                        start_idx = next(i for i, line in enumerate(lines) if line.startswith('Potential/V'))
                    except StopIteration:
                        print(f"Failed to extract data from {file_name}: Axis Titles not as expected. Expected Titles: Potential/V, Diff(i/A), For(i/A), Rev(i/A)")
                        continue
                    data_lines = lines[start_idx + 1:]
                    for channel in range(num_channels):
                        potential, diff = [], []
                        column_index = 1 + (3 * channel)
                        for line in data_lines:
                            parts = line.split(',')
                            if len(parts) > column_index:
                                try:
                                    potential_value = float(parts[0].strip())
                                    diff_value = float(parts[column_index].strip())
                                    potential.append(potential_value)
                                    diff.append(diff_value)
                                except ValueError:
                                    print(f"Failed to parse line in {file_path} for channel {channel + 1}: {line.strip()}")
                                    continue
                        if potential and diff:
                            smoothed_diff = self.savgol_smoothing(np.array(diff), window_length=10, polyorder=1)
                            max_wave_height = self.calculate_wave_heights(smoothed_diff, True)[0]
                            if max_wave_height is not None:
                                wave_height_data[f"Channel_{channel + 1}"][concentration].append(max_wave_height)
                                raw_data.append([file_name, channel + 1, concentration, max_wave_height, signal_type])
                            else:
                                print(f"Failed to extract data from {file_path} (Channel {channel + 1}): No valid wave heights found.")
                        else:
                            print(f"Failed to extract data from {file_path} (Channel {channel + 1}): No valid data found.")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

            # Process wave height data and avoid duplicates
            for channel in range(num_channels):
                channel_key = f"Channel_{channel + 1}"
                sorted_wave_heights = sorted(
                    wave_height_data[channel_key].items(),
                    key=lambda x: x[0]  # Sort by concentration string
                )
                for concentration, wave_heights in sorted_wave_heights:
                    if len(wave_heights) == self.num_trials:  # Use user-specified number of trials
                        avg_wave_height = np.mean(wave_heights)
                        # Avoid duplicates in combined_data
                        if not any(row[0] == f"{concentration}-avg" and row[2] == channel + 1 and row[3] == signal_type for row in combined_data):
                            combined_data.append([f"{concentration}-avg", avg_wave_height, channel + 1, signal_type])
                    else:
                        print(f"Incomplete data for {concentration} in Channel {channel + 1}.")

        def extract_frequency(part):
            """Extract numeric frequency from a string like '600Hz' or '900Hz'. Return None if no frequency is found."""
            match = re.search(r'(\d+)\s*Hz', part, re.IGNORECASE)
            return int(match.group(1)) if match else None

        # Prompt user for the number of trials
        self.num_trials = simpledialog.askinteger("Input", "Enter the number of trials per concentration and frequency:")
        if not self.num_trials:
            print("No number of trials provided. Exiting.")
            return

        # Process Signal-ON files
        folder_path = filedialog.askdirectory(
            initialdir=".",
            title="Select Folder Containing Signal-ON Files"
        )
        if not folder_path:
            print("No folder selected. Exiting.")
            return

        num_channels = simpledialog.askinteger("Input", "Enter the number of channels to extract:")
        if not num_channels:
            print("No number of channels provided. Exiting.")
            return

        raw_data = []
        combined_data = []
        process_files(folder_path, "ON", num_channels, raw_data, combined_data)

        # Process Signal-OFF files
        folder_path = filedialog.askdirectory(
            initialdir=".",
            title="Select Folder Containing Signal-OFF Files"
        )
        if not folder_path:
            print("No folder selected. Exiting.")
            return

        process_files(folder_path, "OFF", num_channels, raw_data, combined_data)

        # Deduplicate raw_data and combined_data
        raw_data = list({tuple(row) for row in raw_data})  # Convert to set and back to list to remove duplicates
        combined_data = list({tuple(row) for row in combined_data})  # Same for combined_data

        # Sort combined data by concentration
        raw_data.sort(key=lambda x: x[2])  # Sort by the concentration string
        combined_data.sort(key=lambda x: x[0])  # Sort by the concentration string

        def save_combined_csv(file_name, signal_type):
            try:
                with open(file_name, mode='w', newline='') as file:
                    csv_writer = csv.writer(file)
                    
                    # Dynamically generate headers based on the number of channels
                    headers = ["File Name"] + [f"Channel {channel} Avg" for channel in range(1, num_channels + 1)]
                    csv_writer.writerow(headers)

                    # Group by frequency and sort within each group by concentration
                    grouped_data = defaultdict(list)
                    for row in combined_data:
                        if row[3] == signal_type:
                            parts = row[0].split('-')  # Split concentration-freq-trial
                            frequency = extract_frequency(parts[1]) if len(parts) > 1 else None
                            grouped_data[frequency].append(row)

                    # Initialize written_rows set to track written rows and avoid duplicates
                    written_rows = set()

                    # Sort groups by frequency (None first, then numerically)
                    for frequency in sorted(grouped_data.keys(), key=lambda x: (x is None, x)):
                        # Sort each group by concentration numerically
                        sorted_group = sorted(grouped_data[frequency], key=lambda x: float(re.search(r'\d+\.?\d*', x[0]).group()))
                        for row in sorted_group:
                            row_data = [row[0]]  # Start with the file name
                            for channel in range(1, num_channels + 1):
                                # Find the row for the current channel
                                channel_row = next((r for r in grouped_data[frequency] if r[2] == channel and r[0] == row[0]), None)
                                if channel_row:
                                    row_data.append(channel_row[1])  # Add the average wave height
                                else:
                                    row_data.append("")  # Add empty value if no data for this channel
                            # Write the row only if it hasn't been written before
                            row_tuple = tuple(row_data)
                            if row_tuple not in written_rows:
                                csv_writer.writerow(row_data)
                                written_rows.add(row_tuple)

            except Exception as e:
                print(f"Failed to save {signal_type} data to {file_name}: {e}")

        # Prompt the user to choose the location for the Signal-ON CSV file
        on_file_name = filedialog.asksaveasfilename(
            title="Save Signal-ON Data As",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not on_file_name:
            print("No file name provided for Signal-ON data. Exiting.")
            return

        # Save Signal-ON data to the chosen location
        save_combined_csv(on_file_name, "ON")

        # Prompt the user to choose the location for the Signal-OFF CSV file
        off_file_name = filedialog.asksaveasfilename(
            title="Save Signal-OFF Data As",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not off_file_name:
            print("No file name provided for Signal-OFF data. Exiting.")
            return

        # Save Signal-OFF data to the chosen location
        save_combined_csv(off_file_name, "OFF")

        # Prompt the user to choose the location for the raw data CSV file
        raw_data_file_name = filedialog.asksaveasfilename(
            title="Save Raw Data As",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not raw_data_file_name:
            print("No file name provided for raw data. Exiting.")
            return

        # Save all wave heights to the chosen location
        try:
            with open(raw_data_file_name, mode='w', newline='') as raw_data_file:
                csv_writer = csv.writer(raw_data_file)
                csv_writer.writerow(["File Name", "Channel", "Concentration", "Wave Height (A)"])
                
                # Track written rows to avoid duplicates
                written_rows = set()

                # Write Signal-ON data
                for row in raw_data:
                    if row[4] == "ON":
                        row_tuple = tuple(row[:4])
                        if row_tuple not in written_rows:
                            csv_writer.writerow(row[:4])
                            written_rows.add(row_tuple)
                
                # Add a blank row
                csv_writer.writerow([])
                
                # Write Signal-OFF data
                for row in raw_data:
                    if row[4] == "OFF":
                        row_tuple = tuple(row[:4])
                        if row_tuple not in written_rows:
                            csv_writer.writerow(row[:4])
                            written_rows.add(row_tuple)

        except Exception as e:
            print(f"Failed to save raw data to {raw_data_file_name}: {e}")

    def get_concentration(self, file_name):
        """Extract concentration from file name based on the first position behind the '-'."""
        parts = file_name.split('-')
        if len(parts) > 1:
            return parts[0]  # Return the first part before the first '-'
        return 'PBS'  # Default to 'PBS' if no '-' is found

    def extract_concentration_value(self, df):
        """Extract numeric concentration values for sorting and plotting."""
        df['Concentration_Value'] = df['Concentration'].replace(['PBS', '0uM'], '0')
        df['Concentration_Value'] = df['Concentration'].str.extract(r'(\d+\.?\d*)').astype(float)
        df['Concentration_Value'] = df['Concentration_Value'].fillna(0)
        return df

    def calculate_kdm(self):
        # Prompt user to select the Signal-ON CSV file
        on_file_path = filedialog.askopenfilename(title="Select Signal-ON CSV file", filetypes=[("CSV files", "*.csv")])
        if not on_file_path:
            print("No ON file selected.")
            return

        # Prompt user to select the Signal-OFF CSV file
        off_file_path = filedialog.askopenfilename(title="Select Signal-OFF CSV file", filetypes=[("CSV files", "*.csv")])
        if not off_file_path:
            print("No OFF file selected.")
            return

        # Read the CSV files
        on_data = pd.read_csv(on_file_path)
        off_data = pd.read_csv(off_file_path)

        # Dynamically generate the expected column names based on the number of channels
        num_channels = len([col for col in on_data.columns if col.startswith("Channel") and col.endswith("Avg")])
        if num_channels == 0:
            messagebox.showerror("Error", "No valid channel data found in the ON or OFF files.")
            return

        required_columns = [f"Channel {i} Avg" for i in range(1, num_channels + 1)]

        # Check for missing columns in ON and OFF data
        missing_on_columns = [col for col in required_columns if col not in on_data.columns]
        missing_off_columns = [col for col in required_columns if col not in off_data.columns]

        if missing_on_columns:
            print(f"ON data file is missing required columns: {', '.join(missing_on_columns)}")
            if messagebox.askyesno("Missing Columns", f"ON data file is missing required columns: {', '.join(missing_on_columns)}. Do you want to proceed without these concentrations?"):
                on_data = on_data.dropna(subset=missing_on_columns, how='any', axis=1)
            else:
                return

        if missing_off_columns:
            print(f"OFF data file is missing required columns: {', '.join(missing_off_columns)}")
            if messagebox.askyesno("Missing Columns", f"OFF data file is missing required columns: {', '.join(missing_off_columns)}. Do you want to proceed without these concentrations?"):
                off_data = off_data.dropna(subset=missing_off_columns, how='any', axis=1)
            else:
                return

        # Extract concentrations from file names
        on_data['Concentration'] = on_data['File Name'].apply(self.get_concentration)
        off_data['Concentration'] = off_data['File Name'].apply(self.get_concentration)

        # Extract numeric concentration values for both datasets
        on_data = self.extract_concentration_value(on_data)
        off_data = self.extract_concentration_value(off_data)

        kdm_results = []
        missing_concentrations_info = []

        for channel in range(1, num_channels + 1):
            channel_column = f"Channel {channel} Avg"
            try:
                # Use the smallest concentration if 'PBS' or '0uM' is not found
                i_min_on = on_data.loc[on_data['Concentration'].isin(['PBS', '0uM', '0']), channel_column].values[0]
            except IndexError:
                i_min_on = on_data.loc[on_data['Concentration_Value'].idxmin(), channel_column]

            try:
                i_min_off = off_data.loc[off_data['Concentration'].isin(['PBS', '0uM', '0']), channel_column].values[0]
            except IndexError:
                i_min_off = off_data.loc[off_data['Concentration_Value'].idxmin(), channel_column]

            missing_concentrations = []
            for _, on_row in on_data.iterrows():
                concentration = on_row['Concentration']
                if concentration in ['PBS', '0uM', '0']:
                    continue
                ion = on_row[channel_column]
                ioff_row = off_data[off_data['Concentration'] == concentration]
                if not ioff_row.empty:
                    ioff = ioff_row[channel_column].values[0]
                    kdm = (ion / i_min_on) - (ioff / i_min_off)
                    percent_signal_on_change = ((ion - i_min_on) / i_min_on) * 100
                    percent_signal_off_change = ((ioff - i_min_off) / i_min_off) * 100
                    kdm_results.append({
                        'Concentration': concentration,
                        'Channel': channel,
                        'KDM': kdm,
                        '% Signal Change ON': percent_signal_on_change,
                        '% Signal Change OFF': percent_signal_off_change
                    })
                else:
                    missing_concentrations.append(concentration)

            if missing_concentrations:
                missing_concentrations_info.append(f"Channel {channel}: {', '.join(missing_concentrations)}")

        if missing_concentrations_info:
            messagebox.showinfo("Missing Concentrations", f"Missing concentrations:\n" + "\n".join(missing_concentrations_info))

        # Convert KDM results to a DataFrame
        kdm_df = pd.DataFrame(kdm_results)

        # Prompt the user to choose the location for the output CSV file
        csv_file_name = filedialog.asksaveasfilename(
            title="Save KDM Results As",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not csv_file_name:
            print("No file name provided.")
            return

        # Save results to the CSV file
        if kdm_results:
            try:
                kdm_df.to_csv(csv_file_name, index=False, columns=['Concentration', 'Channel', 'KDM', '% Signal Change ON', '% Signal Change OFF'])
                print(f"KDM results saved to {csv_file_name}.")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save KDM results: {e}")
        else:
            messagebox.showinfo("No Results", "No valid data to save.")

    def calculate_signal_change_from_files(self):
        # Allow the user to select multiple files
        file_paths = filedialog.askopenfilenames(
            title="Select Files for Signal Change Calculation",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if not file_paths:
            print("No files selected.")
            return

        # Prompt the user to choose the location for the output CSV file
        output_file_name = filedialog.asksaveasfilename(
            title="Save Results As",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not output_file_name:
            print("No file name provided.")
            return

        raw_data_file_name = output_file_name.replace(".csv", "_Raw_Data.csv")

        num_channels = simpledialog.askinteger("Input", "Enter the number of channels to extract:")
        if not num_channels:
            print("No number of channels provided.")
            return

        # Prompt user for the number of trials
        self.num_trials = simpledialog.askinteger("Input", "Enter the number of trials per concentration and frequency:")
        if not self.num_trials:
            print("No number of trials provided.")
            return

        # Process the selected files
        wave_height_data = defaultdict(lambda: defaultdict(list))
        raw_data = []
        for file_path in file_paths:
            file_name = os.path.basename(file_path).replace('.txt', '')  # Remove .txt from file name
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                try:
                    start_idx = next(i for i, line in enumerate(lines) if line.startswith('Potential/V'))
                except StopIteration:
                    print(f"File {file_name} has unexpected axis titles.")
                    continue
                data_lines = lines[start_idx + 1:]
                for channel in range(num_channels):
                    potential, diff = [], []
                    column_index = 1 + (3 * channel)
                    for line in data_lines:
                        parts = line.split(',')
                        if len(parts) > column_index:
                            try:
                                potential_value = float(parts[0].strip())
                                diff_value = float(parts[column_index].strip())
                                potential.append(potential_value)
                                diff.append(diff_value)
                            except ValueError:
                                continue
                    if potential and diff:
                        smoothed_diff = self.savgol_smoothing(np.array(diff), window_length=10, polyorder=1)
                        max_wave_height = self.calculate_wave_heights(smoothed_diff, True)[0]
                        if max_wave_height is not None:
                            wave_height_data[f"Channel_{channel + 1}"][file_name].append(max_wave_height)
                            raw_data.append([file_name, channel + 1, max_wave_height])
                        else:
                            print(f"No valid wave heights found in {file_name} (Channel {channel + 1}).")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

        # Save raw data to a CSV file
        if raw_data:
            try:
                with open(raw_data_file_name, mode='w', newline='') as raw_data_file:
                    csv_writer = csv.writer(raw_data_file)
                    csv_writer.writerow(["File Name", "Channel", "Wave Height (A)"])
                    csv_writer.writerows(raw_data)
                print(f"Raw data saved to {raw_data_file_name}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save raw data: {e}")

        # Calculate average wave heights for each file name and channel
        combined_data = []
        for channel in range(1, num_channels + 1):
            channel_key = f"Channel_{channel}"
            for file_name, wave_heights in wave_height_data[channel_key].items():
                if len(wave_heights) == self.num_trials:  # Use user-specified number of trials
                    formatted_file_name = re.sub(r'-(\d+)$', '.avg', file_name)  # Replace -1, -2, -3 with .avg
                    avg_wave_height = np.mean(wave_heights)
                    combined_data.append([formatted_file_name, avg_wave_height, channel])  # Use formatted file name
                else:
                    print(f"Incomplete data for {file_name} in {channel_key}.")

        # Perform calculations similar to calculate_kdm
        results = []
        for channel in range(1, num_channels + 1):
            channel_data = [row for row in combined_data if row[2] == channel]
            if not channel_data:
                continue

            # Use the smallest wave height as the reference
            i_min = min(row[1] for row in channel_data)

            for row in channel_data:
                file_name, avg_wave_height, _ = row
                percent_signal_change = ((avg_wave_height - i_min) / i_min) * 100
                results.append({
                    'File Name': file_name,  # Use formatted file name
                    'Channel': channel,
                    '% Signal Change': percent_signal_change
                })

        # Save results to the CSV file
        if results:
            try:
                with open(output_file_name, mode='w', newline='') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=['File Name', 'Channel', '% Signal Change'])
                    writer.writeheader()
                    writer.writerows(results)
                messagebox.showinfo("Save Successful", f"Results saved to {output_file_name}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save results: {e}")
        else:
            messagebox.showinfo("No Results", "No valid data to save.")

    def monitor_folder(self):
        """Start monitoring a folder for new files with dynamic plots and processing stages."""
        folder_path = filedialog.askdirectory(
            initialdir=".",
            title="Select Folder to Monitor"
        )
        if not folder_path:
            print("No folder selected. Exiting.")
            return

        num_channels = simpledialog.askinteger("Input", "Enter the number of channels to extract:")
        if not num_channels:
            print("Number of channels not provided. Exiting.")
            return
        
        # Create a new GUI for dynamic plots and buttons
        monitor_window = tk.Toplevel(self.root)
        monitor_window.title("Monitoring Folder")

        # Create a frame for buttons
        button_frame = tk.Frame(monitor_window)
        button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Create a frame for plots
        plot_frame = tk.Frame(monitor_window)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create placeholders for plots
        fig, axes = plt.subplots(num_channels, 1, figsize=(8, 6), sharex=True)
        if num_channels == 1:
            axes = [axes]  # Ensure axes is always a list
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Initialize state variables
        kdm_data = {f"Channel_{i+1}": [] for i in range(num_channels)}  # Store KDM vs time data

        def start_monitoring(stage_name):
            """Start monitoring for files relevant to the specified stage."""

            # Ensure the correct stage is being started
            if stage_name == "Baseline" and self.stage != 0:
                print("Baseline stage already completed. Skipping.")
                return
            elif stage_name == "Upper/Lower Limit" and self.stage != 1:
                print("Upper/Lower Limit stage not ready. Complete Baseline first.")
                return
            elif stage_name == "Monitor" and self.stage != 2:
                print("Monitor stage not ready. Complete Upper/Lower Limit first.")
                return

            # Prompt user for the number of trials for this stage
            num_trials = simpledialog.askinteger("Input", f"Enter the number of trials for {stage_name}:")
            if not num_trials:
                print(f"No number of trials provided for {stage_name}. Exiting.")
                return

            # If starting the Monitor Stage, calculate calibration
            if stage_name == "Monitor":
                def calculate_kdm_for_stage(stage_data, stage_name):
                    """Calculate KDM for a specific stage and populate kdm_results."""
                    for channel, signal_types in stage_data.items():
                        channel_number = channel.split("_")[1]  # Extract the channel number
                        if stage_name == "Baseline":
                            i_min_on = self.get_baseline_i_min(channel_number, "ON")
                            i_min_off = self.get_baseline_i_min(channel_number, "OFF")
                        else:
                            i_min_on = self.get_latest_baseline_i_min(channel_number, "ON")
                            i_min_off = self.get_latest_baseline_i_min(channel_number, "OFF")
                        if i_min_on is None or i_min_off is None:
                            print(f"Missing i_min values for {stage_name} (Channel {channel_number}).")
                            continue

                        # Group data by time and calculate averages
                        grouped_data = defaultdict(lambda: {"ON": [], "OFF": []})
                        for file_name, wave_height in signal_types["ON"]:
                            if stage_name == "Lower Limit" and "-LL" in file_name:
                                time = self.extract_time(file_name)
                                grouped_data[time]["ON"].append(wave_height)
                            elif stage_name == "Upper Limit" and "-UL" in file_name:
                                time = self.extract_time(file_name)
                                grouped_data[time]["ON"].append(wave_height)
                        for file_name, wave_height in signal_types["OFF"]:
                            if stage_name == "Lower Limit" and "-LL" in file_name:
                                time = self.extract_time(file_name)
                                grouped_data[time]["OFF"].append(wave_height)
                            elif stage_name == "Upper Limit" and "-UL" in file_name:
                                time = self.extract_time(file_name)
                                grouped_data[time]["OFF"].append(wave_height)

                        # Calculate KDM for each time
                        for time, values in grouped_data.items():
                            if values["ON"] and values["OFF"]:
                                i_on = np.mean(values["ON"])  # Use average wave height for ON
                                i_off = np.mean(values["OFF"])  # Use average wave height for OFF
                                kdm = (i_on / i_min_on) - (i_off / i_min_off)
                                self.kdm_results[stage_name].append((time, channel_number, kdm, None))  # Concentration is set to None for now

                # Populate KDM results for Baseline, Lower Limit, and Upper Limit stages
                calculate_kdm_for_stage(self.baseline_data, "Baseline")
                calculate_kdm_for_stage(self.upplow_data, "Lower Limit")
                calculate_kdm_for_stage(self.upplow_data, "Upper Limit")

                # Prompt user for upper and lower concentration limits
                try:
                    self.upper_limit = float(simpledialog.askstring("Input", "Enter the upper concentration limit (uM):"))
                    self.lower_limit = float(simpledialog.askstring("Input", "Enter the lower concentration limit (uM):"))
                except ValueError:
                    print("Invalid concentration input. Skipping calibration.")
                    return

                # Use stored upper and lower limits
                y2 = self.upper_limit
                y1 = self.lower_limit

                # Ensure the limits are available
                if y2 is None or y1 is None:
                    print("Upper and lower limits are not set. Cannot save data.")
                    return
                # Calculate calibration for each channel
                channel_calibration = {}
                for channel in range(1, num_channels + 1):
                    try:
                        # Extract KDM values for Lower and Upper Limits
                        lower_limit_kdms = [kdm for _, ch, kdm, _ in self.kdm_results["Lower Limit"] if ch == str(channel)]
                        upper_limit_kdms = [kdm for _, ch, kdm, _ in self.kdm_results["Upper Limit"] if ch == str(channel)]

                        # Check if data is available for both limits
                        if not lower_limit_kdms or not upper_limit_kdms:
                            print(f"Missing data for Channel {channel} in Lower or Upper Limit. Skipping calibration.")
                            continue

                        # Calculate x1 and x2 (mean KDM values for Lower and Upper Limits)
                        x1 = np.mean(lower_limit_kdms)
                        x2 = np.mean(upper_limit_kdms)

                        # Ensure x1 and x2 are not equal to avoid division by zero
                        if x2 == x1:
                            print(f"Cannot calculate slope (m) for Channel {channel} because x2 and x1 are equal.")
                            continue

                        # Calculate slope (m) and intercept (c)
                        m = (y2 - y1) / (x2 - x1)
                        c = y1 - m * x1
                        channel_calibration[str(channel)] = (m, c)

                        # Debugging: Print calculated slope and intercept
                        # print(f"Channel {channel}: m = {m}, c = {c}")
                    except KeyError:
                        print(f"Missing data for Channel {channel} in Lower or Upper Limit. Skipping calibration.")
                    except Exception as e:
                        print(f"Error calculating calibration for Channel {channel}: {e}")

                # Store the calibration for use during monitoring
                self.channel_calibration = channel_calibration

            def is_valid_file(file_name):
                """Check if the file belongs to the current stage."""
                if stage_name == "Baseline":
                    return "-BL-" in file_name
                elif stage_name == "Upper/Lower Limit":
                    return "-UL-" in file_name or "-LL-" in file_name
                elif stage_name == "Monitor":
                    return True
                return False

            processed_files = set()
            def process_file(file_path):
                """Process a single file and calculate KDM."""
                file_name = os.path.basename(file_path)
                if not is_valid_file(file_name):
                    return

                # Retry logic with a small delay before each attempt
                for _ in range(3):  # Retry up to 3 times
                    time.sleep(0.5)  # Small delay before attempting to process the file
                    try:
                        with open(file_path, 'r') as file:
                            lines = file.readlines()
                        start_idx = next(i for i, line in enumerate(lines) if line.startswith('Potential/V'))
                        data_lines = lines[start_idx + 1:]
                        # Process each channel in the file
                        for channel in range(1, num_channels + 1):
                            potential, diff = [], []
                            column_index = 1 + (3 * (channel - 1))  # Adjust column index for each channel
                            for line in data_lines:
                                parts = line.split(',')
                                if len(parts) > column_index:
                                    try:
                                        potential_value = float(parts[0].strip())
                                        diff_value = float(parts[column_index].strip())
                                        potential.append(potential_value)
                                        diff.append(diff_value)
                                    except ValueError:
                                        continue
                            if potential and diff:
                                smoothed_diff = self.savgol_smoothing(np.array(diff), window_length=10, polyorder=1)
                                max_wave_height = self.calculate_wave_heights(smoothed_diff, True)[0]
                                if max_wave_height is not None:
                                    file_time = self.extract_time(file_name)
                                    signal_type = "ON" if "ON" in file_name.upper() else "OFF"
                                    if stage_name == "Baseline":
                                        self.baseline_data[f"Channel_{channel}"][signal_type].append((file_name, max_wave_height))
                                        # print(f"Processed file '{file_name}' in Baseline Stage for channel {channel}.")
                                        # print(f"Max wave height for {file_name} (Channel {channel}): {max_wave_height}")
                                    elif stage_name == "Upper/Lower Limit":
                                        self.upplow_data[f"Channel_{channel}"][signal_type].append((file_name, max_wave_height))
                                        # print(f"Processed file '{file_name}' in Upper/Lower Stage for channel {channel}.")
                                    elif stage_name == "Monitor":
                                        self.monitor_data[f"Channel_{channel}"][signal_type].append((file_name, max_wave_height))
                                        # print(f"Processed file '{file_name}' in Monitor Stage for channel {channel}.")
                                        if self.stage == 2:  # Monitor Stage
                                            calculate_kdm_and_update_plot(channel, file_time)
                        processed_files.add(file_name)  
                        break  # Exit retry loop if successful
                    except Exception as e:
                        print(f"Error processing file '{file_name}': {e}. Retrying...")

            def calculate_kdm_and_update_plot(channel, time):
                """Calculate KDM and update the plot dynamically."""
                channel_key = f"Channel_{channel}"
                signal_on_data = self.monitor_data[channel_key]["ON"]
                signal_off_data = self.monitor_data[channel_key]["OFF"]

                # Check if enough trials are available for the given time
                on_trials = [wave_height for file_name, wave_height in signal_on_data if self.extract_time(file_name) == time]
                off_trials = [wave_height for file_name, wave_height in signal_off_data if self.extract_time(file_name) == time]

                if len(on_trials) == num_trials and len(off_trials) == num_trials:
                    i_on = np.mean(on_trials)
                    i_off = np.mean(off_trials)
                    i_min_on = self.get_latest_baseline_i_min(channel, "ON")
                    i_min_off = self.get_latest_baseline_i_min(channel, "OFF")

                    if i_min_on is not None and i_min_off is not None:
                        kdm = (i_on / i_min_on) - (i_off / i_min_off)
                        m, c = self.channel_calibration.get(str(channel), (None, None))
                        if m is not None and c is not None:
                            concentration = m * kdm + c

                            # Call update_plot to update the graph
                            update_plot(channel, time, concentration)

            def update_plot(channel, time, concentration):
                """Update the plot with new concentration data."""
                ax = axes[channel - 1]
                # Convert time from hhmmss to minutes
                time_in_minutes = int(time[:2]) * 60 + int(time[2:4]) + int(time[4:]) / 60
                kdm_data[f"Channel_{channel}"].append((time_in_minutes, concentration))

                # --- Find all monitoring times for this channel (exclude -BL, -UL, -LL) ---
                monitor_times = [
                    int(self.extract_time(file_name)[:2]) * 60 +
                    int(self.extract_time(file_name)[2:4]) +
                    int(self.extract_time(file_name)[4:]) / 60
                    for file_name, _ in self.monitor_data[f"Channel_{channel}"]["ON"] + self.monitor_data[f"Channel_{channel}"]["OFF"]
                    if all(x not in file_name for x in ["-BL", "-UL", "-LL"])
                ]
                if not monitor_times:
                    return  # Nothing to plot

                min_monitor_time = min(monitor_times)

                # --- Adjust all times so the minimum is 0 ---
                times, concentrations = zip(*[
                    (t - min_monitor_time, c)
                    for t, c in kdm_data[f"Channel_{channel}"]
                    if t in monitor_times  # Only plot monitoring points
                ])

                # Clear the axis and plot the data
                ax.clear()
                ax.scatter(times, concentrations, label=f"Channel {channel}", color='blue', marker='o')

                # Set x-axis ticks to factors of 10 and limit to 7 ticks
                max_time = max(times)
                min_time = min(times)
                tick_interval = max(10, int((max_time - min_time) // 6))  # Ensure tick_interval is an integer
                ticks = list(range(0, int(max_time) + tick_interval, tick_interval))
                ax.set_xticks(ticks)

                # Set axis labels and legend
                ax.set_xlabel("Time (minutes)")
                ax.set_ylabel("Concentration (uM)")
                ax.legend(loc="best")

                # Adjust axis limits and redraw the canvas
                ax.relim()
                ax.autoscale()
                canvas.draw()

            # Start monitoring the folder
            class CustomHandler(FileSystemEventHandler):
                def __init__(self, root, process_file_callback):
                    self.root = root
                    self.process_file_callback = process_file_callback

                def on_created(self, event):
                    if not event.is_directory and event.src_path.endswith('.txt'):
                        # Schedule file processing in the main thread
                        self.root.after(0, lambda: self.process_file_callback(event.src_path))
                
                def on_deleted(self, event):
                    file_name = os.path.basename(event.src_path)
                    if file_name in processed_files:
                        processed_files.remove(file_name)
                        # print(f"Removed '{file_name}' from processed_files due to deletion.")

            # Assign the observer to self.observer
            self.observer = Observer()
            event_handler = CustomHandler(self.root, process_file)
            self.observer.schedule(event_handler, folder_path, recursive=False)
            self.observer.start()
            print(f"Monitoring folder for {stage_name} files: {folder_path}")

        # Add buttons to start monitoring for each stage
        baseline_button = tk.Button(button_frame, text="Start Baseline", command=lambda: start_monitoring("Baseline"))
        baseline_button.pack(pady=5)
        upper_lower_limit_button = tk.Button(button_frame, text="Start Upper/Lower Limit", command=lambda: start_monitoring("Upper/Lower Limit"))
        upper_lower_limit_button.pack(pady=5)
        monitor_button = tk.Button(button_frame, text="Start Monitoring", command=lambda: start_monitoring("Monitor"))
        monitor_button.pack(pady=5)

        # Add a stop button for each stage
        stop_baseline_button = tk.Button(button_frame, text="Stop Baseline", command=lambda: self.stop_monitoring("Baseline"))
        stop_baseline_button.pack(pady=5)

        stop_upper_lower_limit_button = tk.Button(button_frame, text="Stop Upper/Lower Limit", command=lambda: self.stop_monitoring("Upper/Lower Limit"))
        stop_upper_lower_limit_button.pack(pady=5)

        stop_monitor_button = tk.Button(button_frame, text="Stop Monitoring", command=lambda: self.stop_monitoring("Monitor"))
        stop_monitor_button.pack(pady=5)

        def on_close():
            monitor_window.destroy()

        monitor_window.protocol("WM_DELETE_WINDOW", on_close)

    def stop_monitoring(self, stage_name):
        """Stop monitoring for the current stage and update the stage variable."""
        # Map stage names to their corresponding stage numbers
        stage_mapping = {
            "Baseline": 0,
            "Upper/Lower Limit": 1,
            "Monitor": 2
        }

        # Check if the stage being stopped matches the current stage
        if stage_mapping.get(stage_name) != self.stage:
            print(f"Cannot stop {stage_name} stage. Please stop the current stage first.")
            messagebox.showerror("Invalid Action", f"You can only stop the current stage: {list(stage_mapping.keys())[self.stage]}.")
            return

        try:
            if hasattr(self, 'observer') and self.observer is not None and self.observer.is_alive():
                self.observer.stop()
                self.observer.join()
                self.observer = None  # Reset observer to allow re-monitoring
                print(f"Stopped monitoring for {stage_name} files.")
                messagebox.showinfo("Monitoring Stopped", f"{stage_name} monitoring has been stopped.")

                # Update the stage variable
                if stage_name == "Baseline":
                    self.stage = 1  # Move to the Upper/Lower Limit stage
                elif stage_name == "Upper/Lower Limit":
                    self.stage = 2  # Move to the Monitor stage
                elif stage_name == "Monitor":
                    self.stage = 3  # Move to the saving stage
            else:
                print(f"No active monitoring to stop for {stage_name}.")
        except Exception as e:
            print(f"Error stopping monitoring for {stage_name}: {e}")

        # Save data to CSV after stopping monitoring
        if self.stage == 3:
            self.calculate_kdm_watchdog()
            self.save_data_to_csv()

    def save_data_to_csv(self):
        """Save all stages' data into consolidated CSV files."""
        # Deduplicate raw data for each stage
        for stage_data in [self.baseline_data, self.upplow_data, self.monitor_data]:
            for channel, signal_types in stage_data.items():
                for signal_type, data in signal_types.items():
                    stage_data[channel][signal_type] = list(set(data))

        # Prompt user for wave height data file name
        wave_height_file = filedialog.asksaveasfilename(
            title="Save Wave Height Data As",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if wave_height_file:
            try:
                with open(wave_height_file, mode='w', newline='') as file:
                    writer = csv.writer(file)

                    # Write Baseline data
                    writer.writerow(["Baseline"])
                    writer.writerow(["File Name", "Channel", "Signal Type", "Wave Height"])
                    for channel, signal_types in self.baseline_data.items():
                        channel_number = channel.split("_")[1]  # Extract the channel number
                        for signal_type, data in signal_types.items():
                            # Sort data by signal type (ON first, then OFF) and time
                            sorted_data = sorted(
                                data,
                                key=lambda x: (signal_type != "ON", self.extract_time(x[0]))
                            )
                            for file_name, wave_height in sorted_data:
                                writer.writerow([file_name, channel_number, signal_type, wave_height])

                    # Write Lower Limit data
                    writer.writerow([])
                    writer.writerow(["Lower Limit"])
                    writer.writerow(["File Name", "Channel", "Signal Type", "Wave Height"])
                    for channel, signal_types in self.upplow_data.items():
                        channel_number = channel.split("_")[1]  # Extract the channel number
                        for signal_type, data in signal_types.items():
                            # Filter for Lower Limit files
                            lower_limit_data = [d for d in data if "-LL" in d[0]]
                            # Sort data by signal type (ON first, then OFF) and time
                            sorted_data = sorted(
                                lower_limit_data,
                                key=lambda x: (signal_type != "ON", self.extract_time(x[0]))
                            )
                            for file_name, wave_height in sorted_data:
                                writer.writerow([file_name, channel_number, signal_type, wave_height])

                    # Write Upper Limit data
                    writer.writerow([])
                    writer.writerow(["Upper Limit"])
                    writer.writerow(["File Name", "Channel", "Signal Type", "Wave Height"])
                    for channel, signal_types in self.upplow_data.items():
                        channel_number = channel.split("_")[1]  # Extract the channel number
                        for signal_type, data in signal_types.items():
                            # Filter for Upper Limit files
                            upper_limit_data = [d for d in data if "-UL" in d[0]]
                            # Sort data by signal type (ON first, then OFF) and time
                            sorted_data = sorted(
                                upper_limit_data,
                                key=lambda x: (signal_type != "ON", self.extract_time(x[0]))
                            )
                            for file_name, wave_height in sorted_data:
                                writer.writerow([file_name, channel_number, signal_type, wave_height])

                    # Write Monitor data
                    writer.writerow([])
                    writer.writerow(["Monitor"])
                    writer.writerow(["File Name", "Channel", "Signal Type", "Wave Height"])
                    for channel, signal_types in self.monitor_data.items():
                        channel_number = channel.split("_")[1]  # Extract the channel number
                        for signal_type, data in signal_types.items():
                            # Sort data by signal type (ON first, then OFF) and time
                            sorted_data = sorted(
                                data,
                                key=lambda x: (signal_type != "ON", self.extract_time(x[0]))
                            )
                            for file_name, wave_height in sorted_data:
                                writer.writerow([file_name, channel_number, signal_type, wave_height])

                    print(f"Wave height data saved to {wave_height_file}")
            except Exception as e:
                print(f"Failed to save wave height data: {e}")
        
        # Prompt user for average wave height data file name
        avg_wave_height_file = filedialog.asksaveasfilename(
            title="Save Average Wave Height Data As",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if avg_wave_height_file:
            try:
                with open(avg_wave_height_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    
                    # Process Baseline data
                    writer.writerow(["Baseline"])
                    writer.writerow(["Time", "Channel", "Signal Type", "Avg. Height"])
                    for channel, signal_types in self.baseline_data.items():
                        channel_number = channel.split("_")[1]  # Extract the channel number
                        for signal_type, data in signal_types.items():
                            # Group data by time and calculate averages
                            grouped_data = defaultdict(list)
                            for file_name, wave_height in data:
                                time = self.extract_time(file_name)
                                grouped_data[time].append(wave_height)

                            # Sort and calculate the average for each time
                            for time, heights in sorted(grouped_data.items(), key=lambda x: (signal_type != "ON", x[0])):
                                avg_height = np.mean(heights)
                                writer.writerow([time, channel_number, signal_type, avg_height])

                    # Process Lower Limit data
                    writer.writerow([])
                    writer.writerow(["Lower Limit"])
                    writer.writerow(["Time", "Channel", "Signal Type", "Avg. Height"])
                    for channel, signal_types in self.upplow_data.items():
                        channel_number = channel.split("_")[1]  # Extract the channel number
                        for signal_type, data in signal_types.items():
                            # Filter for Lower Limit files
                            lower_limit_data = [d for d in data if "-LL" in d[0]]
                            grouped_data = defaultdict(list)
                            for file_name, wave_height in lower_limit_data:
                                time = self.extract_time(file_name)
                                grouped_data[time].append(wave_height)

                            # Sort and calculate the average for each time
                            for time, heights in sorted(grouped_data.items(), key=lambda x: (signal_type != "ON", x[0])):
                                avg_height = np.mean(heights)
                                writer.writerow([time, channel_number, signal_type, avg_height])

                    # Process Upper Limit data
                    writer.writerow([])
                    writer.writerow(["Upper Limit"])
                    writer.writerow(["Time", "Channel", "Signal Type", "Avg. Height"])
                    for channel, signal_types in self.upplow_data.items():
                        channel_number = channel.split("_")[1]  # Extract the channel number
                        for signal_type, data in signal_types.items():
                            # Filter for Upper Limit files
                            upper_limit_data = [d for d in data if "-UL" in d[0]]
                            grouped_data = defaultdict(list)
                            for file_name, wave_height in upper_limit_data:
                                time = self.extract_time(file_name)
                                grouped_data[time].append(wave_height)

                            # Sort and calculate the average for each time
                            for time, heights in sorted(grouped_data.items(), key=lambda x: (signal_type != "ON", x[0])):
                                avg_height = np.mean(heights)
                                writer.writerow([time, channel_number, signal_type, avg_height])

                    # Process Monitor data
                    writer.writerow([])
                    writer.writerow(["Monitor"])
                    writer.writerow(["Time", "Channel", "Signal Type", "Avg. Height"])
                    for channel, signal_types in self.monitor_data.items():
                        channel_number = channel.split("_")[1]  # Extract the channel number
                        for signal_type, data in signal_types.items():
                            # Group data by time and calculate averages
                            grouped_data = defaultdict(list)
                            for file_name, wave_height in data:
                                time = self.extract_time(file_name)
                                grouped_data[time].append(wave_height)

                            # Sort and calculate the average for each time
                            for time, heights in sorted(grouped_data.items(), key=lambda x: (signal_type != "ON", x[0])):
                                avg_height = np.mean(heights)
                                writer.writerow([time, channel_number, signal_type, avg_height])

                    print(f"Average wave height data saved to {avg_wave_height_file}")
            except Exception as e:
                print(f"Failed to save average wave height data: {e}")
                
        # Use stored upper and lower limits
        y2 = self.upper_limit
        y1 = self.lower_limit

        # Ensure the limits are available
        if y2 is None or y1 is None:
            print("Upper and lower limits are not set. Cannot save data.")
            return
            
        # Calculate m and c for each channel
        channel_calibration = {}
        for channel in self.monitor_data.keys():
            channel_number = channel.split("_")[1]  # Extract the channel number
            # Calculate x1 and x2
            x1 = np.mean([kdm for _, ch, kdm, _ in self.kdm_results["Lower Limit"] if ch == channel_number])
            x2 = np.mean([kdm for _, ch, kdm, _ in self.kdm_results["Upper Limit"] if ch == channel_number])

            if x2 == x1:
                print(f"Cannot calculate slope (m) for Channel {channel_number} because x2 and x1 are equal.")
                continue

            # Calculate m and c
            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1
            channel_calibration[channel_number] = (m, c)
            print(f"Channel {channel_number}: m = {m}, c = {c}")

        # Prompt user for KDM results file name
        kdm_file = filedialog.asksaveasfilename(
            title="Save KDM Results As",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if kdm_file:
            # Debugging: Check the structure of self.kdm_results
            if not isinstance(self.kdm_results, dict):
                print(f"Error: self.kdm_results is not a dictionary. Found type: {type(self.kdm_results)}")
                messagebox.showerror("Error", "KDM results are not properly initialized. Please calculate KDM first.")
                return

            try:
                with open(kdm_file, mode='w', newline='') as file:
                    writer = csv.writer(file)

                    # Write Baseline data
                    writer.writerow(["Baseline"])
                    writer.writerow(["Time", "Channel", "KDM", "Concentration (uM)"])
                    for result in self.kdm_results.get("Baseline", []):
                        time, channel, kdm, concentration = result
                        writer.writerow([time, channel, kdm])

                    # Write Lower Limit data
                    writer.writerow([])
                    writer.writerow(["Lower Limit"])
                    writer.writerow(["Time", "Channel", "KDM", "Concentration (uM)"])
                    for result in self.kdm_results.get("Lower Limit", []):
                        time, channel, kdm, _ = result  # Ignore the placeholder concentration
                        if channel in channel_calibration:
                            m, c = channel_calibration[channel]
                            concentration = m * kdm + c  # Calculate concentration
                        else:
                            concentration = "N/A"  # If calibration is missing, set concentration to "N/A"
                        writer.writerow([time, channel, kdm, concentration])

                    # Write Upper Limit data
                    writer.writerow([])
                    writer.writerow(["Upper Limit"])
                    writer.writerow(["Time", "Channel", "KDM", "Concentration (uM)"])
                    for result in self.kdm_results.get("Upper Limit", []):
                        time, channel, kdm, _ = result  # Ignore the placeholder concentration
                        if channel in channel_calibration:
                            m, c = channel_calibration[channel]
                            concentration = m * kdm + c  # Calculate concentration
                        else:
                            concentration = "N/A"  # If calibration is missing, set concentration to "N/A"
                        writer.writerow([time, channel, kdm, concentration])

                    # Write Monitor data
                    writer.writerow([])
                    writer.writerow(["Monitor"])
                    writer.writerow(["Time", "Channel", "KDM", "Concentration (uM)"])
                    for result in self.kdm_results.get("Monitor", []):
                        time, channel, kdm, _ = result  # Ignore the placeholder concentration
                        if channel in channel_calibration:
                            m, c = channel_calibration[channel]
                            concentration = m * kdm + c  # Calculate concentration
                        else:
                            concentration = "N/A"  # If calibration is missing, set concentration to "N/A"
                        writer.writerow([time, channel, kdm, concentration])

                    print(f"KDM results saved to {kdm_file}")
            except Exception as e:
                print(f"Failed to save KDM results: {e}")
                messagebox.showerror("Error", f"Failed to save KDM results: {e}")

    def calculate_kdm_watchdog(self):
        """Calculate KDM for all sections and save the results."""
        kdm_results = {
            "Baseline": [],
            "Lower Limit": [],
            "Upper Limit": [],
            "Monitor": []
        }

        # Process Baseline section
        for channel, signal_types in self.baseline_data.items():
            channel_number = channel.split("_")[1]  # Extract the channel number
            i_min_on = self.get_baseline_i_min(channel_number, "ON")
            i_min_off = self.get_baseline_i_min(channel_number, "OFF")
            if i_min_on is None or i_min_off is None:
                print(f"Missing i_min values for Baseline (Channel {channel_number}).")
                continue

            # Group data by time and calculate averages
            grouped_data = defaultdict(lambda: {"ON": [], "OFF": []})
            for file_name, wave_height in signal_types["ON"]:
                time = self.extract_time(file_name)
                grouped_data[time]["ON"].append(wave_height)
            for file_name, wave_height in signal_types["OFF"]:
                time = self.extract_time(file_name)
                grouped_data[time]["OFF"].append(wave_height)

            # Calculate KDM for each time
            for time, values in grouped_data.items():
                if values["ON"] and values["OFF"]:
                    i_on = np.mean(values["ON"])  # Use average wave height for ON
                    i_off = np.mean(values["OFF"])  # Use average wave height for OFF
                    kdm = (i_on / i_min_on) - (i_off / i_min_off)
                    kdm_results["Baseline"].append((time, channel_number, kdm, 10))  # Concentration is set to 10

        # Process Lower Limit section
        for channel, signal_types in self.upplow_data.items():
            channel_number = channel.split("_")[1]  # Extract the channel number
            i_min_on = self.get_latest_baseline_i_min(channel_number, "ON")
            i_min_off = self.get_latest_baseline_i_min(channel_number, "OFF")
            if i_min_on is None or i_min_off is None:
                print(f"Missing i_min values for Lower Limit (Channel {channel_number}).")
                continue

            # Group data by time and calculate averages
            grouped_data = defaultdict(lambda: {"ON": [], "OFF": []})
            for file_name, wave_height in signal_types["ON"]:
                if "-LL" in file_name:
                    time = self.extract_time(file_name)
                    grouped_data[time]["ON"].append(wave_height)
            for file_name, wave_height in signal_types["OFF"]:
                if "-LL" in file_name:
                    time = self.extract_time(file_name)
                    grouped_data[time]["OFF"].append(wave_height)

            # Calculate KDM for each time
            for time, values in grouped_data.items():
                if values["ON"] and values["OFF"]:
                    i_on = np.mean(values["ON"])  # Use average wave height for ON
                    i_off = np.mean(values["OFF"])  # Use average wave height for OFF
                    kdm = (i_on / i_min_on) - (i_off / i_min_off)
                    kdm_results["Lower Limit"].append((time, channel_number, kdm, 10))  # Concentration is set to 10

        # Process Upper Limit section
        for channel, signal_types in self.upplow_data.items():
            channel_number = channel.split("_")[1]  # Extract the channel number
            i_min_on = self.get_latest_baseline_i_min(channel_number, "ON")
            i_min_off = self.get_latest_baseline_i_min(channel_number, "OFF")
            if i_min_on is None or i_min_off is None:
                print(f"Missing i_min values for Upper Limit (Channel {channel_number}).")
                continue

            # Group data by time and calculate averages
            grouped_data = defaultdict(lambda: {"ON": [], "OFF": []})
            for file_name, wave_height in signal_types["ON"]:
                if "-UL" in file_name:
                    time = self.extract_time(file_name)
                    grouped_data[time]["ON"].append(wave_height)
            for file_name, wave_height in signal_types["OFF"]:
                if "-UL" in file_name:
                    time = self.extract_time(file_name)
                    grouped_data[time]["OFF"].append(wave_height)

            # Calculate KDM for each time
            for time, values in grouped_data.items():
                if values["ON"] and values["OFF"]:
                    i_on = np.mean(values["ON"])  # Use average wave height for ON
                    i_off = np.mean(values["OFF"])  # Use average wave height for OFF
                    kdm = (i_on / i_min_on) - (i_off / i_min_off)
                    kdm_results["Upper Limit"].append((time, channel_number, kdm, 10))  # Concentration is set to 10

        # Process Monitor section
        for channel, signal_types in self.monitor_data.items():
            channel_number = channel.split("_")[1]  # Extract the channel number
            i_min_on = self.get_latest_baseline_i_min(channel_number, "ON")
            i_min_off = self.get_latest_baseline_i_min(channel_number, "OFF")
            if i_min_on is None or i_min_off is None:
                print(f"Missing i_min values for Monitor (Channel {channel_number}).")
                continue

            # Group data by time and calculate averages
            grouped_data = defaultdict(lambda: {"ON": [], "OFF": []})
            for file_name, wave_height in signal_types["ON"]:
                time = self.extract_time(file_name)
                grouped_data[time]["ON"].append(wave_height)
            for file_name, wave_height in signal_types["OFF"]:
                time = self.extract_time(file_name)
                grouped_data[time]["OFF"].append(wave_height)

            # Calculate KDM for each time
            for time, values in grouped_data.items():
                if values["ON"] and values["OFF"]:
                    i_on = np.mean(values["ON"])  # Use average wave height for ON
                    i_off = np.mean(values["OFF"])  # Use average wave height for OFF
                    kdm = (i_on / i_min_on) - (i_off / i_min_off)
                    kdm_results["Monitor"].append((time, channel_number, kdm, 10))  # Concentration is set to 10

        # Save KDM results
        self.kdm_results = kdm_results

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Wave Height Extraction Tool")
    app = WaveHeightExtractionTool(root)
    root.mainloop()