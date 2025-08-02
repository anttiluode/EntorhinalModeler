#!/usr/bin/env python3
"""
The Enhanced Entorhinal Modeler v4 (Brain-Optimized)
===================================================

Specifically tuned to capture subtle brainwave patterns and amplify them
into visible moiré interference patterns. Enhanced sensitivity for EEG signals.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import butter, filtfilt, hilbert, welch
from scipy.ndimage import gaussian_filter
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

class EnhancedEntorhinalModeler:
    """Brain-optimized visualization of Moiré interference from EEG theta"""
    
    def __init__(self):
        self.eeg_raw = None
        self.sample_rate = 250
        self.current_time = 0
        self.window_size = 4.0  # Longer window for better frequency resolution
        self.is_playing = False
        
        # Enhanced frequency bands for better brain signal capture
        self.freq_bands = { 
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        # Brain signal enhancement parameters
        self.brain_amplification = 1000.0  # Strong amplification for weak EEG
        self.noise_threshold = 1e-8  # Very sensitive threshold
        self.adaptive_scaling = True
        
        self.setup_gui()
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Enhanced Entorhinal Modeler - Brain Signal Optimized")
        self.root.geometry("1600x1000")
        
        left_panel = ttk.Frame(self.root, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # EEG Controls with enhanced sensitivity options
        eeg_frame = ttk.LabelFrame(left_panel, text="Brain Signal Source")
        eeg_frame.pack(fill=tk.X, pady=5)
        ttk.Button(eeg_frame, text="Load EEG File", command=self.load_eeg_file).pack(fill=tk.X, pady=5)
        
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(eeg_frame, textvariable=self.channel_var, state="readonly", width=18)
        self.channel_combo.pack(fill=tk.X, pady=5)
        self.channel_combo.bind('<<ComboboxSelected>>', self.on_channel_change)
        
        # Brain Signal Enhancement Controls
        enhance_frame = ttk.LabelFrame(left_panel, text="Brain Signal Enhancement")
        enhance_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(enhance_frame, text="Signal Amplification:").pack(anchor=tk.W)
        self.amplification_var = tk.DoubleVar(value=1000.0)
        self.amp_scale = ttk.Scale(enhance_frame, from_=100.0, to=10000.0, variable=self.amplification_var, 
                                  orient=tk.HORIZONTAL, command=self.update_visualization)
        self.amp_scale.pack(fill=tk.X, pady=2)
        self.amp_label = ttk.Label(enhance_frame, text="1000x")
        self.amp_label.pack(anchor=tk.E)
        
        ttk.Label(enhance_frame, text="Frequency Focus:").pack(anchor=tk.W)
        self.freq_focus_var = tk.StringVar(value="theta")
        freq_combo = ttk.Combobox(enhance_frame, textvariable=self.freq_focus_var, 
                                 values=["theta", "alpha", "beta", "gamma", "all"], state="readonly")
        freq_combo.pack(fill=tk.X, pady=2)
        freq_combo.bind('<<ComboboxSelected>>', self.update_visualization)
        
        # Adaptive processing checkbox
        self.adaptive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(enhance_frame, text="Adaptive Brain Signal Processing", 
                       variable=self.adaptive_var, command=self.update_visualization).pack(anchor=tk.W)
        
        # Playback Controls
        playback_frame = ttk.Frame(eeg_frame)
        playback_frame.pack(fill=tk.X, pady=5)
        self.play_button = ttk.Button(playback_frame, text="▶ Play", command=self.toggle_playback)
        self.play_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(playback_frame, text="⏮ Reset", command=self.reset_time).pack(side=tk.LEFT, padx=2)

        self.time_var = tk.DoubleVar()
        self.time_slider = ttk.Scale(eeg_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                   variable=self.time_var, command=self.update_visualization)
        self.time_slider.pack(fill=tk.X, pady=5, padx=5)
        self.time_label = ttk.Label(eeg_frame, text="0.0s / 0.0s")
        self.time_label.pack(anchor=tk.E, padx=5)

        # Enhanced Model Parameters
        model_frame = ttk.LabelFrame(left_panel, text="Brain-Tuned Moiré Parameters")
        model_frame.pack(fill=tk.X, pady=10)
        
        self.model_type_var = tk.StringVar(value="Brain Theta Modulation")
        ttk.Radiobutton(model_frame, text="Brain Theta Modulation", variable=self.model_type_var, 
                       value="Brain Theta Modulation", command=self.update_visualization).pack(anchor=tk.W)
        ttk.Radiobutton(model_frame, text="Multi-Band Interference", variable=self.model_type_var, 
                       value="Multi-Band Interference", command=self.update_visualization).pack(anchor=tk.W)

        ttk.Label(model_frame, text="\nTheta Grid Sensitivity:").pack(anchor=tk.W)
        self.sensitivity_var = tk.DoubleVar(value=10.0)
        ttk.Scale(model_frame, from_=1.0, to=50.0, variable=self.sensitivity_var, 
                 orient=tk.HORIZONTAL, command=self.update_visualization).pack(fill=tk.X)
        
        ttk.Label(model_frame, text="Interference Strength:").pack(anchor=tk.W)
        self.interference_var = tk.DoubleVar(value=5.0)
        ttk.Scale(model_frame, from_=0.1, to=20.0, variable=self.interference_var, 
                 orient=tk.HORIZONTAL, command=self.update_visualization).pack(fill=tk.X)
        
        ttk.Label(model_frame, text="Spatial Resolution:").pack(anchor=tk.W)
        self.resolution_var = tk.DoubleVar(value=0.5)
        ttk.Scale(model_frame, from_=0.1, to=2.0, variable=self.resolution_var, 
                 orient=tk.HORIZONTAL, command=self.update_visualization).pack(fill=tk.X)
        
        # Real-time brain signal analysis
        analysis_frame = ttk.LabelFrame(left_panel, text="Brain Signal Analysis")
        analysis_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.analysis_label = ttk.Label(analysis_frame, text="Load brain data...", 
                                       font=("Consolas", 9), wraplength=350)
        self.analysis_label.pack(pady=10, padx=5)

        # Visualization Panel
        right_panel = ttk.Frame(self.root)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.setup_plots()

    def setup_plots(self):
        self.ax1.set_title("Enhanced Brain Signals", fontweight='bold')
        self.ax2.set_title("Brain-Driven Moiré Pattern", fontweight='bold')
        self.ax3.set_title("Frequency Analysis", fontweight='bold')
        self.ax4.set_title("3D Brain Field", fontweight='bold')
        for ax in [self.ax1, self.ax2]:
            ax.set_xticks([]); ax.set_yticks([])
        self.fig.tight_layout(pad=3.0)

    def enhance_brain_signal(self, eeg_data):
        """Advanced brain signal enhancement specifically for weak EEG"""
        if len(eeg_data) < 100:
            return eeg_data
            
        # Remove DC offset and detrend
        eeg_data = eeg_data - np.mean(eeg_data)
        eeg_data = eeg_data - np.linspace(eeg_data[0], eeg_data[-1], len(eeg_data))
        
        # Adaptive noise reduction
        if self.adaptive_var.get():
            # Estimate noise floor
            noise_est = np.std(eeg_data[:int(len(eeg_data)*0.1)])
            if noise_est > 0:
                # Soft thresholding to reduce noise while preserving brain signals
                threshold = noise_est * 2
                eeg_data = np.sign(eeg_data) * np.maximum(np.abs(eeg_data) - threshold, 0)
        
        # Apply strong amplification for weak brain signals
        eeg_data *= self.amplification_var.get()
        
        # Prevent saturation
        max_val = np.max(np.abs(eeg_data))
        if max_val > 1000:
            eeg_data = eeg_data / max_val * 1000
            
        return eeg_data

    def extract_brain_frequencies(self, eeg_data):
        """Extract and enhance specific brain frequency bands"""
        enhanced_data = self.enhance_brain_signal(eeg_data)
        nyquist = self.sample_rate / 2
        
        band_signals = {}
        band_powers = {}
        
        focus = self.freq_focus_var.get()
        bands_to_process = [focus] if focus != "all" else list(self.freq_bands.keys())
        
        for band_name in bands_to_process:
            if band_name not in self.freq_bands:
                continue
                
            low, high = self.freq_bands[band_name]
            low_norm = max(low / nyquist, 0.01)
            high_norm = min(high / nyquist, 0.99)
            
            try:
                # High-order filter for better frequency isolation
                b, a = butter(6, [low_norm, high_norm], btype='band')
                filtered = filtfilt(b, a, enhanced_data)
                
                # Extract envelope for amplitude modulation
                analytic_signal = hilbert(filtered)
                envelope = np.abs(analytic_signal)
                
                # Calculate instantaneous power
                power = np.mean(envelope**2)
                
                band_signals[band_name] = filtered
                band_powers[band_name] = power
                
            except Exception as e:
                print(f"Error processing {band_name}: {e}")
                band_signals[band_name] = np.zeros_like(enhanced_data)
                band_powers[band_name] = 0
        
        return band_signals, band_powers

    def create_brain_hex_grid(self, X, Y, brain_signal, sensitivity):
        """Create hexagonal grid modulated by brain activity"""
        if len(brain_signal) == 0:
            return np.zeros_like(X)
            
        # Use brain signal statistics to modulate grid
        signal_power = np.var(brain_signal)
        signal_mean = np.mean(np.abs(brain_signal))
        
        # Dynamic spacing based on brain activity
        base_spacing = 5.0 + signal_power * sensitivity * 0.1
        
        # Phase modulation from brain signal
        phase_mod = signal_mean * sensitivity * 100
        
        # Create hexagonal lattice
        omega = (4 * np.pi) / (base_spacing * np.sqrt(3))
        
        # Three rotated gratings for hexagonal pattern
        rotations = np.deg2rad([0, 60, 120])
        vectors = [np.array([np.cos(r), np.sin(r)]) for r in rotations]
        
        grid = sum(np.cos(omega * (vec[0]*X + vec[1]*Y) + phase_mod) for vec in vectors)
        
        # Apply brain signal envelope modulation
        if len(brain_signal) > 10:
            # Resample brain signal to match grid
            brain_envelope = np.abs(hilbert(brain_signal))
            avg_envelope = np.mean(brain_envelope)
            grid *= (1 + avg_envelope * sensitivity * 0.01)
        
        return np.exp(0.3 * (grid + 1.5))

    def generate_brain_moire_pattern(self, band_signals, band_powers):
        """Generate moiré pattern driven by real brain activity"""
        grid_size = 300
        x, y = np.linspace(-50, 50, grid_size), np.linspace(-50, 50, grid_size)
        X, Y = np.meshgrid(x, y)
        
        model_type = self.model_type_var.get()
        sensitivity = self.sensitivity_var.get()
        interference = self.interference_var.get()
        resolution = self.resolution_var.get()
        
        if model_type == "Brain Theta Modulation":
            # Focus on theta band for primary modulation
            theta_signal = band_signals.get('theta', np.array([]))
            
            if len(theta_signal) > 0:
                # Create two grids with slightly different theta modulations
                grid1 = self.create_brain_hex_grid(X, Y, theta_signal, sensitivity)
                
                # Second grid with phase shift based on theta power
                theta_power = band_powers.get('theta', 0)
                phase_shift = theta_power * interference * 0.1
                
                X_shifted = X * np.cos(phase_shift) - Y * np.sin(phase_shift)
                Y_shifted = X * np.sin(phase_shift) + Y * np.cos(phase_shift)
                grid2 = self.create_brain_hex_grid(X_shifted, Y_shifted, theta_signal, sensitivity * 1.1)
                
                moire = grid1 + grid2
                analysis_text = f"Brain Theta Modulation Active\nTheta Power: {theta_power:.2e}\nSensitivity: {sensitivity:.1f}\nPhase Shift: {phase_shift:.3f} rad"
            else:
                moire = np.zeros_like(X)
                analysis_text = "No theta signal detected\nIncrease amplification or check channel"
                
        else:  # Multi-Band Interference
            moire = np.zeros_like(X)
            total_power = 0
            
            for band_name, signal in band_signals.items():
                if len(signal) > 0:
                    power = band_powers[band_name]
                    grid = self.create_brain_hex_grid(X, Y, signal, sensitivity * power * 1000)
                    moire += grid * power
                    total_power += power
            
            if total_power > 0:
                moire /= total_power
                analysis_text = f"Multi-Band Interference\nTotal Power: {total_power:.2e}\nActive Bands: {len(band_signals)}\nResolution: {resolution:.2f}"
            else:
                analysis_text = "No brain signals detected\nCheck amplification and channel"
        
        # Apply spatial smoothing based on resolution
        if resolution != 1.0:
            sigma = (2.0 - resolution) * 2.0
            moire = gaussian_filter(moire, sigma=sigma)
        
        self.analysis_label.config(text=analysis_text)
        return moire

    def update_visualization(self, value=None):
        if not self.eeg_raw or not self.channel_var.get(): 
            return
        
        self.current_time = self.time_var.get()
        duration = self.eeg_raw.n_times / self.sample_rate
        self.time_label.config(text=f"{self.current_time:.1f}s / {duration:.1f}s")
        self.amp_label.config(text=f"{self.amplification_var.get():.0f}x")

        # Extract brain data with enhanced window
        start_sample = int(max(0, self.current_time - self.window_size/2) * self.sample_rate)
        end_sample = int(min(duration, self.current_time + self.window_size/2) * self.sample_rate)
        ch_idx = self.eeg_raw.ch_names.index(self.channel_var.get())
        eeg_data, _ = self.eeg_raw[ch_idx, start_sample:end_sample]
        eeg_data = eeg_data.flatten()
        
        if len(eeg_data) < 100: 
            return

        # Extract brain frequency bands
        band_signals, band_powers = self.extract_brain_frequencies(eeg_data)
        
        # Generate brain-driven moiré pattern
        moire = self.generate_brain_moire_pattern(band_signals, band_powers)
        
        # Clear and update plots
        [ax.clear() for ax in self.fig.axes]
        self.setup_plots()

        # Plot 1: Enhanced brain signals
        time_axis = np.linspace(0, len(eeg_data)/self.sample_rate, len(eeg_data))
        colors = {'theta': 'orange', 'alpha': 'green', 'beta': 'blue', 'gamma': 'red'}
        
        offset = 0
        for band_name, signal in band_signals.items():
            if len(signal) > 0:
                power = band_powers[band_name]
                self.ax1.plot(time_axis, signal + offset, color=colors.get(band_name, 'gray'), 
                            label=f'{band_name}: {power:.2e}', alpha=0.8)
                offset += np.max(np.abs(signal)) * 2.5
        
        self.ax1.set_title("Enhanced Brain Signals", fontweight='bold')
        self.ax1.legend(loc='upper right', fontsize=8)
        self.ax1.grid(True, alpha=0.3)

        # Plot 2: Brain-driven moiré pattern
        self.ax2.imshow(moire, cmap='plasma', origin='lower', aspect='auto')
        self.ax2.set_title("Brain-Driven Moiré Pattern", fontweight='bold')

        # Plot 3: Frequency analysis
        if band_powers:
            bands = list(band_powers.keys())
            powers = list(band_powers.values())
            bars = self.ax3.bar(bands, powers, color=[colors.get(b, 'gray') for b in bands])
            self.ax3.set_title("Brain Frequency Powers", fontweight='bold')
            self.ax3.set_ylabel("Power")
            self.ax3.tick_params(axis='x', rotation=45)
            
            # Add power values on bars
            for bar, power in zip(bars, powers):
                height = bar.get_height()
                self.ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{power:.1e}', ha='center', va='bottom', fontsize=8)

        # Plot 4: 3D representation
        self.ax4.remove()
        self.ax4 = self.fig.add_subplot(2, 2, 4, projection='3d')
        
        # Downsample for 3D visualization
        moire_3d = moire[::10, ::10]
        x_3d = np.arange(moire_3d.shape[1])
        y_3d = np.arange(moire_3d.shape[0])
        X_3d, Y_3d = np.meshgrid(x_3d, y_3d)
        
        self.ax4.plot_surface(X_3d, Y_3d, moire_3d, cmap='plasma', alpha=0.8)
        self.ax4.set_title("3D Brain Field", fontweight='bold')

        self.fig.tight_layout()
        self.canvas.draw()

    def load_eeg_file(self):
        filepath = filedialog.askopenfilename(title="Select EEG File", 
                                            filetypes=[("EDF files", "*.edf"), ("All files", "*.*")])
        if not filepath: return
        try:
            self.eeg_raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            self.sample_rate = self.eeg_raw.info['sfreq']
            duration = self.eeg_raw.n_times / self.sample_rate
            all_channels = self.eeg_raw.ch_names
            
            self.channel_combo['values'] = all_channels
            if all_channels: self.channel_combo.set(all_channels[0])
            self.time_slider.config(to=duration)
            messagebox.showinfo("Success", f"Brain data loaded!\nDuration: {duration:.1f}s\nSample rate: {self.sample_rate} Hz\nChannels: {len(all_channels)}")
            self.on_channel_change()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load brain data:\n{e}")

    def on_channel_change(self, event=None):
        self.update_visualization()

    def toggle_playback(self):
        self.is_playing = not self.is_playing
        self.play_button.config(text="⏸ Pause" if self.is_playing else "▶ Play")
        if self.is_playing: self.start_playback()

    def start_playback(self):
        if not self.is_playing or not self.eeg_raw: return
        current_time = self.time_var.get()
        max_time = self.eeg_raw.n_times / self.sample_rate
        if current_time >= max_time:
            self.is_playing = False; self.play_button.config(text="▶ Play")
            return
        new_time = min(current_time + 0.2, max_time)
        self.time_var.set(new_time)
        self.update_visualization()
        self.root.after(200, self.start_playback)

    def reset_time(self):
        self.time_var.set(0)
        self.is_playing = False
        self.play_button.config(text="▶ Play")
        self.update_visualization()

    def run(self):
        print("🧠⚡️ Enhanced Entorhinal Modeler - Brain-Optimized ⚡️🧠")
        print("Specifically tuned for weak brainwave signals!")
        print("Features:")
        print("- 1000x+ signal amplification")
        print("- Adaptive noise reduction")
        print("- Brain-specific frequency isolation")
        print("- Real-time moiré generation from neural activity")
        self.root.mainloop()

def main():
    visualizer = EnhancedEntorhinalModeler()
    visualizer.run()

if __name__ == "__main__":
    main()