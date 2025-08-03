#!/usr/bin/env python3
"""
Multi-Channel Neural Moiré Mapper v6 (Fixed)
============================================

Real-time visualization of neural network interference through cross-channel moiré patterns.
Based on Blair et al. 2007 grid cell methodology with enhanced multi-channel support.
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
import itertools

class MultiChannelNeuralMapper:
    """Multi-channel neural network mapping through moiré interference"""
    
    def __init__(self):
        self.eeg_raw = None
        self.sample_rate = 250
        self.current_time = 0
        self.window_size = 4.0
        self.is_playing = False
        
        # Enhanced frequency bands
        self.freq_bands = { 
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        # Multi-channel parameters
        self.selected_channels = []
        self.channel_pairs = []
        self.neural_signatures = {}
        self._colorbar = None  # Track colorbar to prevent multiplication
        
        # Brain signal enhancement
        self.brain_amplification = 1000.0
        self.noise_threshold = 1e-8
        self.adaptive_scaling = True
        
        self.setup_gui()
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Multi-Channel Neural Moiré Mapper")
        self.root.geometry("1800x1200")
        
        # Create main panels
        left_panel = ttk.Frame(self.root, width=450)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left_panel.pack_propagate(False)

        # EEG File Loading
        eeg_frame = ttk.LabelFrame(left_panel, text="Brain Signal Source")
        eeg_frame.pack(fill=tk.X, pady=5)
        ttk.Button(eeg_frame, text="Load EEG File", command=self.load_eeg_file).pack(fill=tk.X, pady=5)
        
        # Multi-Channel Selection
        channel_frame = ttk.LabelFrame(left_panel, text="Neural Channel Mapping")
        channel_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(channel_frame, text="Available Channels:").pack(anchor=tk.W)
        
        # Channel listbox with multi-selection
        listbox_frame = ttk.Frame(channel_frame)
        listbox_frame.pack(fill=tk.X, pady=5)
        
        self.channel_listbox = tk.Listbox(listbox_frame, selectmode=tk.MULTIPLE, height=8)
        channel_scroll = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.channel_listbox.yview)
        self.channel_listbox.config(yscrollcommand=channel_scroll.set)
        
        self.channel_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        channel_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Channel selection buttons
        button_frame = ttk.Frame(channel_frame)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Update Selection", command=self.update_channel_selection).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Clear All", command=self.clear_channels).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Frontal Focus", command=self.select_frontal_channels).pack(side=tk.LEFT, padx=2)
        
        # Quick presets
        preset_frame = ttk.Frame(channel_frame)
        preset_frame.pack(fill=tk.X, pady=5)
        ttk.Button(preset_frame, text="Left-Right", command=lambda: self.select_preset(['F3', 'F4', 'C3', 'C4', 'P3', 'P4'])).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Front-Back", command=lambda: self.select_preset(['Fp1', 'Fp2', 'Fz', 'Oz', 'O1', 'O2'])).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Central", command=lambda: self.select_preset(['Fz', 'Cz', 'Pz'])).pack(side=tk.LEFT, padx=2)
        
        # Selected channels display
        ttk.Label(channel_frame, text="Selected Channels:").pack(anchor=tk.W, pady=(10,0))
        self.selected_label = ttk.Label(channel_frame, text="None selected", foreground="blue")
        self.selected_label.pack(anchor=tk.W)
        
        # Channel pair display
        ttk.Label(channel_frame, text="Active Neural Pairs:").pack(anchor=tk.W, pady=(5,0))
        self.pairs_label = ttk.Label(channel_frame, text="Select channels to see pairs", foreground="green", wraplength=400)
        self.pairs_label.pack(anchor=tk.W)

        # Brain Signal Enhancement
        enhance_frame = ttk.LabelFrame(left_panel, text="Neural Signal Enhancement")
        enhance_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(enhance_frame, text="Signal Amplification:").pack(anchor=tk.W)
        self.amplification_var = tk.DoubleVar(value=1000.0)
        self.amp_scale = ttk.Scale(enhance_frame, from_=100.0, to=15000.0, variable=self.amplification_var, 
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
        
        # Adaptive processing
        self.adaptive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(enhance_frame, text="Adaptive Cross-Channel Processing", 
                       variable=self.adaptive_var, command=self.update_visualization).pack(anchor=tk.W)

        # Playback Controls
        playback_frame = ttk.LabelFrame(left_panel, text="Neural Navigation")
        playback_frame.pack(fill=tk.X, pady=5)
        
        control_buttons = ttk.Frame(playback_frame)
        control_buttons.pack(fill=tk.X, pady=5)
        self.play_button = ttk.Button(control_buttons, text="▶ Play", command=self.toggle_playback)
        self.play_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(control_buttons, text="⏮ Reset", command=self.reset_time).pack(side=tk.LEFT, padx=2)

        self.time_var = tk.DoubleVar()
        self.time_slider = ttk.Scale(playback_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                   variable=self.time_var, command=self.update_visualization)
        self.time_slider.pack(fill=tk.X, pady=5, padx=5)
        self.time_label = ttk.Label(playback_frame, text="0.0s / 0.0s")
        self.time_label.pack(anchor=tk.E, padx=5)

        # Neural Analysis Parameters
        analysis_frame = ttk.LabelFrame(left_panel, text="Neural Analysis Parameters")
        analysis_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(analysis_frame, text="Cross-Channel Sensitivity:").pack(anchor=tk.W)
        self.sensitivity_var = tk.DoubleVar(value=15.0)
        ttk.Scale(analysis_frame, from_=1.0, to=100.0, variable=self.sensitivity_var, 
                 orient=tk.HORIZONTAL, command=self.update_visualization).pack(fill=tk.X)
        
        ttk.Label(analysis_frame, text="Interference Strength:").pack(anchor=tk.W)
        self.interference_var = tk.DoubleVar(value=8.0)
        ttk.Scale(analysis_frame, from_=0.1, to=50.0, variable=self.interference_var, 
                 orient=tk.HORIZONTAL, command=self.update_visualization).pack(fill=tk.X)
        
        ttk.Label(analysis_frame, text="Pattern Resolution:").pack(anchor=tk.W)
        self.resolution_var = tk.DoubleVar(value=0.7)
        ttk.Scale(analysis_frame, from_=0.1, to=3.0, variable=self.resolution_var, 
                 orient=tk.HORIZONTAL, command=self.update_visualization).pack(fill=tk.X)
        
        # Neural signature mode
        self.signature_var = tk.StringVar(value="Cross-Channel Interference")
        ttk.Radiobutton(analysis_frame, text="Cross-Channel Interference", variable=self.signature_var, 
                       value="Cross-Channel Interference", command=self.update_visualization).pack(anchor=tk.W)
        ttk.Radiobutton(analysis_frame, text="Regional Neural Map", variable=self.signature_var, 
                       value="Regional Neural Map", command=self.update_visualization).pack(anchor=tk.W)
        ttk.Radiobutton(analysis_frame, text="Network Synchronization", variable=self.signature_var, 
                       value="Network Synchronization", command=self.update_visualization).pack(anchor=tk.W)

        # Real-time neural analysis
        neural_frame = ttk.LabelFrame(left_panel, text="Neural State Analysis")
        neural_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.neural_label = ttk.Label(neural_frame, text="Load EEG and select channels...", 
                                     font=("Consolas", 9), wraplength=420)
        self.neural_label.pack(pady=10, padx=5)

        # Visualization Panel
        right_panel = ttk.Frame(self.root)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(14, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.setup_plots()

    def setup_plots(self):
        self.ax1.set_title("Multi-Channel Brain Signals", fontweight='bold', fontsize=12)
        self.ax2.set_title("Cross-Channel Neural Moiré", fontweight='bold', fontsize=12)
        self.ax3.set_title("Neural Network Analysis", fontweight='bold', fontsize=12)
        self.ax4.set_title("3D Neural Field", fontweight='bold', fontsize=12)
        for ax in [self.ax1, self.ax2]:
            ax.set_xticks([]); ax.set_yticks([])
        self.fig.tight_layout(pad=3.0)

    def update_channel_selection(self):
        """Update selected channels from listbox"""
        selected_indices = self.channel_listbox.curselection()
        if not selected_indices:
            self.selected_channels = []
            self.channel_pairs = []
        else:
            all_channels = self.channel_listbox.get(0, tk.END)
            self.selected_channels = [all_channels[i] for i in selected_indices]
            # Generate all possible pairs for cross-channel analysis
            self.channel_pairs = list(itertools.combinations(self.selected_channels, 2))
        
        # Update display labels
        if self.selected_channels:
            self.selected_label.config(text=", ".join(self.selected_channels))
            if len(self.channel_pairs) <= 10:
                pairs_text = " | ".join([f"{p[0]}-{p[1]}" for p in self.channel_pairs])
            else:
                pairs_text = f"{len(self.channel_pairs)} neural pairs active"
            self.pairs_label.config(text=pairs_text)
        else:
            self.selected_label.config(text="None selected")
            self.pairs_label.config(text="Select channels to see pairs")
        
        if self.eeg_raw:
            self.update_visualization()

    def clear_channels(self):
        """Clear all channel selections"""
        self.channel_listbox.selection_clear(0, tk.END)
        self.update_channel_selection()

    def select_frontal_channels(self):
        """Auto-select frontal cortex channels"""
        if not hasattr(self, 'channel_listbox') or self.channel_listbox.size() == 0:
            print("Cannot select frontal channels - no EEG file loaded yet")
            return
            
        # More comprehensive frontal channel patterns
        frontal_patterns = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'AFz', 'AF3', 'AF4', 'AF7', 'AF8']
        self.select_channels_by_name(frontal_patterns)

    def select_preset(self, channel_names):
        """Select a preset group of channels"""
        self.select_channels_by_name(channel_names)

    def select_channels_by_name(self, channel_names):
        """Select channels by their names"""
        if not hasattr(self, 'channel_listbox') or self.channel_listbox.size() == 0:
            print(f"Cannot select channels - no EEG file loaded yet")
            return
            
        self.channel_listbox.selection_clear(0, tk.END)
        all_channels = list(self.channel_listbox.get(0, tk.END))
        
        selected_count = 0
        for i, channel in enumerate(all_channels):
            # More flexible matching - check if any preset channel name is contained in the actual channel name
            for preset_channel in channel_names:
                if (preset_channel.lower() in channel.lower() or 
                    channel.lower() in preset_channel.lower() or
                    preset_channel == channel):
                    self.channel_listbox.selection_set(i)
                    selected_count += 1
                    break
        
        print(f"Selected {selected_count} channels from preset: {channel_names}")
        self.update_channel_selection()

    def enhance_brain_signal(self, eeg_data):
        """Advanced brain signal enhancement for multi-channel analysis"""
        if len(eeg_data) < 100:
            return eeg_data
            
        # Remove DC offset and detrend
        eeg_data = eeg_data - np.mean(eeg_data)
        eeg_data = eeg_data - np.linspace(eeg_data[0], eeg_data[-1], len(eeg_data))
        
        # Adaptive noise reduction
        if self.adaptive_var.get():
            noise_est = np.std(eeg_data[:int(len(eeg_data)*0.1)])
            if noise_est > 0:
                threshold = noise_est * 1.5  # More aggressive for multi-channel
                eeg_data = np.sign(eeg_data) * np.maximum(np.abs(eeg_data) - threshold, 0)
        
        # Apply amplification
        eeg_data *= self.amplification_var.get()
        
        # Prevent saturation
        max_val = np.max(np.abs(eeg_data))
        if max_val > 2000:
            eeg_data = eeg_data / max_val * 2000
            
        return eeg_data

    def extract_multichannel_signals(self):
        """Extract and process signals from all selected channels"""
        if not self.selected_channels or not self.eeg_raw:
            return {}, {}
        
        duration = self.eeg_raw.n_times / self.sample_rate
        start_sample = int(max(0, self.current_time - self.window_size/2) * self.sample_rate)
        end_sample = int(min(duration, self.current_time + self.window_size/2) * self.sample_rate)
        
        channel_signals = {}
        channel_powers = {}
        
        for channel_name in self.selected_channels:
            if channel_name not in self.eeg_raw.ch_names:
                continue
                
            ch_idx = self.eeg_raw.ch_names.index(channel_name)
            eeg_data, _ = self.eeg_raw[ch_idx, start_sample:end_sample]
            eeg_data = eeg_data.flatten()
            
            if len(eeg_data) < 100:
                continue
            
            # Enhance signal
            enhanced_data = self.enhance_brain_signal(eeg_data)
            
            # Extract frequency bands
            band_signals, band_powers = self.extract_brain_frequencies(enhanced_data)
            
            channel_signals[channel_name] = band_signals
            channel_powers[channel_name] = band_powers
        
        return channel_signals, channel_powers

    def extract_brain_frequencies(self, eeg_data):
        """Extract frequency bands from enhanced EEG data"""
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
                b, a = butter(6, [low_norm, high_norm], btype='band')
                filtered = filtfilt(b, a, eeg_data)
                
                # Extract envelope and power
                analytic_signal = hilbert(filtered)
                envelope = np.abs(analytic_signal)
                power = np.mean(envelope**2)
                
                band_signals[band_name] = filtered
                band_powers[band_name] = power
                
            except Exception as e:
                print(f"Error processing {band_name}: {e}")
                band_signals[band_name] = np.zeros_like(eeg_data)
                band_powers[band_name] = 0
        
        return band_signals, band_powers

    def create_cross_channel_moire(self, channel1_signals, channel2_signals, channel1_name, channel2_name):
        """Create moiré interference pattern between two channels"""
        grid_size = 400
        x, y = np.linspace(-60, 60, grid_size), np.linspace(-60, 60, grid_size)
        X, Y = np.meshgrid(x, y)
        
        moire_pattern = np.zeros_like(X)
        
        # Get focus frequency band
        focus = self.freq_focus_var.get()
        if focus == "all":
            focus = "theta"  # Default to theta for cross-channel analysis
        
        if focus in channel1_signals and focus in channel2_signals:
            signal1 = channel1_signals[focus]
            signal2 = channel2_signals[focus]
            
            if len(signal1) > 0 and len(signal2) > 0:
                # Calculate relative phase and amplitude differences
                power1 = np.var(signal1)
                power2 = np.var(signal2)
                phase_diff = np.angle(np.mean(hilbert(signal1)) * np.conj(np.mean(hilbert(signal2))))
                
                # Create hexagonal grids with different orientations based on channel characteristics
                sensitivity = self.sensitivity_var.get()
                
                # Grid 1 based on channel 1
                spacing1 = 8.0 + power1 * sensitivity * 0.01
                omega1 = (4 * np.pi) / (spacing1 * np.sqrt(3))
                grid1 = np.sum([np.cos(omega1 * (np.cos(r)*X + np.sin(r)*Y) + power1 * 10) 
                               for r in np.deg2rad([0, 60, 120])], axis=0)
                
                # Grid 2 based on channel 2 with phase offset
                spacing2 = 8.0 + power2 * sensitivity * 0.01
                omega2 = (4 * np.pi) / (spacing2 * np.sqrt(3))
                rotation = phase_diff + np.pi/12  # Small rotation for interference
                grid2 = np.sum([np.cos(omega2 * (np.cos(r + rotation)*X + np.sin(r + rotation)*Y) + power2 * 10) 
                               for r in np.deg2rad([0, 60, 120])], axis=0)
                
                # Create interference pattern
                interference_strength = self.interference_var.get()
                moire_pattern = np.tanh((grid1 + grid2) * interference_strength * 0.1)
                
                # Apply dynamic modulation based on signal correlation
                correlation = np.corrcoef(signal1[:min(len(signal1), len(signal2))], 
                                        signal2[:min(len(signal1), len(signal2))])[0, 1]
                if not np.isnan(correlation):
                    moire_pattern *= (1 + correlation * 0.5)
        
        # Apply spatial smoothing
        resolution = self.resolution_var.get()
        if resolution != 1.0:
            sigma = (3.0 - resolution) * 1.5
            moire_pattern = gaussian_filter(moire_pattern, sigma=sigma)
        
        return moire_pattern

    def generate_neural_moire_field(self, channel_signals, channel_powers):
        """Generate neural field from all channel pairs"""
        grid_size = 400
        neural_field = np.zeros((grid_size, grid_size))
        
        signature_mode = self.signature_var.get()
        pair_count = 0
        total_neural_power = 0
        
        if signature_mode == "Cross-Channel Interference":
            # Generate interference patterns for all channel pairs
            for i, (ch1, ch2) in enumerate(self.channel_pairs[:10]):  # Limit to first 10 pairs
                if ch1 in channel_signals and ch2 in channel_signals:
                    pair_moire = self.create_cross_channel_moire(
                        channel_signals[ch1], channel_signals[ch2], ch1, ch2)
                    neural_field += pair_moire
                    pair_count += 1
                    
                    # Calculate neural power for this pair
                    focus = self.freq_focus_var.get() if self.freq_focus_var.get() != "all" else "theta"
                    if focus in channel_powers[ch1] and focus in channel_powers[ch2]:
                        pair_power = np.sqrt(channel_powers[ch1][focus] * channel_powers[ch2][focus])
                        total_neural_power += pair_power
            
            if pair_count > 0:
                neural_field /= pair_count
                
        elif signature_mode == "Regional Neural Map":
            # Create regional patterns based on brain anatomy
            frontal_channels = [ch for ch in self.selected_channels if any(x in ch for x in ['Fp', 'F', 'FC'])]
            central_channels = [ch for ch in self.selected_channels if any(x in ch for x in ['C', 'Cz'])]
            parietal_channels = [ch for ch in self.selected_channels if any(x in ch for x in ['P', 'Pz'])]
            
            regions = {'Frontal': frontal_channels, 'Central': central_channels, 'Parietal': parietal_channels}
            
            for region_name, region_channels in regions.items():
                if len(region_channels) >= 2:
                    # Average signals in this region
                    region_pairs = list(itertools.combinations(region_channels, 2))[:3]  # Max 3 pairs per region
                    for ch1, ch2 in region_pairs:
                        if ch1 in channel_signals and ch2 in channel_signals:
                            region_moire = self.create_cross_channel_moire(
                                channel_signals[ch1], channel_signals[ch2], ch1, ch2)
                            neural_field += region_moire * 0.33  # Weight by region
                            pair_count += 1
                            
        else:  # Network Synchronization
            # Calculate global synchronization patterns
            if len(self.selected_channels) >= 3:
                # Create a global synchronization field
                all_signals = []
                focus = self.freq_focus_var.get() if self.freq_focus_var.get() != "all" else "theta"
                
                for ch in self.selected_channels:
                    if ch in channel_signals and focus in channel_signals[ch]:
                        signal = channel_signals[ch][focus]
                        if len(signal) > 0:
                            all_signals.append(signal)
                
                if len(all_signals) >= 2:
                    # Calculate cross-correlation matrix
                    min_len = min(len(s) for s in all_signals)
                    sync_matrix = np.corrcoef([s[:min_len] for s in all_signals])
                    global_sync = np.nanmean(sync_matrix[np.triu_indices_from(sync_matrix, k=1)])
                    
                    # Create synchronized pattern
                    x, y = np.linspace(-60, 60, grid_size), np.linspace(-60, 60, grid_size)
                    X, Y = np.meshgrid(x, y)
                    neural_field = np.sin(X * global_sync * 10) * np.cos(Y * global_sync * 10)
                    neural_field = np.exp(neural_field * self.sensitivity_var.get() * 0.1)
                    total_neural_power = abs(global_sync) * len(all_signals)
        
        analysis_text = f"{signature_mode}\n"
        analysis_text += f"Active Channels: {len(self.selected_channels)}\n"
        analysis_text += f"Neural Pairs: {len(self.channel_pairs)}\n"
        analysis_text += f"Total Power: {total_neural_power:.3e}\n"
        analysis_text += f"Sensitivity: {self.sensitivity_var.get():.1f}\n"
        
        if pair_count > 0:
            analysis_text += f"Processing {pair_count} interference patterns"
        else:
            analysis_text += "Select 2+ channels for neural mapping"
        
        self.neural_label.config(text=analysis_text)
        return neural_field

    def update_visualization(self, value=None):
        if not self.eeg_raw: 
            print("No EEG data loaded")
            return
            
        if not self.selected_channels:
            print("No channels selected for analysis")
            return
        
        self.current_time = self.time_var.get()
        duration = self.eeg_raw.n_times / self.sample_rate
        self.time_label.config(text=f"{self.current_time:.1f}s / {duration:.1f}s")
        self.amp_label.config(text=f"{self.amplification_var.get():.0f}x")

        # Extract multi-channel signals
        channel_signals, channel_powers = self.extract_multichannel_signals()
        
        if not channel_signals:
            print("No valid channel signals extracted")
            return

        # Generate neural moiré field
        neural_field = self.generate_neural_moire_field(channel_signals, channel_powers)
        
        # Clear and update plots with aggressive colorbar cleanup
        # First, remove ALL colorbars from the figure
        try:
            # Get all axes and remove any that are colorbars
            all_axes = list(self.fig.axes)
            main_axes = [self.ax1, self.ax2, self.ax3, self.ax4]
            
            for ax in all_axes:
                if ax not in main_axes:
                    try:
                        ax.remove()
                    except:
                        pass
            
            # Clear collections from ax3 that might have colorbars
            if hasattr(self.ax3, 'images'):
                for img in self.ax3.images:
                    if hasattr(img, 'colorbar') and img.colorbar is not None:
                        try:
                            img.colorbar.remove()
                        except:
                            pass
        except:
            pass
        
        # Clear the main plots
        self.ax1.clear()
        self.ax2.clear() 
        self.ax3.clear()
        self.ax4.clear()
        
        # Reset colorbar tracking
        self._colorbar = None
            
        self.setup_plots()

        # Plot 1: Multi-channel brain signals
        colors = {'theta': 'orange', 'alpha': 'green', 'beta': 'blue', 'gamma': 'red'}
        channel_colors = plt.cm.tab10(np.linspace(0, 1, len(self.selected_channels)))
        
        offset = 0
        max_offset = 0
        for i, channel_name in enumerate(self.selected_channels[:8]):  # Show max 8 channels
            if channel_name in channel_signals:
                focus = self.freq_focus_var.get() if self.freq_focus_var.get() != "all" else "theta"
                if focus in channel_signals[channel_name]:
                    signal = channel_signals[channel_name][focus]
                    if len(signal) > 0:
                        time_axis = np.linspace(0, len(signal)/self.sample_rate, len(signal))
                        power = channel_powers[channel_name][focus] if focus in channel_powers[channel_name] else 0
                        
                        self.ax1.plot(time_axis, signal + offset, 
                                    color=channel_colors[i], 
                                    label=f'{channel_name}: {power:.1e}', 
                                    alpha=0.8, linewidth=1.5)
                        offset += np.max(np.abs(signal)) * 2.2
                        max_offset = offset
        
        self.ax1.set_title(f"Multi-Channel Brain Signals ({self.freq_focus_var.get().upper()})", fontweight='bold')
        self.ax1.legend(loc='upper right', fontsize=8, ncol=2)
        self.ax1.grid(True, alpha=0.2)
        self.ax1.set_xlim(0, self.window_size)

        # Plot 2: Cross-channel neural moiré
        self.ax2.imshow(neural_field, cmap='plasma', origin='lower', aspect='auto', 
                       extent=[-60, 60, -60, 60])
        self.ax2.set_title(f"Neural Moiré Field ({len(self.channel_pairs)} pairs)", fontweight='bold')

        # Plot 3: Neural network analysis
        if len(self.selected_channels) >= 2:
            # Create neural power matrix
            focus = self.freq_focus_var.get() if self.freq_focus_var.get() != "all" else "theta"
            power_matrix = np.zeros((len(self.selected_channels), len(self.selected_channels)))
            
            for i, ch1 in enumerate(self.selected_channels):
                for j, ch2 in enumerate(self.selected_channels):
                    if ch1 in channel_powers and ch2 in channel_powers:
                        if focus in channel_powers[ch1] and focus in channel_powers[ch2]:
                            # Calculate neural coupling strength
                            power1 = channel_powers[ch1][focus]
                            power2 = channel_powers[ch2][focus]
                            coupling = np.sqrt(power1 * power2)
                            power_matrix[i, j] = coupling
            
            # Visualize neural network
            im = self.ax3.imshow(power_matrix, cmap='viridis', aspect='auto')
            self.ax3.set_xticks(range(len(self.selected_channels)))
            self.ax3.set_yticks(range(len(self.selected_channels)))
            self.ax3.set_xticklabels(self.selected_channels, rotation=45, ha='right', fontsize=8)
            self.ax3.set_yticklabels(self.selected_channels, fontsize=8)
            self.ax3.set_title(f"Neural Coupling Matrix ({focus.upper()})", fontweight='bold')
            
            # Create colorbar with complete cleanup first
            try:
                # Remove any existing colorbar from this specific image
                if hasattr(im, 'colorbar') and im.colorbar is not None:
                    im.colorbar.remove()
                
                # Create new colorbar using figure method with specific positioning
                divider = None
                try:
                    from mpl_toolkits.axes_grid1 import make_axes_locatable
                    divider = make_axes_locatable(self.ax3)
                    cax = divider.append_axes("right", size="5%", pad=0.1)
                    self._colorbar = self.fig.colorbar(im, cax=cax)
                except ImportError:
                    # Fallback if axes_grid1 not available
                    self._colorbar = self.fig.colorbar(im, ax=self.ax3, shrink=0.8, aspect=20)
                
                self._colorbar.set_label('Coupling Strength', rotation=270, labelpad=15)
                
            except Exception as e:
                print(f"Colorbar error (non-critical): {e}")
                # If colorbar fails, just continue without it
                pass
        else:
            self.ax3.text(0.5, 0.5, 'Select 2+ channels\nfor network analysis', 
                         ha='center', va='center', transform=self.ax3.transAxes, fontsize=12)

        # Plot 4: 3D neural field
        try:
            self.ax4.remove()
            self.ax4 = self.fig.add_subplot(2, 2, 4, projection='3d')
            
            # Downsample for 3D visualization
            field_3d = neural_field[::15, ::15]
            x_3d = np.arange(field_3d.shape[1]) * 3
            y_3d = np.arange(field_3d.shape[0]) * 3
            X_3d, Y_3d = np.meshgrid(x_3d, y_3d)
            
            # Create surface plot
            surf = self.ax4.plot_surface(X_3d, Y_3d, field_3d, cmap='plasma', alpha=0.8, 
                                       linewidth=0, antialiased=True)
            self.ax4.set_title("3D Neural Landscape", fontweight='bold')
            self.ax4.set_xlabel("Spatial Dimension 1")
            self.ax4.set_ylabel("Spatial Dimension 2")
            self.ax4.set_zlabel("Neural Intensity")
            
            # Set viewing angle for better visualization
            self.ax4.view_init(elev=30, azim=45)
        except Exception as e:
            print(f"3D plot error (non-critical): {e}")

        # Update the figure
        try:
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"Canvas draw error (non-critical): {e}")

    def load_eeg_file(self):
        """Load EEG file and populate channel list"""
        filepath = filedialog.askopenfilename(title="Select EEG File", 
                                            filetypes=[("EDF files", "*.edf"), ("All files", "*.*")])
        if not filepath: return
        try:
            self.eeg_raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            self.sample_rate = self.eeg_raw.info['sfreq']
            duration = self.eeg_raw.n_times / self.sample_rate
            all_channels = self.eeg_raw.ch_names
            
            # Populate channel listbox
            self.channel_listbox.delete(0, tk.END)
            for channel in all_channels:
                self.channel_listbox.insert(tk.END, channel)
            
            self.time_slider.config(to=duration)
            messagebox.showinfo("Success", 
                              f"Multi-channel neural data loaded!\n"
                              f"Duration: {duration:.1f}s\n"
                              f"Sample rate: {self.sample_rate} Hz\n"
                              f"Channels: {len(all_channels)}\n\n"
                              f"Select multiple channels for neural mapping!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load neural data:\n{e}")

    def toggle_playback(self):
        """Toggle neural navigation playback"""
        if not self.selected_channels:
            print("No channels selected - stopping playback")
            return
            
        self.is_playing = not self.is_playing
        self.play_button.config(text="⏸ Pause" if self.is_playing else "▶ Play")
        if self.is_playing: 
            self.start_playback()

    def start_playback(self):
        """Start neural navigation playback"""
        if not self.is_playing or not self.eeg_raw: 
            return
            
        if not self.selected_channels:
            print("No channels selected - stopping playback")
            self.is_playing = False
            self.play_button.config(text="▶ Play")
            return
            
        current_time = self.time_var.get()
        max_time = self.eeg_raw.n_times / self.sample_rate
        if current_time >= max_time:
            self.is_playing = False
            self.play_button.config(text="▶ Play")
            return
            
        new_time = min(current_time + 0.3, max_time)  # Slightly slower for neural analysis
        self.time_var.set(new_time)
        self.update_visualization()
        self.root.after(300, self.start_playback)

    def reset_time(self):
        """Reset neural navigation to beginning"""
        self.time_var.set(0)
        self.is_playing = False
        self.play_button.config(text="▶ Play")
        if self.eeg_raw and self.selected_channels:
            self.update_visualization()

    def run(self):
        """Launch the Multi-Channel Neural Mapper"""
        print("🧠🌊 Multi-Channel Neural Moiré Mapper v6 🌊🧠")
        print("=" * 60)
        print("Discover neural patterns through cross-channel interference!")
        print("")
        print("Features:")
        print("🔸 Multi-channel neural mapping")
        print("🔸 Cross-channel moiré interference patterns") 
        print("🔸 Regional brain network analysis")
        print("🔸 Real-time neural signature detection")
        print("🔸 3D neural landscape visualization")
        print("🔸 Individual neural fingerprinting")
        print("")
        print("Usage:")
        print("1. Load your multi-channel EEG file")
        print("2. Select multiple brain channels")
        print("3. Watch neural patterns emerge through interference!")
        print("=" * 60)
        self.root.mainloop()

def main():
    """Launch the neural mapper"""
    mapper = MultiChannelNeuralMapper()
    mapper.run()

if __name__ == "__main__":
    main()