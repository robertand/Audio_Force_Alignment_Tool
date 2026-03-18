import torch
import torchaudio
import numpy as np
import soundfile as sf
import librosa
from scipy import signal

class AudioProcessor:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_audio(self, file_path):
        """Load audio file and resample if necessary"""
        try:
            # Try torchaudio first
            waveform, sr = torchaudio.load(file_path)
            waveform = waveform.mean(dim=0).numpy()  # Convert to mono
            
            if sr != self.target_sr:
                # Resample using librosa
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
                
        except:
            # Fallback to soundfile
            waveform, sr = sf.read(file_path)
            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=1)  # Convert to mono
            
            if sr != self.target_sr:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
        
        # Normalize
        waveform = waveform / np.max(np.abs(waveform) + 1e-8)
        
        return waveform, sr
    
    def find_best_cut_point(self, audio, sr, target_time, search_window=0.1):
        """
        Find best point to cut audio near target_time by looking for
        zero-crossings or lowest energy points.
        """
        target_sample = int(target_time * sr)
        window_samples = int(search_window * sr)
        
        start_search = max(0, target_sample - window_samples // 2)
        end_search = min(len(audio), target_sample + window_samples // 2)
        
        search_region = audio[start_search:end_search]

        if len(search_region) < 2:
            return target_sample

        # 1. Look for zero-crossings
        zero_crossings = np.where(np.diff(np.sign(search_region)))[0]
        if len(zero_crossings) > 0:
            # Find the closest zero-crossing to our target sample
            closest_idx = zero_crossings[np.argmin(np.abs(zero_crossings + start_search - target_sample))]
            return start_search + closest_idx

        # 2. Fallback: lowest absolute value (near zero)
        min_idx = np.argmin(np.abs(search_region))
        return start_search + min_idx

    def extract_segment(self, audio, sr, start_time, end_time, text=""):
        """Extract audio segment with zero-crossing alignment and interjection padding"""

        # Check for interjections in brackets like [laughs]
        padding = 0.05  # Standard 50ms padding
        if '[' in text and ']' in text:
            padding = 0.15  # Extra 150ms for interjections

        # Apply padding to boundaries
        start_time = max(0, start_time - padding)
        end_time = end_time + padding

        # Refine boundaries with waveform analysis
        start_sample = self.find_best_cut_point(audio, sr, start_time)
        end_sample = self.find_best_cut_point(audio, sr, end_time)

        # Extract
        segment = audio[start_sample:end_sample]
        
        # Apply small fade in/out to avoid clicks
        fade_length = int(0.01 * sr)  # 10ms fade
        if len(segment) > 2 * fade_length:
            fade_in = np.linspace(0, 1, fade_length)
            fade_out = np.linspace(1, 0, fade_length)
            segment[:fade_length] *= fade_in
            segment[-fade_length:] *= fade_out
            
        return segment
    
    def detect_silence(self, audio, sr, threshold=0.01, min_silence_duration=0.1):
        """Detect silence regions in audio"""
        # Compute energy
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        energy = librosa.feature.rms(
            y=audio, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]
        
        # Normalize energy
        energy = energy / np.max(energy + 1e-8)
        
        # Find silence frames
        silence_frames = energy < threshold
        
        # Convert frames to time
        times = librosa.frames_to_time(
            np.arange(len(silence_frames)), 
            sr=sr, 
            hop_length=hop_length
        )
        
        # Group silence frames into segments
        silence_segments = []
        in_silence = False
        start_time = 0
        
        for i, is_silence in enumerate(silence_frames):
            if is_silence and not in_silence:
                in_silence = True
                start_time = times[i]
            elif not is_silence and in_silence:
                in_silence = False
                duration = times[i] - start_time
                if duration >= min_silence_duration:
                    silence_segments.append((start_time, times[i]))
        
        # Handle last segment
        if in_silence:
            duration = times[-1] - start_time
            if duration >= min_silence_duration:
                silence_segments.append((start_time, times[-1]))
        
        return silence_segments