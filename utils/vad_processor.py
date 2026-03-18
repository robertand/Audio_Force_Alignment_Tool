import webrtcvad
import numpy as np
import collections
import struct

class VADProcessor:
    def __init__(self, aggressiveness=2, frame_duration=30):
        """
        Initialize VAD processor
        aggressiveness: 0-3, higher = more aggressive filtering
        frame_duration: ms per frame (must be 10, 20, or 30)
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_duration = frame_duration
        self.frame_size = int(16000 * frame_duration / 1000)  # 16kHz sample rate
        
    def process_vad(self, audio, sr, min_speech_duration=0.1, min_silence_duration=0.1):
        """
        Process audio with VAD to detect speech segments
        Returns list of (start_time, end_time) for speech segments
        """
        # Convert to 16-bit PCM if necessary
        if audio.dtype != np.int16:
            # Normalize and convert to int16
            audio = (audio * 32767).astype(np.int16)
        
        # Resample to 16kHz if necessary
        if sr != 16000:
            # Simple resampling (you might want to use librosa here)
            import librosa
            audio = librosa.resample(audio.astype(float), orig_sr=sr, target_sr=16000)
            audio = (audio * 32767).astype(np.int16)
            sr = 16000
        
        # Process frames
        speech_flags = []
        for i in range(0, len(audio) - self.frame_size, self.frame_size):
            frame = audio[i:i + self.frame_size]
            is_speech = self.vad.is_speech(frame.tobytes(), sr)
            speech_flags.append(is_speech)
        
        # Group frames into segments
        segments = []
        in_speech = False
        start_frame = 0
        
        min_speech_frames = int(min_speech_duration * 1000 / self.frame_duration)
        min_silence_frames = int(min_silence_duration * 1000 / self.frame_duration)
        
        for i, is_speech in enumerate(speech_flags):
            if is_speech and not in_speech:
                in_speech = True
                start_frame = i
            elif not is_speech and in_speech:
                # Check if speech duration meets minimum
                if i - start_frame >= min_speech_frames:
                    start_time = start_frame * self.frame_duration / 1000
                    end_time = i * self.frame_duration / 1000
                    segments.append((start_time, end_time))
                in_speech = False
        
        # Handle last segment
        if in_speech and len(speech_flags) - start_frame >= min_speech_frames:
            start_time = start_frame * self.frame_duration / 1000
            end_time = len(speech_flags) * self.frame_duration / 1000
            segments.append((start_time, end_time))
        
        return segments
    
    def validate_speech_boundaries(self, audio, sr, proposed_start, proposed_end, 
                                   padding=0.05):
        """
        Validate that proposed boundaries contain speech
        Returns adjusted boundaries if needed
        """
        # Convert to appropriate format for VAD
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)
        
        # Analyze region around boundaries
        analysis_window = 0.2  # 200ms around each boundary
        
        # Check start boundary
        start_region_start = max(0, proposed_start - analysis_window)
        start_region_end = min(proposed_end, proposed_start + analysis_window)
        
        start_samples = int(start_region_start * sr)
        start_end_samples = int(start_region_end * sr)
        
        if start_end_samples > start_samples:
            start_region = audio[start_samples:start_end_samples]
            start_vad = self._analyze_region(start_region, sr)
            
            # Find first speech in this region
            first_speech = self._find_first_speech(start_region, sr)
            if first_speech is not None:
                proposed_start = start_region_start + first_speech
                # Add small padding
                proposed_start = max(0, proposed_start - padding)
        
        # Check end boundary
        end_region_start = max(proposed_start, proposed_end - analysis_window)
        end_region_end = min(len(audio)/sr, proposed_end + analysis_window)
        
        end_start_samples = int(end_region_start * sr)
        end_end_samples = int(end_region_end * sr)
        
        if end_end_samples > end_start_samples:
            end_region = audio[end_start_samples:end_end_samples]
            
            # Find last speech in this region
            last_speech = self._find_last_speech(end_region, sr)
            if last_speech is not None:
                proposed_end = end_region_start + last_speech
                # Add small padding
                proposed_end = min(len(audio)/sr, proposed_end + padding)
        
        return proposed_start, proposed_end
    
    def _analyze_region(self, audio_region, sr):
        """Analyze a region with VAD"""
        flags = []
        for i in range(0, len(audio_region) - self.frame_size, self.frame_size):
            frame = audio_region[i:i + self.frame_size]
            flags.append(self.vad.is_speech(frame.tobytes(), sr))
        return flags
    
    def _find_first_speech(self, audio_region, sr):
        """Find first speech frame in region"""
        for i in range(0, len(audio_region) - self.frame_size, self.frame_size):
            frame = audio_region[i:i + self.frame_size]
            if self.vad.is_speech(frame.tobytes(), sr):
                return i / sr
        return None
    
    def _find_last_speech(self, audio_region, sr):
        """Find last speech frame in region"""
        for i in range(len(audio_region) - self.frame_size, 0, -self.frame_size):
            frame = audio_region[i:i + self.frame_size]
            if self.vad.is_speech(frame.tobytes(), sr):
                return (i + self.frame_size) / sr
        return None