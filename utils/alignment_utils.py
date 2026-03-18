import numpy as np
from scipy import signal
import soundfile as sf
import os

class AlignmentUtils:
    def __init__(self):
        pass
    
    def validate_segments(self, expected_segments, vad_segments, sr, 
                          tolerance=0.05, min_segment_duration=0.5):
        """
        Validate and adjust expected segments using VAD segments
        """
        validated = []
        
        for seg in expected_segments:
            original_start = seg['start']
            original_end = seg['end']
            
            # Find closest VAD segment
            best_overlap = 0
            best_vad = None
            
            for vad_start, vad_end in vad_segments:
                # Calculate overlap
                overlap_start = max(seg['start'], vad_start)
                overlap_end = min(seg['end'], vad_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_vad = (vad_start, vad_end)
            
            adjusted_start = seg['start']
            adjusted_end = seg['end']
            vad_confidence = 1.0
            
            if best_vad and best_overlap > 0:
                vad_start, vad_end = best_vad
                
                # Adjust boundaries to include the VAD segment
                adjusted_start = min(seg['start'], vad_start)
                adjusted_end = max(seg['end'], vad_end)
                
                # Calculate confidence based on overlap
                vad_confidence = best_overlap / (seg['end'] - seg['start'])
                
                # Don't extend too far
                max_extension = 0.5  # Maximum 500ms extension
                adjusted_start = max(adjusted_start, seg['start'] - max_extension)
                adjusted_end = min(adjusted_end, seg['end'] + max_extension)
            
            # Ensure minimum duration
            duration = adjusted_end - adjusted_start
            if duration < min_segment_duration:
                # Extend to minimum duration
                center = (adjusted_start + adjusted_end) / 2
                adjusted_start = center - min_segment_duration / 2
                adjusted_end = center + min_segment_duration / 2
            
            # Add segment
            validated_seg = seg.copy()
            validated_seg.update({
                'start': adjusted_start,
                'end': adjusted_end,
                'original_start': original_start,
                'original_end': original_end,
                'vad_confidence': vad_confidence
            })
            validated.append(validated_seg)
        
        return validated
    
    def apply_crossfade(self, audio, sr, fade_duration=0.01):
        """Apply crossfade to audio segment"""
        fade_samples = int(fade_duration * sr)
        
        if len(audio) > 2 * fade_samples:
            # Create fade curves
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            
            # Apply fades
            audio_copy = audio.copy()
            audio_copy[:fade_samples] *= fade_in
            audio_copy[-fade_samples:] *= fade_out
            
            return audio_copy
        
        return audio
    
    def concatenate_segments(self, segment_paths, crossfade_duration=0.01):
        """Concatenate multiple audio segments with crossfade"""
        segments = []
        sample_rate = None
        
        # Load all segments
        for path in segment_paths:
            audio, sr = sf.read(path)
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                # Resample if necessary (simplified - you might want better resampling)
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            
            segments.append(audio)
        
        if len(segments) == 0:
            return np.array([])
        
        if len(segments) == 1:
            return segments[0]
        
        # Concatenate with crossfade
        fade_samples = int(crossfade_duration * sample_rate)
        output = segments[0]
        
        for next_seg in segments[1:]:
            # Apply crossfade between segments
            if len(output) > fade_samples and len(next_seg) > fade_samples:
                # Create fade curves
                fade_out = np.linspace(1, 0, fade_samples)
                fade_in = np.linspace(0, 1, fade_samples)
                
                # Apply fades
                output[-fade_samples:] *= fade_out
                next_seg[:fade_samples] *= fade_in
                
                # Concatenate
                output = np.concatenate([output, next_seg])
            else:
                # Simple concatenation if segments are too short
                output = np.concatenate([output, next_seg])
        
        return output
    
    def detect_interjections(self, text):
        """Detect and mark interjections in text"""
        interjection_patterns = [
            '[exhales]', '[laughs]', '[sighs]', '[coughs]',
            '[whispers]', '[gasps]', '[clears throat]'
        ]
        
        interjections = []
        for pattern in interjection_patterns:
            if pattern in text.lower():
                interjections.append(pattern)
        
        return interjections
    
    def create_timeline(self, segments):
        """Create a timeline visualization data"""
        timeline = []
        for i, seg in enumerate(segments):
            timeline.append({
                'index': i,
                'start': seg['start'],
                'end': seg['end'],
                'speaker': seg['speaker'],
                'text': seg.get('text_en', '')[:50] + '...' if len(seg.get('text_en', '')) > 50 else seg.get('text_en', ''),
                'duration': seg['end'] - seg['start']
            })
        return timeline