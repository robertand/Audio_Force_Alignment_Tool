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

    def parse_srt(self, file_path):
        import re
        segments = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()

        # Regex to match SRT blocks
        # Support both \n and \r\n
        pattern = re.compile(r'(\d+)\s*\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s*\n(.*?)(?=\n\n|\n\r\n|\r\n\r\n|\n*$)', re.DOTALL)

        def time_to_seconds(t_str):
            h, m, s_ms = t_str.split(':')
            s, ms = s_ms.split(',')
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

        for match in pattern.finditer(content):
            start_str = match.group(2)
            end_str = match.group(3)
            text = match.group(4).strip()

            # Robust speaker detection
            speaker = "Unknown"
            # Look for "Name: " at the start of the line or the entire text
            speaker_match = re.match(r'^([^:\n\[\]]+):\s*(.*)', text, re.DOTALL)
            if speaker_match:
                potential_speaker = speaker_match.group(1).strip()
                # If it looks like a name (not too many words and not just tags)
                if 0 < len(potential_speaker.split()) <= 3:
                    speaker = potential_speaker
                    text = speaker_match.group(2).strip()

            segments.append({
                'start': time_to_seconds(start_str),
                'end': time_to_seconds(end_str),
                'speaker': speaker,
                'text_ro': text,
                'text_en': ""
            })
        return segments

    def reconstruct_character_track(self, aligned_segments, sr, total_duration=None):
        """
        Place aligned audio segments on a silent track at their target timecodes.
        aligned_segments: list of (target_start, target_end, audio_segment)
        """
        if not aligned_segments and total_duration is None:
            return np.array([])

        # Find required total duration
        if total_duration is None:
            total_duration = max(s[1] for s in aligned_segments) if aligned_segments else 0

        full_track = np.zeros(int(total_duration * sr))

        for target_start, target_end, audio_seg in aligned_segments:
            start_sample = int(target_start * sr)
            end_sample = start_sample + len(audio_seg)

            # Ensure we don't exceed track length
            if end_sample > len(full_track):
                end_sample = len(full_track)
                audio_seg = audio_seg[:end_sample - start_sample]

            # Place audio segment (or use a small crossfade/add to existing)
            full_track[start_sample:end_sample] = audio_seg

        return full_track