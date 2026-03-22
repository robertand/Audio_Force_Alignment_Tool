import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
import librosa
import difflib
from utils.text_utils import normalize_text, strip_diacritics
import logging

logger = logging.getLogger(__name__)

class MMSDialogueAligner:
    def __init__(self, model_id="MahmoudAshraf/mms-300m-1130-forced-aligner", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading MMS aligner on {self.device}")
        
        # Load model and processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
        
        # Move to GPU if available
        if self.device == "cuda":
            self.model = self.model.to(self.device)
            # Use mixed precision for faster inference
            self.model = self.model.half() if torch.cuda.get_device_capability()[0] >= 7 else self.model
        
        self.target_sr = 16000
        
        # Alignment parameters
        self.search_window_sec = 60.0
        self.overlap_sec = 3.0
        self.min_confidence = 0.35

    def align_character_audio(self, audio_path, expected_segments, language="ron", **kwargs):
        """
        Align audio with expected segments using MMS model with Segmented Local Alignment.
        """
        # Mapping common lang codes
        lang_map = {
            'ro': 'ron',
            'ron': 'ron',
            'en': 'eng',
            'eng': 'eng'
        }
        language = lang_map.get(language, language)

        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.target_sr)
        total_audio_duration = len(audio) / self.target_sr

        aligned_results = []
        current_audio_ptr = 0.0  # Time in seconds in the input audio file

        for expected in expected_segments:
            expected_text = expected.get('text_ro', '').strip()
            expected_duration = expected['end'] - expected['start']

            if not expected_text:
                start = min(current_audio_ptr, total_audio_duration)
                end = min(current_audio_ptr + expected_duration, total_audio_duration)
                aligned_results.append({
                    'original': expected,
                    'aligned': {
                        'start': start, 
                        'end': end, 
                        'text': "", 
                        'confidence': 1.0
                    }
                })
                current_audio_ptr = end
                continue

            # Search locally in the chunk
            search_start_time = max(0.0, current_audio_ptr - self.overlap_sec)
            search_end_time = min(
                total_audio_duration, 
                search_start_time + self.search_window_sec
            )

            start_sample = int(search_start_time * self.target_sr)
            end_sample = int(search_end_time * self.target_sr)
            chunk = audio[start_sample:end_sample]

            best_match = None
            highest_ratio = 0.0

            if len(chunk) >= self.target_sr:  # At least 1 second
                # Select language for MMS
                if hasattr(self.processor.tokenizer, 'set_target_lang'):
                    try:
                        self.processor.tokenizer.set_target_lang(language)
                    except Exception as e:
                        logger.warning(f"Failed to set target language {language}: {e}")

                # Process with GPU
                inputs = self.processor(
                    chunk, 
                    sampling_rate=self.target_sr, 
                    return_tensors="pt"
                )
                
                # Move inputs to same device as model
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    if self.device == "cuda" and hasattr(self.model, 'half'):
                        # Use mixed precision
                        with torch.cuda.amp.autocast():
                            logits = self.model(**inputs).logits[0]
                    else:
                        logits = self.model(**inputs).logits[0]

                # Get predictions
                predicted_ids = torch.argmax(logits, dim=-1)
                tokens = self.processor.tokenizer.convert_ids_to_tokens(
                    predicted_ids.cpu().tolist()
                )

                # Extract segments
                local_segments = self._extract_segments(
                    tokens, 
                    logits.shape[0], 
                    len(chunk), 
                    search_start_time
                )

                # Find best match
                norm_expected = normalize_text(expected_text)
                strip_expected = strip_diacritics(norm_expected)

                for seg in local_segments:
                    norm_trans = normalize_text(seg['text'])
                    ratio = difflib.SequenceMatcher(
                        None, norm_expected, norm_trans
                    ).ratio()
                    
                    strip_ratio = difflib.SequenceMatcher(
                        None, strip_expected, strip_diacritics(norm_trans)
                    ).ratio()
                    
                    combined_ratio = max(ratio, strip_ratio * 0.9)

                    if combined_ratio > highest_ratio:
                        highest_ratio = combined_ratio
                        best_match = seg

            # Results and fallback logic
            if highest_ratio >= self.min_confidence and best_match:
                aligned_results.append({
                    'original': expected,
                    'aligned': {
                        'start': best_match['start'],
                        'end': best_match['end'],
                        'text': best_match['text'],
                        'confidence': highest_ratio
                    }
                })
                current_audio_ptr = best_match['end']
            else:
                # Fallback: Use expected duration from current position
                start = min(current_audio_ptr, total_audio_duration)
                end = min(current_audio_ptr + expected_duration, total_audio_duration)
                aligned_results.append({
                    'original': expected,
                    'aligned': {
                        'start': start, 
                        'end': end, 
                        'text': "", 
                        'confidence': 0.0
                    }
                })
                current_audio_ptr = end

        return aligned_results

    def _extract_segments(self, tokens, num_frames, chunk_length, base_time):
        """Extract word segments from token sequence"""
        segments = []
        curr_text = ""
        curr_start = 0.0
        chunk_duration = chunk_length / self.target_sr

        for i, token in enumerate(tokens):
            if token != self.processor.tokenizer.pad_token:
                if not curr_text:
                    curr_start = (i / num_frames) * chunk_duration
                curr_text += token.replace("|", " ")
            
            if (token == "|" or i == len(tokens) - 1) and curr_text:
                curr_end = (i / num_frames) * chunk_duration
                segments.append({
                    'start': base_time + curr_start,
                    'end': base_time + curr_end,
                    'text': curr_text.strip()
                })
                curr_text = ""

        return segments