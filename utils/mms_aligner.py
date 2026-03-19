import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
import librosa
import difflib
from utils.text_utils import normalize_text, strip_diacritics

class MMSDialogueAligner:
    def __init__(self, model_id="MahmoudAshraf/mms-300m-1130-forced-aligner"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device)
        self.target_sr = 16000

    def align_character_audio(self, audio_path, expected_segments, language="ron"):
        """
        Align audio with expected segments using MMS model with Segmented Local Alignment.
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.target_sr)
        total_audio_duration = len(audio) / self.target_sr

        aligned_results = []
        current_audio_ptr = 0.0 # Time in seconds in the input audio file

        # INCREASE SEARCH WINDOW to be more robust (from 40s to 120s)
        SEARCH_WINDOW_SEC = 120.0

        for expected in expected_segments:
            expected_text = expected.get('text_ro', '').strip()
            expected_duration = expected['end'] - expected['start']

            if not expected_text:
                start = min(current_audio_ptr, total_audio_duration)
                end = min(current_audio_ptr + expected_duration, total_audio_duration)
                aligned_results.append({
                    'original': expected,
                    'aligned': {'start': start, 'end': end, 'text': "", 'confidence': 1.0}
                })
                current_audio_ptr = end
                continue

            # 1. Search locally in the chunk
            search_start_time = max(0.0, current_audio_ptr - 5.0) # 5s overlap for safety
            search_end_time = min(total_audio_duration, search_start_time + SEARCH_WINDOW_SEC)

            start_sample = int(search_start_time * self.target_sr)
            end_sample = int(search_end_time * self.target_sr)
            chunk = audio[start_sample:end_sample]

            best_match = None
            highest_ratio = 0.0

            if len(chunk) >= 1000:
                inputs = self.processor(chunk, sampling_rate=self.target_sr, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    logits = self.model(**inputs).logits[0]

                predicted_ids = torch.argmax(logits, dim=-1)
                tokens = self.processor.tokenizer.convert_ids_to_tokens(predicted_ids.tolist())

                local_segments = []
                curr_text = ""
                curr_start = 0.0
                num_frames = logits.shape[0]
                chunk_duration = len(chunk) / self.target_sr

                for i, token in enumerate(tokens):
                    if token != self.processor.tokenizer.pad_token:
                        if not curr_text:
                            curr_start = (i / num_frames) * chunk_duration
                        curr_text += token.replace("|", " ")
                    if (token == "|" or i == len(tokens) - 1) and curr_text:
                        curr_end = (i / num_frames) * chunk_duration
                        local_segments.append({
                            'start': search_start_time + curr_start,
                            'end': search_start_time + curr_end,
                            'text': curr_text.strip()
                        })
                        curr_text = ""

                norm_expected = normalize_text(expected_text)
                strip_expected = strip_diacritics(norm_expected)

                for seg in local_segments:
                    norm_trans = normalize_text(seg['text'])
                    ratio = difflib.SequenceMatcher(None, norm_expected, norm_trans).ratio()
                    strip_ratio = difflib.SequenceMatcher(None, strip_expected, strip_diacritics(norm_trans)).ratio()
                    combined_ratio = max(ratio, strip_ratio * 0.9)

                    if combined_ratio > highest_ratio:
                        highest_ratio = combined_ratio
                        best_match = seg

            # 2. Results and fallback logic
            if highest_ratio >= 0.35 and best_match:
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
                # Fallback: Just COPY EXACT DURATION FROM THE CURRENT POSITION
                # (User request: "copiaza pistele asa cum sunt" when failing)
                start = min(current_audio_ptr, total_audio_duration)
                end = min(current_audio_ptr + expected_duration, total_audio_duration)
                aligned_results.append({
                    'original': expected,
                    'aligned': {'start': start, 'end': end, 'text': "[Direct Copy Fallback]", 'confidence': 0.0}
                })
                current_audio_ptr = end

        return aligned_results
