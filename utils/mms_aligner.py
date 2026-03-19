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
        Align audio with expected segments using MMS model.
        When Aligner is disabled (or as fallback), it returns segments based exactly on original durations.
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.target_sr)
        total_audio_duration = len(audio) / self.target_sr

        # 1. Get Logits
        inputs = self.processor(audio, sampling_rate=self.target_sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]

        # 2. Map predicted tokens to roughly their time in the file
        predicted_ids = torch.argmax(logits, dim=-1)
        tokens = self.processor.tokenizer.convert_ids_to_tokens(predicted_ids.tolist())
        transcribed_segments = []

        current_text = ""
        start_time = 0.0
        num_frames = logits.shape[0]

        for i, token in enumerate(tokens):
            if token != self.processor.tokenizer.pad_token:
                if not current_text:
                    start_time = (i / num_frames) * total_audio_duration
                current_text += token.replace("|", " ")

            if (token == "|" or i == len(tokens) - 1) and current_text:
                end_time = (i / num_frames) * total_audio_duration
                transcribed_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': current_text.strip()
                })
                current_text = ""

        # 3. Match using sequential logic
        aligned_results = []
        last_match_idx = -1

        # Track where we are in the input audio for duration-based fallback
        current_audio_pointer = 0.0

        for expected in expected_segments:
            expected_text = expected.get('text_ro', '').strip()
            # Duration based on metadata
            expected_duration = expected['end'] - expected['start']

            if not expected_text:
                # Still try to map if there's no text (e.g. music/sound)
                aligned_results.append({
                    'original': expected,
                    'aligned': {
                        'start': min(current_audio_pointer, total_audio_duration),
                        'end': min(current_audio_pointer + expected_duration, total_audio_duration),
                        'text': "",
                        'confidence': 1.0
                    }
                })
                current_audio_pointer += expected_duration
                continue

            best_match = None
            best_match_idx = -1
            highest_ratio = 0.0

            search_start = last_match_idx + 1
            # Search window
            search_end = min(len(transcribed_segments), search_start + 15)

            norm_expected = normalize_text(expected_text)
            strip_expected = strip_diacritics(norm_expected)

            for i in range(search_start, search_end):
                transcribed = transcribed_segments[i]
                trans_text = transcribed['text'].strip()
                norm_trans = normalize_text(trans_text)

                ratio = difflib.SequenceMatcher(None, norm_expected, norm_trans).ratio()
                strip_ratio = difflib.SequenceMatcher(None, strip_expected, strip_diacritics(norm_trans)).ratio()

                combined_ratio = max(ratio, strip_ratio * 0.9)

                if combined_ratio > highest_ratio:
                    highest_ratio = combined_ratio
                    best_match = transcribed
                    best_match_idx = i

            if highest_ratio >= 0.3 and best_match:
                aligned_results.append({
                    'original': expected,
                    'aligned': {
                        'start': best_match['start'],
                        'end': best_match['end'],
                        'text': best_match['text'],
                        'confidence': highest_ratio
                    }
                })
                last_match_idx = best_match_idx
                current_audio_pointer = best_match['end']
            else:
                # Fallback: take next chunk from input audio based on metadata duration
                # This ensures tracks are "copied as they are" if aligner fails
                start = min(current_audio_pointer, total_audio_duration)
                end = min(current_audio_pointer + expected_duration, total_audio_duration)

                aligned_results.append({
                    'original': expected,
                    'aligned': {
                        'start': start,
                        'end': end,
                        'text': "[Fallback Duration]",
                        'confidence': 0.0
                    }
                })
                current_audio_pointer = end

        return aligned_results
