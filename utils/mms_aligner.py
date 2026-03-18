import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
import librosa
import difflib

class MMSDialogueAligner:
    def __init__(self, model_id="MahmoudAshraf/mms-300m-1130-forced-aligner"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device)
        self.target_sr = 16000

    def align_character_audio(self, audio_path, expected_segments, language="ron"):
        """
        Align audio with expected segments using MMS model
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.target_sr)

        # Romanian ISO-639-3 code is 'ron' (which is the default for Romanian in MMS usually)
        # The processor and model are already loaded.
        # In a real implementation of forced alignment with Wav2Vec2CTC, we'd use
        # the trellis/CTCSegmentation algorithm.

        # 1. Get Logits
        inputs = self.processor(audio, sampling_rate=self.target_sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]

        # 2. Basic greedy decoding as a simplified "transcription" for matching
        # (For true forced alignment we'd need more complex logic, but we follow the user's
        # request to try this model)
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids.unsqueeze(0))[0]

        # We need timestamps. Simple linear mapping of logits to time as a first pass.
        # total_duration = len(audio) / self.target_sr
        # num_frames = logits.shape[0]
        # frame_duration = total_duration / num_frames

        # However, for forced alignment, we should ideally use CTC segmentation.
        # But to keep it consistent with our current architecture:
        # we'll return the greedy transcription with rough timestamps and use our sequential matcher.

        # Map predicted tokens to roughly their time in the file
        tokens = self.processor.tokenizer.convert_ids_to_tokens(predicted_ids.tolist())
        transcribed_segments = []

        # Group tokens into words/sentences with rough timestamps
        current_text = ""
        start_time = 0.0

        num_frames = logits.shape[0]
        total_duration = len(audio) / self.target_sr

        # Basic heuristic: non-blank tokens are speech
        for i, token in enumerate(tokens):
            if token != self.processor.tokenizer.pad_token:
                if not current_text:
                    start_time = (i / num_frames) * total_duration
                current_text += token.replace("|", " ")

            if (token == "|" or i == len(tokens) - 1) and current_text:
                end_time = (i / num_frames) * total_duration
                transcribed_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': current_text.strip()
                })
                current_text = ""

        # 3. Match using the same sequential logic as Whisper for consistency
        aligned_results = []
        last_match_idx = -1

        for expected in expected_segments:
            expected_text = expected.get('text_ro', '').strip()
            if not expected_text:
                continue

            best_match = None
            best_match_idx = -1
            highest_ratio = 0.0

            search_start = last_match_idx + 1
            search_end = min(len(transcribed_segments), search_start + 15)

            for i in range(search_start, search_end):
                transcribed = transcribed_segments[i]
                trans_text = transcribed['text'].strip()
                ratio = difflib.SequenceMatcher(None, expected_text.lower(), trans_text.lower()).ratio()

                if ratio > highest_ratio:
                    highest_ratio = ratio
                    best_match = transcribed
                    best_match_idx = i

            # Aggressive matching with fallback
            if highest_ratio < 0.3 and search_start < len(transcribed_segments):
                best_match = transcribed_segments[search_start]
                best_match_idx = search_start
                highest_ratio = 0.1

            if best_match:
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
            else:
                aligned_results.append({
                    'original': expected,
                    'aligned': None
                })

        return aligned_results
