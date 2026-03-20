import ctc_forced_aligner
import re
import logging
import os
import numpy as np
import librosa

logger = logging.getLogger(__name__)

class CTCAligner:
    def __init__(self, device="cpu"):
        self.device = device

    def preprocess_text_clean(self, text):
        # Remove Eleven Labs tags like [tag]
        clean_text = re.sub(r'\[.*?\]', '', text)
        return clean_text.strip()

    def align(self, audio_path, text_list, language="ro"):
        # The deskpai version of ctc_forced_aligner uses onnx

        # Preprocess each text in the list
        cleaned_texts = [self.preprocess_text_clean(t) for t in text_list]
        full_transcript = " ".join(cleaned_texts)

        try:
            # Use the aligner API from deskpai/ctc-forced-aligner
            audio, sr = librosa.load(audio_path, sr=16000)

            # This is based on typical usage for this library version
            # results = ctc_forced_aligner.align(audio, full_transcript, self.device)
            results = ctc_forced_aligner.align(audio, full_transcript)

            # Match results back to text_list segments
            # For simplicity in this tool, if it returns a list of objects with start/end
            formatted_results = []
            for res in results:
                formatted_results.append({
                    'start': float(res.start) / 1000.0 if res.start > 1000 else float(res.start), # Handle ms vs s
                    'end': float(res.end) / 1000.0 if res.end > 1000 else float(res.end),
                    'text': res.text
                })
            return formatted_results

        except Exception as e:
            logger.error(f"CTC Alignment failed: {e}")
            return None
