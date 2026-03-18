import whisper_timestamped as whisper
import numpy as np
import difflib

class DialogueAligner:
    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)

    def align_character_audio(self, audio_path, expected_segments, language="ro"):
        """
        Transcribe audio and align with expected segments from metadata
        """
        # 1. Transcribe with timestamps
        result = whisper.transcribe(self.model, audio_path, language=language)
        transcribed_segments = result['segments']

        # 2. Match transcriptions to expected segments
        # We'll use a simple matching logic based on text similarity
        aligned_results = []

        for expected in expected_segments:
            expected_text = expected.get('text_ro', '').strip()
            if not expected_text:
                continue

            best_match = None
            highest_ratio = 0.0

            for transcribed in transcribed_segments:
                trans_text = transcribed['text'].strip()
                ratio = difflib.SequenceMatcher(None, expected_text.lower(), trans_text.lower()).ratio()

                if ratio > highest_ratio:
                    highest_ratio = ratio
                    best_match = transcribed

            # Only accept if ratio is good enough
            if best_match and highest_ratio > 0.4:
                aligned_results.append({
                    'original': expected,
                    'aligned': {
                        'start': best_match['start'],
                        'end': best_match['end'],
                        'text': best_match['text'],
                        'confidence': highest_ratio
                    }
                })
            else:
                # If no match found, keep the original timecode as fallback
                aligned_results.append({
                    'original': expected,
                    'aligned': None
                })

        return aligned_results
