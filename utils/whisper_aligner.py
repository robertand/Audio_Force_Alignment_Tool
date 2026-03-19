import whisper_timestamped as whisper
import numpy as np
import difflib
import whisper as openai_whisper

class DialogueAligner:
    def __init__(self, model_name="base"):
        self.current_model_name = model_name
        self.model = openai_whisper.load_model(model_name)

    def ensure_model(self, model_name):
        if model_name != self.current_model_name:
            self.model = openai_whisper.load_model(model_name)
            self.current_model_name = model_name

    def align_character_audio(self, audio_path, expected_segments, language="ro", initial_prompt=None, model_name=None):
        """
        Transcribe audio and align with expected segments from metadata
        """
        if model_name:
            self.ensure_model(model_name)

        # To avoid ffmpeg in whisper.load_audio, we can load it ourselves
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)

        result = self.model.transcribe(audio, language=language, word_timestamps=True, initial_prompt=initial_prompt)
        transcribed_segments = result['segments']

        # 2. Match transcriptions to expected segments
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
