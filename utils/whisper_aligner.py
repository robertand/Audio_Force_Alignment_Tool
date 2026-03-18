import whisper_timestamped as whisper
import numpy as np
import difflib
import whisper as openai_whisper

class DialogueAligner:
    def __init__(self, model_name="base"):
        # Load the native openai-whisper model
        self.model = openai_whisper.load_model(model_name)

    def align_character_audio(self, audio_path, expected_segments, language="ro"):
        """
        Transcribe audio and align with expected segments from metadata
        """
        # 1. Transcribe with timestamps using native whisper
        # This avoids the ffmpeg requirement for some operations and the hook issues
        # although whisper itself might still want ffmpeg for loading if it's not a numpy array.

        # To avoid ffmpeg in whisper.load_audio, we can load it ourselves
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)

        result = self.model.transcribe(
            audio,
            language=language,
            word_timestamps=True,
            no_speech_threshold=0.5,
            logprob_threshold=-0.8
        )
        transcribed_segments = result['segments']

        # 2. Match transcriptions to expected segments (Sequential/Greedy)
        aligned_results = []
        last_match_idx = -1

        for expected in expected_segments:
            expected_text = expected.get('text_ro', '').strip()
            if not expected_text:
                continue

            best_match = None
            best_match_idx = -1
            highest_ratio = 0.0

            # Search in a window after the last match
            # Search up to 10 segments ahead to find the best match for current line
            search_start = last_match_idx + 1
            search_end = min(len(transcribed_segments), search_start + 10)

            for i in range(search_start, search_end):
                transcribed = transcribed_segments[i]
                trans_text = transcribed['text'].strip()
                ratio = difflib.SequenceMatcher(None, expected_text.lower(), trans_text.lower()).ratio()

                if ratio > highest_ratio:
                    highest_ratio = ratio
                    best_match = transcribed
                    best_match_idx = i

            # Only accept if ratio is good enough
            if best_match and highest_ratio > 0.35:
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
                # If no match found, keep the original timecode as fallback
                # but don't advance last_match_idx so we can keep searching
                aligned_results.append({
                    'original': expected,
                    'aligned': None
                })

        return aligned_results
