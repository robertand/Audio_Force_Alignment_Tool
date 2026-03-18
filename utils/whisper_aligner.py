import numpy as np
import difflib
import whisper as openai_whisper
import torch

class DialogueAligner:
    def __init__(self, model_name="base"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load the native openai-whisper model
        self.model = openai_whisper.load_model(model_name, device=self.device)

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

        try:
            result = self.model.transcribe(
                audio,
                language=language,
                word_timestamps=True,
                no_speech_threshold=0.3,
                logprob_threshold=-1.0,
                condition_on_previous_text=False,
                fp16=(self.device == "cuda")
            )
        except (RuntimeError, ValueError, TypeError) as e:
            # Catching more than just RuntimeError as Whisper/Torch errors can sometimes
            # manifest as other types depending on where the dimension mismatch is caught
            err_msg = str(e).lower()
            if "tensor" in err_msg or "size" in err_msg or "dimension" in err_msg or "match" in err_msg:
                print(f"Whisper word_timestamps failed ({type(e).__name__}), retrying without it: {e}")
                # Fallback: transcribe without word timestamps to avoid tensor dimension mismatch errors
                # This uses segment-level timestamps which are more stable
                result = self.model.transcribe(
                    audio,
                    language=language,
                    word_timestamps=False,
                    no_speech_threshold=0.3,
                    logprob_threshold=-1.0,
                    condition_on_previous_text=False,
                    fp16=(self.device == "cuda")
                )
            else:
                raise e
        transcribed_segments = result['segments']

        # 2. Match transcriptions to expected segments (Sequential/Aggressive)
        aligned_results = []
        last_match_idx = -1

        for expected in expected_segments:
            expected_text = expected.get('text_ro', '').strip()
            if not expected_text:
                continue

            best_match = None
            best_match_idx = -1
            highest_ratio = 0.0

            # 1. Try fuzzy matching in a window
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

            # 2. Sequential Fallback: If no good text match, take the next chronological segment
            # as long as it exists and isn't too far from the last one
            if highest_ratio < 0.35 and search_start < len(transcribed_segments):
                best_match = transcribed_segments[search_start]
                best_match_idx = search_start
                highest_ratio = 0.2  # Placeholder for fallback confidence

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
                # Fallback to metadata timing if absolutely nothing in audio
                aligned_results.append({
                    'original': expected,
                    'aligned': None
                })

        return aligned_results
