import torch
import numpy as np
import difflib
import logging
import whisperx
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class WhisperXAligner:
    def __init__(self, model_name="large-v3", device=None, compute_type=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type or ("float16" if self.device == "cuda" else "int8")
        self.current_model_name = model_name

        logger.info(f"Loading WhisperX model {model_name} on {self.device} ({self.compute_type})")
        try:
            self.model = whisperx.load_model(model_name, device=self.device, compute_type=self.compute_type)
        except Exception as e:
            logger.error(f"Failed to load WhisperX model {model_name}: {e}")
            if "model.bin" in str(e):
                logger.warning("This model might not be in CTranslate2 format. Falling back to 'large-v3'")
                self.model = whisperx.load_model("large-v3", device=self.device, compute_type=self.compute_type)
                self.current_model_name = "large-v3"
            else:
                raise e

        # Cache for alignment models
        self.align_models = {}

    def ensure_model(self, model_name):
        if model_name != self.current_model_name:
            logger.info(f"Switching WhisperX model to {model_name}")
            # Free memory
            if hasattr(self, 'model'):
                del self.model
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            try:
                self.model = whisperx.load_model(model_name, device=self.device, compute_type=self.compute_type)
                self.current_model_name = model_name
            except Exception as e:
                logger.error(f"Failed to switch WhisperX model to {model_name}: {e}")
                if "model.bin" in str(e):
                    logger.warning("Falling back to 'large-v3' due to CT2 incompatibility")
                    self.model = whisperx.load_model("large-v3", device=self.device, compute_type=self.compute_type)
                    self.current_model_name = "large-v3"
                else:
                    raise e

    def align_character_audio(self, audio_path: str, expected_segments: List[Dict],
                              language="ro", initial_prompt=None, offset_time=0.0, **kwargs):
        """
        Transcribe and align using WhisperX for ultra-exact timestamps.
        """
        from utils.text_utils import clean_text_for_alignment

        # Mapping common lang codes
        lang_map = {'ro': 'ro', 'ron': 'ro', 'en': 'en', 'eng': 'en'}
        language = lang_map.get(language, language)

        try:
            # 1. Load audio
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)

            if offset_time > 0:
                # Ensure we don't slice past the end
                start_sample = min(int(offset_time * sr), len(audio) - 1)
                audio = audio[start_sample:]

            # 2. Initial Transcription
            logger.info(f"Transcribing {audio_path} with WhisperX (Initial Prompt: {initial_prompt})...")

            # WhisperX uses faster-whisper under the hood.
            # Note: initial_prompt was causing TypeError in some versions,
            # we will omit it for now as WhisperX's transcribe method
            # doesn't explicitly expose it in all versions/wrappers.

            result = self.model.transcribe(
                audio,
                batch_size=kwargs.get('batch_size', 16),
                language=language
            )

            # 3. Load Alignment Model (Romanian specific)
            align_model_name = "infinitejoy/wav2vec2-large-xls-r-300m-romanian" if language == "ro" else None

            if language not in self.align_models:
                logger.info(f"Loading alignment model for {language}: {align_model_name}")
                self.align_models[language] = whisperx.load_align_model(
                    language_code=language,
                    device=self.device,
                    model_name=align_model_name
                )

            align_model, metadata = self.align_models[language]

            # 4. Perform Forced Alignment
            logger.info("Performing WhisperX forced alignment...")
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                self.device,
                return_char_alignments=False
            )

            transcribed_segments = result.get('segments', [])

            # 5. Match with expected segments
            aligned_results = []

            for expected in expected_segments:
                expected_text = clean_text_for_alignment(expected.get('text_ro', ''))
                if not expected_text:
                    aligned_results.append({'original': expected, 'aligned': None})
                    continue

                best_match = None
                highest_ratio = 0.0

                for transcribed in transcribed_segments:
                    # WhisperX segments already have word-level timestamps inside if needed
                    # But for the block level, we match the text. Use global coordinates for matching.
                    trans_start = float(transcribed['start']) + offset_time
                    trans_end = float(transcribed['end']) + offset_time

                    norm_trans = clean_text_for_alignment(transcribed['text'])
                    ratio = difflib.SequenceMatcher(None, expected_text, norm_trans).ratio()

                    if ratio > highest_ratio:
                        highest_ratio = ratio
                        best_match = transcribed

                if best_match and highest_ratio > 0.4:
                    aligned_results.append({
                        'original': expected,
                        'aligned': {
                            'start': float(best_match['start']) + offset_time,
                            'end': float(best_match['end']) + offset_time,
                            'text': best_match['text'],
                            'confidence': float(highest_ratio),
                            'words': best_match.get('words', [])
                        }
                    })
                else:
                    aligned_results.append({'original': expected, 'aligned': None})

            # Handle sequential gaps if necessary (same logic as whisper_aligner)
            full_audio_duration = (len(audio) / 16000.0) + offset_time
            self._fill_gaps(aligned_results, full_audio_duration, offset_time=offset_time)

            return aligned_results

        except Exception as e:
            logger.error(f"WhisperX alignment error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [{'original': s, 'aligned': None} for s in expected_segments]

    def _fill_gaps(self, aligned_results, total_duration, offset_time=0.0):
        """Sequential interpolation for missing segments"""
        for i in range(len(aligned_results)):
            if aligned_results[i]['aligned'] is not None:
                continue

            gap_start = offset_time
            gap_end = total_duration

            for j in range(i - 1, -1, -1):
                if aligned_results[j]['aligned']:
                    gap_start = aligned_results[j]['aligned']['end']
                    break

            for j in range(i + 1, len(aligned_results)):
                if aligned_results[j]['aligned']:
                    gap_end = aligned_results[j]['aligned']['start']
                    break

            low_bound_idx = -1
            for j in range(i - 1, -1, -1):
                if aligned_results[j]['aligned']:
                    low_bound_idx = j
                    break

            high_bound_idx = len(aligned_results)
            for j in range(i + 1, len(aligned_results)):
                if aligned_results[j]['aligned']:
                    high_bound_idx = j
                    break

            gap_count = high_bound_idx - low_bound_idx - 1
            my_pos = i - low_bound_idx
            gap_dur = max(0.1, gap_end - gap_start)
            chunk_dur = gap_dur / max(1, gap_count)

            est_start = gap_start + (my_pos - 1) * chunk_dur
            est_end = est_start + chunk_dur

            aligned_results[i]['aligned'] = {
                'start': est_start,
                'end': est_end,
                'text': "[Estimated Position]",
                'confidence': 0.0
            }
