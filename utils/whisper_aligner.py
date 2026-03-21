import whisper_timestamped as whisper
import numpy as np
import difflib
import torch
import logging

logger = logging.getLogger(__name__)

class DialogueAligner:
    def __init__(self, model_name="base", device=None):
        self.current_model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading Whisper {model_name} on {self.device}")
        self.model = whisper.load_model(model_name, device=self.device)

    def ensure_model(self, model_name, device=None):
        if model_name != self.current_model_name:
            logger.info(f"Switching to Whisper {model_name} on {device or self.device}")
            self.model = whisper.load_model(model_name, device=device or self.device)
            self.current_model_name = model_name

    def align_character_audio(self, audio_path, expected_segments, language="ro", 
                              initial_prompt=None, model_name=None, 
                              min_confidence=0.4, use_word_timestamps=True):
        """
        Transcribe audio and align with expected segments from metadata
        """
        if model_name:
            self.ensure_model(model_name)

        try:
            # Load audio
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Move to GPU if available
            if self.device == "cuda":
                audio = torch.from_numpy(audio).to(self.device)

            # Transcribe with word timestamps
            # Use specific decoding parameters to improve sensitivity, especially for Romanian
            result = self.model.transcribe(
                audio, 
                language=language, 
                word_timestamps=use_word_timestamps,
                initial_prompt=initial_prompt,
                beam_size=5 if self.device == "cuda" else 1,  # Better quality on GPU
                no_speech_threshold=0.3,
                logprob_threshold=-1.0,
                condition_on_previous_text=False
            )
            
            transcribed_segments = result.get('segments', [])

            # 2. Match transcriptions to expected segments (Strict Sequential Pass)
            aligned_results = []

            # Greedy Sequential Search with Word-Level Snapping
            # This ensures segments are found in order and don't jump backward
            trans_idx = 0
            for expected in expected_segments:
                expected_text = expected.get('text_ro', '').strip()
                if not expected_text:
                    aligned_results.append({'original': expected, 'aligned': None})
                    continue

                best_match = None
                highest_ratio = 0.0
                best_trans_idx = trans_idx

                # Search among remaining transcriptions
                # Look ahead up to 20 segments to find best match in sequence
                search_limit = min(trans_idx + 20, len(transcribed_segments))
                for i in range(trans_idx, search_limit):
                    transcribed = transcribed_segments[i]
                    ratio = difflib.SequenceMatcher(
                        None,
                        self._normalize_text(expected_text),
                        self._normalize_text(transcribed['text'])
                    ).ratio()

                    if ratio > highest_ratio:
                        highest_ratio = ratio
                        best_match = transcribed
                        best_trans_idx = i

                # If we found a decent match in sequence
                if best_match and highest_ratio > 0.4:
                    # Update global transcription index to maintain order
                    trans_idx = best_trans_idx + 1

                    # Word-level snapping: precisely define start/end by words
                    words = best_match.get('words', [])
                    if words:
                        start = words[0]['start']
                        end = words[-1]['end']
                    else:
                        start = best_match['start']
                        end = best_match['end']

                    aligned_results.append({
                        'original': expected,
                        'aligned': {
                            'start': start,
                            'end': end,
                            'text': best_match['text'],
                            'confidence': highest_ratio,
                            'words': words
                        }
                    })
                else:
                    # Not found in immediate sequence
                    aligned_results.append({'original': expected, 'aligned': None})

            # 3. Gap-Based Re-search & Verification
            # For any missing segments, search in the specific audio gap between neighbors
            for i in range(len(aligned_results)):
                if aligned_results[i]['aligned'] is not None:
                    continue

                gap_start = 0.0
                gap_end = audio.shape[0] / 16000.0

                if i > 0 and aligned_results[i-1]['aligned']:
                    gap_start = aligned_results[i-1]['aligned']['end']

                if i < len(aligned_results) - 1 and aligned_results[i+1]['aligned']:
                    gap_end = aligned_results[i+1]['aligned']['start']

                expected_text = aligned_results[i]['original'].get('text_ro', '').strip()
                best_gap_match = None
                highest_gap_ratio = 0.0

                # Search all transcriptions specifically within this gap
                for transcribed in transcribed_segments:
                    if transcribed['start'] >= gap_start - 0.5 and transcribed['end'] <= gap_end + 0.5:
                        ratio = difflib.SequenceMatcher(None, self._normalize_text(expected_text), self._normalize_text(transcribed['text'])).ratio()
                        if ratio > highest_gap_ratio:
                            highest_gap_ratio = ratio
                            best_gap_match = transcribed

                if best_gap_match and highest_gap_ratio >= 0.3:
                    words = best_gap_match.get('words', [])
                    aligned_results[i]['aligned'] = {
                        'start': words[0]['start'] if words else best_gap_match['start'],
                        'end': words[-1]['end'] if words else best_gap_match['end'],
                        'text': best_gap_match['text'],
                        'confidence': highest_gap_ratio,
                        'words': words
                    }
                else:
                    # Third Pass: Chronological Interpolation (Fallback)
                    # If we still haven't found it, place it linearly in the gap
                    # based on its position relative to other missing segments in this same gap
                    missing_in_gap = []
                    my_idx_in_missing = 0

                    # Count how many are missing between our gap boundaries
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

                    gap_dur = gap_end - gap_start
                    chunk_dur = gap_dur / max(1, gap_count)

                    est_start = gap_start + (my_pos - 1) * chunk_dur
                    est_end = est_start + chunk_dur

                    aligned_results[i]['aligned'] = {
                        'start': est_start,
                        'end': est_end,
                        'text': "[Estimated Position]",
                        'confidence': 0.1,
                        'words': []
                    }

            # 4. Third Pass: Word-Level Verification
            # Double check that the final aligned text actually matches the expected Romanian text
            for entry in aligned_results:
                aligned = entry.get('aligned')
                if not aligned:
                    continue

                expected_text = self._normalize_text(entry['original'].get('text_ro', ''))
                aligned_text = self._normalize_text(aligned.get('text', ''))

                # If similarity is very low, mark as low confidence
                ratio = difflib.SequenceMatcher(None, expected_text, aligned_text).ratio()
                if ratio < 0.3:
                    logger.warning(f"Low word match confidence for segment: {expected_text} vs {aligned_text}")
                    aligned['confidence'] = min(aligned['confidence'], ratio)

            return aligned_results

        except Exception as e:
            logger.error(f"Alignment error: {str(e)}")
            # Return empty alignments on error
            return [{'original': seg, 'aligned': None} for seg in expected_segments]
    
    def _normalize_text(self, text):
        """Normalize text for better matching"""
        import unicodedata
        import re
        
        # Lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Normalize diacritics
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        
        # Collapse spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text