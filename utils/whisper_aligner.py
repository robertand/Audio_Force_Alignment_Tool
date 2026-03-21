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

            # 2. Match transcriptions to expected segments (Global Pass)
            # This allows finding the best match anywhere to expose outliers
            aligned_results = []

            for expected in expected_segments:
                expected_text = expected.get('text_ro', '').strip()
                if not expected_text:
                    continue

                best_match = None
                highest_ratio = 0.0
                best_words = []

                for transcribed in transcribed_segments:
                    trans_text = transcribed['text'].strip()
                    
                    # Calculate similarity
                    ratio = difflib.SequenceMatcher(
                        None, 
                        expected_text.lower(), 
                        trans_text.lower()
                    ).ratio()
                    
                    # Also check with normalized text
                    norm_expected = self._normalize_text(expected_text)
                    norm_trans = self._normalize_text(trans_text)
                    norm_ratio = difflib.SequenceMatcher(
                        None, norm_expected, norm_trans
                    ).ratio()
                    
                    combined_ratio = max(ratio, norm_ratio)

                    if combined_ratio > highest_ratio:
                        highest_ratio = combined_ratio
                        best_match = transcribed
                        best_words = transcribed.get('words', [])

                # Only accept if confidence is good enough
                if best_match and highest_ratio >= min_confidence:
                    # Use word-level timestamps if available
                    if best_words and len(best_words) > 0:
                        aligned_start = best_words[0]['start']
                        aligned_end = best_words[-1]['end']
                    else:
                        aligned_start = best_match['start']
                        aligned_end = best_match['end']

                    aligned_results.append({
                        'original': expected,
                        'aligned': {
                            'start': aligned_start,
                            'end': aligned_end,
                            'text': best_match['text'],
                            'confidence': highest_ratio,
                            'words': best_words
                        }
                    })
                else:
                    # If no match found, create empty alignment
                    aligned_results.append({
                        'original': expected,
                        'aligned': None
                    })

            # 3. Outlier Detection and Localized Re-search
            # If segment N was found far after segment N-1 but also far before N+1,
            # but N itself has a timestamp that's way out of order, re-search.
            for i in range(len(aligned_results)):
                # Determine local search window
                gap_start = 0.0
                gap_end = 9999.0 # Large default

                # Check neighbors to define the valid chronological window
                if i > 0:
                    prev = aligned_results[i-1].get('aligned')
                    if prev: gap_start = prev['end']

                if i < len(aligned_results) - 1:
                    nxt = aligned_results[i+1].get('aligned')
                    if nxt: gap_end = nxt['start']

                curr = aligned_results[i].get('aligned')
                is_outlier = False

                if curr:
                    # Out of order if not within neighbors' bounds
                    if curr['start'] < gap_start - 0.5 or curr['end'] > gap_end + 0.5:
                        is_outlier = True
                else:
                    # If no match found at all, it's a candidate for gap search
                    is_outlier = True

                if is_outlier:
                    logger.info(f"Segment {i} is outlier or missing (Window: {gap_start} - {gap_end}). Re-aligning...")

                    expected_text = aligned_results[i]['original'].get('text_ro', '').strip()

                    best_local_match = None
                    highest_local_ratio = 0.0

                    for transcribed in transcribed_segments:
                        # Only look in the gap
                        if transcribed['start'] >= gap_start - 0.5 and transcribed['end'] <= gap_end + 0.5:
                            ratio = difflib.SequenceMatcher(None, expected_text.lower(), transcribed['text'].strip().lower()).ratio()
                            if ratio > highest_local_ratio:
                                highest_local_ratio = ratio
                                best_local_match = transcribed

                    if best_local_match and highest_local_ratio >= 0.3: # Relaxed threshold for local gap search
                        logger.info(f"Found better local match for outlier {i} at {best_local_match['start']}")
                        aligned_results[i]['aligned'] = {
                            'start': best_local_match['start'],
                            'end': best_local_match['end'],
                            'text': best_local_match['text'],
                            'confidence': highest_local_ratio,
                            'words': best_local_match.get('words', [])
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