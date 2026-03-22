import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
import librosa
import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

@dataclass
class Word:
    text: str
    start: float
    end: float
    score: float

class MMSDialogueAligner:
    def __init__(self, model_id="MahmoudAshraf/mms-300m-1130-forced-aligner", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading MMS aligner on {self.device}")
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)

        if self.device == "cuda":
            self.model = self.model.to(self.device)
            self.model = self.model.half() if torch.cuda.get_device_capability()[0] >= 7 else self.model

        self.target_sr = 16000

    def align_character_audio(self, audio_path, expected_segments, language="ro", **kwargs):
        """
        Align audio with expected segments using pure transformers/torch CTC alignment.
        """
        from utils.text_utils import clean_text_for_alignment

        # Mapping common lang codes
        lang_map = {'ro': 'ron', 'ron': 'ron', 'en': 'eng', 'eng': 'eng'}
        language = lang_map.get(language, language)

        try:
            # 1. Load audio
            audio, sr = librosa.load(audio_path, sr=self.target_sr)
            if len(audio) == 0:
                return [{'original': s, 'aligned': None} for s in expected_segments]

            # 2. Preprocess text
            texts = [clean_text_for_alignment(s.get('text_ro', '')) for s in expected_segments]
            full_text = " ".join(texts)
            if not full_text:
                return [{'original': s, 'aligned': None} for s in expected_segments]

            # 3. Get Emissions
            inputs = self.processor(audio, sampling_rate=self.target_sr, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                if self.device == "cuda":
                    with torch.amp.autocast('cuda'):
                        logits = self.model(**inputs).logits[0]
                else:
                    logits = self.model(**inputs).logits[0]

            log_probs = torch.log_softmax(logits, dim=-1).cpu()

            # 4. CTC Alignment Logic (Trellis)
            clean_text = full_text.lower().replace(" ", "|")

            # Filter tokens to vocab
            vocab = self.processor.tokenizer.get_vocab()
            tokens = []
            for c in clean_text:
                if c in vocab:
                    tokens.append(vocab[c])
                else:
                    tokens.append(self.processor.tokenizer.unk_token_id)

            # Forced alignment (Simplified Trellis)
            trellis = self._get_trellis(log_probs, tokens)
            path = self._backtrack(trellis, log_probs, tokens)

            if not path:
                logger.warning("Alignment path not found")
                return [{'original': s, 'aligned': None} for s in expected_segments]

            # 5. Extract Word Timestamps
            stride_s = len(audio) / self.target_sr / log_probs.shape[0]
            word_timestamps = self._get_word_timestamps(path, clean_text, stride_s)

            # 6. Map back to segments
            aligned_results = []
            word_idx = 0

            for expected in expected_segments:
                expected_text = clean_text_for_alignment(expected.get('text_ro', ''))
                if not expected_text:
                    aligned_results.append({'original': expected, 'aligned': None})
                    continue

                num_words = len(expected_text.split())
                segment_words = word_timestamps[word_idx : word_idx + num_words]

                if segment_words:
                    aligned_results.append({
                        'original': expected,
                        'aligned': {
                            'start': float(segment_words[0].start),
                            'end': float(segment_words[-1].end),
                            'text': " ".join([w.text for w in segment_words]),
                            'confidence': float(np.mean([w.score for w in segment_words]))
                        }
                    })
                    word_idx += num_words
                else:
                    aligned_results.append({'original': expected, 'aligned': None})

            return aligned_results

        except Exception as e:
            logger.error(f"Alignment failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [{'original': s, 'aligned': None} for s in expected_segments]

    def _get_trellis(self, log_probs, tokens, blank_id=0):
        num_frames = log_probs.shape[0]
        num_tokens = len(tokens)

        # Ensure log_probs contains the tokens we are looking for
        # Vocab size might be smaller than some token IDs if something went wrong
        max_token_id = log_probs.shape[1] - 1
        safe_tokens = [min(t, max_token_id) for t in tokens]

        trellis = torch.full((num_frames + 1, num_tokens + 1), -float("inf"))
        trellis[0, 0] = 0

        for j in range(1, num_tokens + 1):
            trellis[0, j] = -float("inf")

        for i in range(1, num_frames + 1):
            for j in range(1, num_tokens + 1):
                # Stay at same token (consuming a frame) OR Move to next token
                token_id = safe_tokens[j - 1]

                stay = trellis[i - 1, j] + log_probs[i - 1, blank_id]
                move = trellis[i - 1, j - 1] + log_probs[i - 1, token_id]
                trellis[i, j] = torch.logaddexp(stay, move)

        return trellis

    def _backtrack(self, trellis, log_probs, tokens, blank_id=0):
        i, j = trellis.shape[0] - 1, trellis.shape[1] - 1
        path = []

        max_token_id = log_probs.shape[1] - 1
        safe_tokens = [min(t, max_token_id) for t in tokens]

        while j > 0:
            assert i > 0
            token_id = safe_tokens[j - 1]
            stay = trellis[i - 1, j] + log_probs[i - 1, blank_id]
            move = trellis[i - 1, j - 1] + log_probs[i - 1, token_id]

            if move > stay:
                path.append(Point(j - 1, i - 1, float(move.exp())))
                j -= 1
                i -= 1
            else:
                i -= 1

        return path[::-1]

    def _get_word_timestamps(self, path: List[Point], text: str, stride_s: float):
        words = []
        current_word = ""
        word_start = -1
        word_scores = []

        for p in path:
            char = text[p.token_index]

            if char == "|":
                if current_word:
                    words.append(Word(
                        current_word,
                        word_start * stride_s,
                        p.time_index * stride_s,
                        float(np.mean(word_scores)) if word_scores else 0.0
                    ))
                    current_word = ""
                    word_scores = []
            else:
                if not current_word:
                    word_start = p.time_index
                current_word += char
                word_scores.append(p.score)

        if current_word:
            words.append(Word(
                current_word,
                word_start * stride_s,
                path[-1].time_index * stride_s,
                float(np.mean(word_scores)) if word_scores else 0.0
            ))

        return words
