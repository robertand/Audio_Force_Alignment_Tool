import torch
import numpy as np
import librosa
import logging
from ctc_forced_aligner import (
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

logger = logging.getLogger(__name__)

class MMSDialogueAligner:
    def __init__(self, model_id="MahmoudAshraf/mms-300m-1130-forced-aligner", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        logger.info(f"Loading MMS aligner model {model_id} on {self.device}")
        
        self.alignment_model, self.alignment_tokenizer = load_alignment_model(
            self.device,
            model_path=model_id,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.target_sr = 16000

    def align_character_audio(self, audio_path, expected_segments, language="ro", **kwargs):
        """
        Align audio with expected segments using ctc-forced-aligner logic.
        """
        # Mapping common lang codes to ISO 639-3
        lang_map = {
            'ro': 'ron',
            'ron': 'ron',
            'en': 'eng',
            'eng': 'eng'
        }
        iso_language = lang_map.get(language, language)

        try:
            # 1. Load and process full audio using librosa
            audio, sr = librosa.load(audio_path, sr=self.target_sr)
            # Ensure 1D then move to torch
            audio_waveform = torch.from_numpy(audio).to(self.alignment_model.dtype).to(self.alignment_model.device)
            total_duration = len(audio) / self.target_sr

            # 2. Concatenate all text for global alignment
            texts = [s.get('text_ro', '').strip() for s in expected_segments]
            full_text = " ".join(texts)

            if not full_text:
                return [{'original': s, 'aligned': None} for s in expected_segments]

            # 3. Generate emissions
            emissions, stride = generate_emissions(
                self.alignment_model, audio_waveform, batch_size=16
            )

            # 4. Preprocess text
            # We use romanize=True as MMS models expect specific vocabulary
            tokens_starred, text_starred = preprocess_text(
                full_text, romanize=True, language=iso_language,
            )

            # Filter tokens to only include those in the tokenizer's vocabulary to avoid ValueError
            vocab = self.alignment_tokenizer.get_vocab()
            tokens_starred = [t if t in vocab else self.alignment_tokenizer.unk_token for t in tokens_starred]

            # 5. Get alignments
            segments, scores, blank_token = get_alignments(
                emissions, tokens_starred, self.alignment_tokenizer,
            )

            # 6. Get spans
            spans = get_spans(tokens_starred, segments, blank_token)

            # 7. Postprocess to get word timestamps
            word_timestamps = postprocess_results(text_starred, spans, stride, scores)

            # 8. Map word timestamps back to expected segments
            aligned_results = []
            word_idx = 0

            for expected in expected_segments:
                expected_text = expected.get('text_ro', '').strip()
                if not expected_text:
                    aligned_results.append({'original': expected, 'aligned': None})
                    continue

                # Count words in this segment
                expected_words = expected_text.split()
                num_words = len(expected_words)

                # Find corresponding words in global alignment
                segment_words = word_timestamps[word_idx : word_idx + num_words]

                if segment_words:
                    start_time = segment_words[0]['start']
                    end_time = segment_words[-1]['end']
                    
                    # Calculate average confidence for the segment
                    conf = np.mean([w.get('score', 0.0) for w in segment_words])
                    
                    aligned_results.append({
                        'original': expected,
                        'aligned': {
                            'start': float(start_time),
                            'end': float(end_time),
                            'text': " ".join([w['text'] for w in segment_words]),
                            'confidence': float(conf)
                        }
                    })
                    word_idx += num_words
                else:
                    aligned_results.append({'original': expected, 'aligned': None})

            return aligned_results

        except Exception as e:
            logger.error(f"MMS Alignment failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return empty alignments on error
            return [{'original': seg, 'aligned': None} for seg in expected_segments]
