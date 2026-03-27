from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import os
import json
import pandas as pd
import numpy as np
import tempfile
import shutil
import threading
import uuid
import time
import zipfile
from datetime import datetime
from werkzeug.utils import secure_filename
import torch
import torchaudio
import soundfile as sf
from utils.audio_processor import AudioProcessor
from utils.vad_processor import VADProcessor
from utils.alignment_utils import AlignmentUtils
import traceback
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TUTORIAL_FOLDER'] = 'uploads/tutorials'
app.config['DOCUMENT_FOLDER'] = 'uploads/documents'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max file size
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_FILES_PER_JOB'] = 20
app.config['CLEANUP_INTERVAL'] = 600  # 10 minutes
app.config['JOB_TIMEOUT'] = 3600  # 1 hour

# Create folders if they don't exist
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
Path(app.config['TUTORIAL_FOLDER']).mkdir(parents=True, exist_ok=True)
Path(app.config['DOCUMENT_FOLDER']).mkdir(parents=True, exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(parents=True, exist_ok=True)

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    logger.info(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    logger.info(f"   CUDA Version: {torch.version.cuda}")
    logger.info(f"   Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    logger.warning("⚠️ GPU not detected. Using CPU (will be slower)")

# Initialize processors
audio_processor = AudioProcessor()
vad_processor = VADProcessor()
alignment_utils = AlignmentUtils()

# Job management with metadata
JOBS = {}
jobs_lock = threading.Lock()

# Lazy-loaded aligners with GPU support
_whisper_aligner = None
_whisperx_aligner = None
aligner_lock = threading.Lock()

def get_whisper_aligner(model_name="base"):
    global _whisper_aligner
    with aligner_lock:
        if _whisper_aligner is None:
            try:
                from utils.whisper_aligner import DialogueAligner
                logger.info(f"💾 Loading initial Whisper model: {model_name}")
                _whisper_aligner = DialogueAligner(model_name=model_name, device=device)
            except Exception as e:
                logger.error(f"Failed to load Whisper aligner: {e}")
                return None
        elif _whisper_aligner.current_model_name != model_name:
            logger.info(f"💾 Switching cached Whisper model from {_whisper_aligner.current_model_name} to {model_name}")
            _whisper_aligner.ensure_model(model_name)
        else:
            logger.info(f"💾 Reusing cached Whisper model: {model_name}")
        return _whisper_aligner

def get_whisperx_aligner(model_name="TransferRapid/whisper-large-v3-turbo_ro"):
    global _whisperx_aligner
    with aligner_lock:
        if _whisperx_aligner is None:
            try:
                from utils.whisperx_aligner import WhisperXAligner
                logger.info(f"💾 Loading initial WhisperX model: {model_name}")
                _whisperx_aligner = WhisperXAligner(model_name=model_name, device=device)
            except Exception as e:
                logger.error(f"Failed to load WhisperX aligner: {e}")
                return None
        else:
            if _whisperx_aligner.current_model_name != model_name:
                logger.info(f"💾 Switching cached WhisperX model from {_whisperx_aligner.current_model_name} to {model_name}")
                _whisperx_aligner.ensure_model(model_name)
            else:
                logger.info(f"💾 Reusing cached WhisperX model: {model_name}")
        return _whisperx_aligner

def cleanup_old_jobs():
    """Periodically remove jobs older than JOB_TIMEOUT"""
    while True:
        try:
            time.sleep(app.config['CLEANUP_INTERVAL'])
            now = time.time()
            to_delete = []
            
            with jobs_lock:
                for job_id, job in list(JOBS.items()):
                    if now - job.get('created_at', 0) > app.config['JOB_TIMEOUT']:
                        to_delete.append(job_id)

            for job_id in to_delete:
                with jobs_lock:
                    job = JOBS.pop(job_id, None)
                    if job:
                        # Clean up uploaded files
                        for file_info in job.get('files', []):
                            path = file_info.get('path')
                            if path and os.path.exists(path):
                                try:
                                    os.remove(path)
                                    logger.info(f"Deleted file: {path}")
                                except Exception as e:
                                    logger.warning(f"Failed to delete {path}: {e}")
                        
                        # Clean up output files
                        output_files = job.get('output_files', [])
                        for filename in output_files:
                            path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
                            if os.path.exists(path):
                                try:
                                    os.remove(path)
                                    logger.info(f"Deleted output: {path}")
                                except Exception as e:
                                    logger.warning(f"Failed to delete {path}: {e}")
                
                logger.info(f"Cleaned up {len(to_delete)} expired jobs")
                
        except Exception as e:
            logger.error(f"Cleanup thread error: {e}")
            time.sleep(60)  # Wait a minute before retrying

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_jobs, daemon=True)
cleanup_thread.start()
logger.info("Cleanup thread started")

def apply_alignment_post_processing(alignments, onsets, preceding_end=None):
    """
    Apply common refinement rules to alignments:
    1. Silence Crop (Hump detection)
    2. Hump-to-Word Rule (Trim if too many humps)
    3. End Padding (+0.5s)
    4. NO OVERLAP Rule
    """
    from utils.text_utils import clean_text_for_alignment

    # 1. & 2. Silence Crop and Hump-to-Word
    for idx_p, entry in enumerate(alignments):
        aligned = entry.get('aligned')
        if not aligned:
            continue

        # Find first onset in or near segment for start refinement
        current_start = aligned['start']
        first_local_onset = None
        for onset in onsets:
            if abs(onset - current_start) < 1.0: # 1s window
                first_local_onset = onset
                break

        if first_local_onset is not None:
            aligned['start'] = max(0, first_local_onset - 0.1)

        # Hump-to-Word Rule
        expected_text = clean_text_for_alignment(entry['original'].get('text_ro', ''))
        num_words = len(expected_text.split())

        if num_words > 0:
            # Get onsets in current segment
            seg_onsets = [o for o in onsets if aligned['start'] <= o <= aligned['end']]

            if len(seg_onsets) > num_words:
                # Trim to shortly after last valid word's hump
                aligned['end'] = seg_onsets[num_words-1] + 0.2

        # 3. Add 0.5s Default Padding to End
        aligned['end'] += 0.5

    # 4. Apply NO OVERLAP Rule
    for i in range(len(alignments)):
        curr = alignments[i].get('aligned')
        if not curr: continue

        # Compare with preceding end boundary
        effective_prev_end = preceding_end if (i == 0) else (alignments[i-1].get('aligned')['end'] if alignments[i-1].get('aligned') else None)

        if effective_prev_end is not None:
            # Loosen overlap rule for high confidence matches
            min_gap = 0.01 if (curr.get('confidence', 0) > 0.8) else 0.05

            if curr['start'] < effective_prev_end + min_gap:
                curr['start'] = effective_prev_end + min_gap
                if curr['end'] <= curr['start']:
                    curr['end'] = curr['start'] + 0.1

    return alignments

def background_alignment(job_id, segments, speakers_to_process, audio_files_info, 
                        use_whisper, model_name, initial_prompt, use_whisperx=False):
    """Background task for alignment processing"""
    try:
        aligner = None
        if use_whisperx:
            aligner = get_whisperx_aligner(model_name)
            if aligner is None:
                raise Exception("Failed to initialize WhisperX aligner")
        elif use_whisper:
            aligner = get_whisper_aligner(model_name)
            if aligner is None:
                raise Exception("Failed to initialize Whisper aligner")

        all_alignments = []
        total_duration = max(s['end'] for s in segments) if segments else 0

        # Calculate total segments to process for better progress bar
        total_segments_to_process = sum(1 for s in segments if s['speaker'] in speakers_to_process)
        processed_segments_count = 0

        if total_segments_to_process == 0:
            total_segments_to_process = 1 # Avoid division by zero

        for speaker in speakers_to_process:
            # Update status
            with jobs_lock:
                if job_id in JOBS:
                    JOBS[job_id]['status'] = f'🎯 Aligning {speaker}...'

            audio_info = audio_files_info.get(speaker)
            if not audio_info:
                logger.warning(f"No audio info for speaker {speaker}")
                continue

            audio_path = audio_info['path']

            # Filter segments for this character
            char_segments = [s for s in segments if s['speaker'] == speaker]

            # Detect onsets for the speaker's audio (used for hump-to-word logic)
            speaker_audio, speaker_sr = audio_processor.load_audio(audio_path)
            onsets = audio_processor.detect_onsets(speaker_audio, speaker_sr)

            # Identify first 10% of segments for hump-based refinement
            num_refinement_segments = max(1, len(char_segments) // 10)

            if aligner:
                # Use aligner to find where the dialogue actually is
                try:
                    alignments = aligner.align_character_audio(
                        audio_path, char_segments,
                        language="ro",
                        initial_prompt=initial_prompt
                    )
                except Exception as e:
                    logger.error(f"Alignment failed for {speaker}: {e}")
                    alignments = [{'original': seg, 'aligned': None} for seg in char_segments]

                # Post-process for refinement using onsets
                try:
                    if alignments and onsets is not None and len(onsets) > 0:
                        # Apply general post-processing
                        alignments = apply_alignment_post_processing(alignments, onsets)

                        # Special case for VERY first segment: Force start at 0:00
                        # and refine duration based on word count vs. onsets
                        first_entry = alignments[0]
                        first_aligned = first_entry.get('aligned')
                        if first_aligned:
                            # Force start of the first segment to be the first audio hump (crop leading silence)
                            # But keep csv_start at 0 so it's placed at the track beginning
                            first_onset = onsets[0] if len(onsets) > 0 else 0.0
                            first_aligned['start'] = max(0, first_onset - 0.5) # Leave 0.5s padding
                            first_entry['original']['start'] = 0.0 # Force csv_start

                            # Count words in Romanian text
                            words_ro = first_entry['original'].get('text_ro', '').split()
                            word_count = len(words_ro)

                            if word_count > 0:
                                # If we have enough onsets, use them for end boundary
                                if len(onsets) >= word_count:
                                    target_onset_idx = min(word_count, len(onsets) - 1)
                                    first_aligned['end'] = max(onsets[target_onset_idx], first_aligned['start'] + 0.1)
                                else:
                                    # Fallback: estimate duration based on words (e.g. 0.4s/word)
                                    first_aligned['end'] = first_aligned['start'] + (word_count * 0.4)
                except Exception as e:
                    logger.error(f"Hump-refinement failed for {speaker}: {e}")

                for entry in alignments:
                    orig = entry['original']
                    aligned = entry.get('aligned')

                    alignment_info = {
                        'speaker': speaker,
                        'text_ro': orig.get('text_ro', ''),
                        'text_en': orig.get('text_en', ''),
                        'csv_start': orig['start'],
                        'csv_end': orig['end'],
                        'original_audio_filename': audio_info['filename'],
                        'confidence': 0.0,
                        'source_start': 0,
                        'source_end': orig['end'] - orig['start']
                    }

                    if aligned:
                        alignment_info.update({
                            'source_start': aligned.get('start', 0),
                            'source_end': aligned.get('end', orig['end'] - orig['start']),
                            'confidence': aligned.get('confidence', 0.0),
                            'aligned_text': aligned.get('text', '')
                        })
                    
                    all_alignments.append(alignment_info)

                    # Update progress per segment
                    processed_segments_count += 1
                    with jobs_lock:
                        if job_id in JOBS:
                            JOBS[job_id]['progress'] = int((processed_segments_count / total_segments_to_process) * 100)
            else:
                # No alignment, just map CSV timing to audio source starting at 0
                for orig in char_segments:
                    all_alignments.append({
                        'speaker': speaker,
                        'text_ro': orig.get('text_ro', ''),
                        'text_en': orig.get('text_en', ''),
                        'csv_start': orig['start'],
                        'csv_end': orig['end'],
                        'original_audio_filename': audio_info['filename'],
                        'source_start': 0,
                        'source_end': orig['end'] - orig['start'],
                        'confidence': 1.0
                    })
                    # Update progress per segment
                    processed_segments_count += 1
                    with jobs_lock:
                        if job_id in JOBS:
                            JOBS[job_id]['progress'] = int((processed_segments_count / total_segments_to_process) * 100)

        with jobs_lock:
            if job_id in JOBS:
                JOBS[job_id]['progress'] = 100
                JOBS[job_id]['status'] = '✅ Completed'
                JOBS[job_id]['results'] = all_alignments
                JOBS[job_id]['success'] = True
                JOBS[job_id]['total_duration'] = total_duration

        logger.info(f"Job {job_id} completed successfully with {len(all_alignments)} segments")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        logger.error(traceback.format_exc())
        with jobs_lock:
            if job_id in JOBS:
                JOBS[job_id]['status'] = '❌ Failed'
                JOBS[job_id]['error'] = str(e)
                JOBS[job_id]['success'] = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/system-info')
def system_info():
    """Get system information including GPU status"""
    gpu_info = {
        'available': torch.cuda.is_available(),
        'name': None,
        'memory': None,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
    }
    
    if gpu_info['available']:
        gpu_info['name'] = torch.cuda.get_device_name(0)
        gpu_info['memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    
    return jsonify({
        'device': device,
        'gpu': gpu_info,
        'cpu_count': os.cpu_count()
    })

@app.route('/api/process', methods=['POST'])
def process_audio():
    """Start alignment process"""
    try:
        # Get metadata and settings
        segments = json.loads(request.form.get('segments', '[]'))
        use_whisper = request.form.get('use_whisper') == 'true'
        use_whisperx = request.form.get('use_whisperx') == 'true'
        use_mms = request.form.get('use_mms') == 'true'
        speakers_to_process = request.form.getlist('speakers')
        model_name = request.form.get('whisper_model', 'base')
        initial_prompt = request.form.get('initial_prompt', '')
        
        if not segments or not speakers_to_process:
            return jsonify({'success': False, 'error': 'Missing metadata or speakers'}), 400

        job_id = str(uuid.uuid4())
        
        # Save audio files
        audio_files_info = {}
        files_to_track = []
        
        for speaker in speakers_to_process:
            audio_key = f'audio_{speaker}'
            audio_file = request.files.get(audio_key)
            if audio_file:
                # Sanitize filename
                safe_speaker = secure_filename(speaker)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{job_id}_{safe_speaker}_{timestamp}.wav"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                audio_file.save(filepath)
                audio_files_info[speaker] = {
                    'path': filepath,
                    'filename': audio_file.filename,
                    'size': os.path.getsize(filepath)
                }
                files_to_track.append({
                    'path': filepath,
                    'type': 'upload'
                })

        if not audio_files_info:
            return jsonify({'success': False, 'error': 'No audio files uploaded'}), 400

        # Get total duration
        total_duration = max(s['end'] for s in segments) if segments else 0

        # Create job entry
        with jobs_lock:
            JOBS[job_id] = {
                'id': job_id,
                'progress': 0,
                'status': '⏳ Starting...',
                'results': None,
                'success': None,
                'error': None,
                'files': files_to_track,
                'output_files': [],
                'total_duration': total_duration,
                'created_at': time.time(),
                'metadata': {
                    'speakers': speakers_to_process,
                    'use_whisper': use_whisper,
                    'use_whisperx': use_whisperx,
                    'use_mms': use_mms,
                    'model': model_name if (use_whisper or use_whisperx) else 'none',
                    'segment_count': len(segments)
                }
            }

        # Start background thread
        thread = threading.Thread(
            target=background_alignment,
            args=(job_id, segments, speakers_to_process, audio_files_info, 
                  use_whisper, model_name, initial_prompt, use_whisperx)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Processing started'
        })

    except Exception as e:
        logger.error(f"Process error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/generate-final', methods=['POST'])
def generate_final():
    """Generate final audio tracks"""
    try:
        data = request.json
        job_id = data.get('job_id')
        adjusted_data = data.get('alignments', [])
        download_all = data.get('download_all', False)

        if not job_id or not adjusted_data:
            return jsonify({'success': False, 'error': 'Missing job_id or alignments'}), 400

        with jobs_lock:
            job = JOBS.get(job_id)
            if not job:
                return jsonify({'success': False, 'error': 'Job not found'}), 404
            total_duration = job.get('total_duration', 0)

        # Group by speaker
        speakers_segments = {}
        for seg in adjusted_data:
            speaker = seg['speaker']
            if speaker not in speakers_segments:
                speakers_segments[speaker] = []
            speakers_segments[speaker].append(seg)

        output_tracks = []
        source_audio_cache = {}
        output_files = []

        # Process each speaker
        for speaker, speaker_segs in speakers_segments.items():
            aligned_for_reconstruction = []
            sr = 16000

            # Find audio path for this speaker from the original upload
            audio_path = None
            with jobs_lock:
                for file_info in job.get('files', []):
                    if file_info.get('path', '').find(speaker) != -1:
                        audio_path = file_info.get('path')
                        break

            if not audio_path or not os.path.exists(audio_path):
                logger.warning(f"Audio file not found for speaker {speaker}")
                continue

            # Load audio
            if audio_path not in source_audio_cache:
                source_audio_cache[audio_path] = audio_processor.load_audio(audio_path)
            audio, sr = source_audio_cache[audio_path]

            # Extract segments
            for seg in speaker_segs:
                try:
                    tempo = float(seg.get('tempo', 1.0))
                    extracted = audio_processor.extract_segment(
                        audio, sr,
                        float(seg['source_start']),
                        float(seg['source_end']),
                        seg.get('text_ro', ''),
                        tempo=tempo
                    )

                    aligned_for_reconstruction.append((
                        float(seg['csv_start']),
                        float(seg['csv_end']),
                        extracted
                    ))
                except Exception as e:
                    logger.error(f"Failed to extract segment for {speaker}: {e}")

            if aligned_for_reconstruction:
                # Reconstruct full track
                full_track = alignment_utils.reconstruct_character_track(
                    aligned_for_reconstruction, sr, total_duration=total_duration
                )

                # Save track
                safe_speaker = secure_filename(speaker)
                output_filename = f"{job_id}_{safe_speaker}_aligned.wav"
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                sf.write(output_path, full_track, sr)

                output_tracks.append({
                    'speaker': speaker,
                    'file': output_filename,
                    'duration': len(full_track) / sr,
                    'size': os.path.getsize(output_path)
                })
                output_files.append(output_filename)

        # Update job with output files
        with jobs_lock:
            if job_id in JOBS:
                JOBS[job_id]['output_files'] = output_files

        if download_all and len(output_tracks) > 1:
            # Create zip file
            zip_filename = f"{job_id}_all_tracks.zip"
            zip_path = os.path.join(app.config['OUTPUT_FOLDER'], zip_filename)
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for track in output_tracks:
                    file_path = os.path.join(app.config['OUTPUT_FOLDER'], track['file'])
                    zipf.write(file_path, track['file'])
            
            output_tracks.append({
                'speaker': 'all',
                'file': zip_filename,
                'duration': sum(t['duration'] for t in output_tracks),
                'size': os.path.getsize(zip_path)
            })

        return jsonify({
            'success': True,
            'message': 'Tracks generated successfully',
            'output_files': output_tracks
        })

    except Exception as e:
        logger.error(f"Generate final error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download/<filename>')
def download_file(filename):
    """Download a generated file"""
    filename = secure_filename(filename)
    return send_file(
        os.path.join(app.config['OUTPUT_FOLDER'], filename),
        as_attachment=True,
        download_name=filename
    )

@app.route('/api/preview/<filename>')
def preview_file(filename):
    """Preview a generated file"""
    filename = secure_filename(filename)
    return send_file(
        os.path.join(app.config['OUTPUT_FOLDER'], filename),
        mimetype='audio/wav'
    )

@app.route('/api/original-audio/<job_id>/<speaker>')
def get_original_audio(job_id, speaker):
    """Get original uploaded audio for preview"""
    with jobs_lock:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        # Find audio file for speaker
        for file_info in job.get('files', []):
            if speaker in file_info.get('path', ''):
                return send_file(file_info['path'], mimetype='audio/wav')
        
        return jsonify({'error': 'Speaker audio not found'}), 404

@app.route('/api/waveform-data/<job_id>/<speaker>')
def get_waveform_data(job_id, speaker):
    """Get downsampled waveform peaks for visualization"""
    with jobs_lock:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404

        audio_path = None
        for file_info in job.get('files', []):
            if speaker in file_info.get('path', ''):
                audio_path = file_info.get('path')
                break

        if not audio_path or not os.path.exists(audio_path):
            return jsonify({'error': 'Speaker audio not found'}), 404

    try:
        audio, sr = audio_processor.load_audio(audio_path)

        # Downsample to ~100 points per second for UI performance
        target_points = int(len(audio) / sr * 100)
        if target_points > 10000: target_points = 10000

        if len(audio) > target_points:
            # Simple max-pooling for peaks
            win_size = len(audio) // target_points
            peaks = []
            for i in range(0, len(audio) - win_size, win_size):
                peaks.append(float(np.max(np.abs(audio[i:i+win_size]))))
        else:
            peaks = [float(x) for x in audio]

        return jsonify({
            'peaks': peaks,
            'sample_rate': sr,
            'duration': len(audio) / sr
        })
    except Exception as e:
        logger.error(f"Waveform data error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/realign', methods=['POST'])
def realign_segments():
    """Re-align segments starting from a specific index with a time offset"""
    try:
        data = request.json
        job_id = data.get('job_id')
        speaker = data.get('speaker')
        start_index = data.get('start_index')
        offset_time = float(data.get('offset_time', 0.0))
        current_alignments = data.get('current_alignments', [])

        if not job_id or not speaker or start_index is None:
            return jsonify({'success': False, 'error': 'Missing parameters'}), 400

        with jobs_lock:
            job = JOBS.get(job_id)
            if not job:
                return jsonify({'success': False, 'error': 'Job not found'}), 404

            # 1. Update existing results with current UI state to preserve manual edits
            if current_alignments:
                for idx, adj in enumerate(current_alignments):
                    if idx < len(job['results']):
                        job['results'][idx].update({
                            'source_start': adj['source_start'],
                            'source_end': adj['source_end'],
                            'tempo': adj['tempo']
                        })

            # Find all results and their original indices for this speaker
            all_results = job.get('results', [])
            speaker_data = [(idx, r) for idx, r in enumerate(all_results) if r['speaker'] == speaker]

            if start_index >= len(speaker_data):
                return jsonify({'success': False, 'error': 'Invalid start index'}), 400

            # Identify segments and their indices to re-align
            indices_to_update = [item[0] for item in speaker_data[start_index:]]
            segments_to_realign = [item[1] for item in speaker_data[start_index:]]

            # Get original metadata for these segments
            expected_segments = []
            for r in segments_to_realign:
                expected_segments.append({
                    'start': r['csv_start'],
                    'end': r['csv_end'],
                    'speaker': r['speaker'],
                    'text_ro': r['text_ro'],
                    'text_en': r['text_en']
                })

            # Find audio path
            audio_path = None
            for file_info in job.get('files', []):
                if speaker in file_info.get('path', ''):
                    audio_path = file_info.get('path')
                    break

            if not audio_path:
                return jsonify({'success': False, 'error': 'Audio not found'}), 404

        # Initialize Aligner (using singleton getters to keep model loaded)
        use_whisperx = job['metadata'].get('use_whisperx', False)
        use_whisper = job['metadata'].get('use_whisper', False)
        model_name = job['metadata'].get('model', 'base')

        aligner = None
        if use_whisperx:
            aligner = get_whisperx_aligner(model_name)
        elif use_whisper:
            aligner = get_whisper_aligner(model_name)

        if aligner is None:
            return jsonify({'success': False, 'error': 'Aligner not available'}), 500

        # Re-align with offset
        # Provide text context from preceding segment for better linguistic context
        initial_prompt = ""
        if start_index > 0:
            initial_prompt = speaker_data[start_index - 1][1].get('text_ro', '')

        new_alignments = aligner.align_character_audio(
            audio_path, expected_segments,
            language="ro",
            offset_time=offset_time,
            initial_prompt=initial_prompt
        )

        # Determine the preceding end boundary for the first segment in the batch
        preceding_end = None
        if start_index > 0:
            preceding_end = speaker_data[start_index - 1][1].get('source_end')

        # Apply post-processing refinement to new alignments
        speaker_audio, speaker_sr = audio_processor.load_audio(audio_path)
        onsets = audio_processor.detect_onsets(speaker_audio, speaker_sr)
        new_alignments = apply_alignment_post_processing(new_alignments, onsets, preceding_end=preceding_end)

        # Update job results with newly aligned segments using exact indices
        with jobs_lock:
            for i, new_entry in enumerate(new_alignments):
                if i < len(indices_to_update):
                    global_idx = indices_to_update[i]
                    orig = new_entry['original']
                    aligned = new_entry.get('aligned')

                    update_info = {
                        'source_start': aligned.get('start', 0) if aligned else 0,
                        'source_end': aligned.get('end', orig['end'] - orig['start']) if aligned else orig['end'] - orig['start'],
                        'confidence': aligned.get('confidence', 0.0) if aligned else 0.0
                    }

                    job['results'][global_idx].update(update_info)

        return jsonify({
            'success': True,
            'message': f'Re-aligned {len(new_alignments)} segments',
            'results': job['results']
        })

    except Exception as e:
        logger.error(f"Re-align error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/job/<job_id>')
def job_status(job_id):
    """Get job status"""
    with jobs_lock:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        # Create safe copy without full file paths
        safe_job = {
            'id': job['id'],
            'progress': job['progress'],
            'status': job['status'],
            'success': job['success'],
            'error': job['error'],
            'created_at': job['created_at'],
            'metadata': job.get('metadata', {}),
            'total_duration': job.get('total_duration', 0)
        }
        
        if job.get('results') is not None:
            safe_job['results'] = job['results']
        
        if job.get('output_files'):
            safe_job['output_files'] = job['output_files']
        
        return jsonify(safe_job)

@app.route('/api/upload-tutorial', methods=['POST'])
def upload_tutorial():
    """Upload a video tutorial"""
    try:
        video_file = request.files.get('video')
        if not video_file:
            return jsonify({'success': False, 'error': 'No video file provided'}), 400

        filename = secure_filename(video_file.filename)
        filepath = os.path.join(app.config['TUTORIAL_FOLDER'], filename)
        video_file.save(filepath)

        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'Tutorial uploaded successfully'
        })
    except Exception as e:
        logger.error(f"Upload tutorial error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload-document', methods=['POST'])
def upload_document():
    """Upload a document (PDF, DOCX)"""
    try:
        doc_file = request.files.get('document')
        if not doc_file:
            return jsonify({'success': False, 'error': 'No document file provided'}), 400

        filename = secure_filename(doc_file.filename)
        filepath = os.path.join(app.config['DOCUMENT_FOLDER'], filename)
        doc_file.save(filepath)

        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'Document uploaded successfully'
        })
    except Exception as e:
        logger.error(f"Upload document error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/tutorials', methods=['GET'])
def list_tutorials():
    """List available video tutorials"""
    try:
        tutorials = []
        for filename in os.listdir(app.config['TUTORIAL_FOLDER']):
            if filename.lower().endswith(('.mp4', '.webm', '.ogg', '.mov')):
                tutorials.append({
                    'filename': filename,
                    'url': f"/api/serve-tutorial/{filename}"
                })
        return jsonify({'success': True, 'tutorials': tutorials})
    except Exception as e:
        logger.error(f"List tutorials error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List available documents"""
    try:
        documents = []
        for filename in os.listdir(app.config['DOCUMENT_FOLDER']):
            if filename.lower().endswith(('.pdf', '.docx')):
                documents.append({
                    'filename': filename,
                    'url': f"/api/serve-document/{filename}",
                    'type': 'pdf' if filename.lower().endswith('.pdf') else 'docx'
                })
        return jsonify({'success': True, 'documents': documents})
    except Exception as e:
        logger.error(f"List documents error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/serve-tutorial/<filename>')
def serve_tutorial(filename):
    """Serve a tutorial video file"""
    return send_from_directory(app.config['TUTORIAL_FOLDER'], secure_filename(filename))

@app.route('/api/serve-document/<filename>')
def serve_document(filename):
    """Serve a document file"""
    return send_from_directory(app.config['DOCUMENT_FOLDER'], secure_filename(filename))

@app.route('/api/upload-metadata', methods=['POST'])
def upload_metadata():
    """Upload and parse metadata file"""
    temp_dir = None
    try:
        metadata_file = request.files.get('metadata')
        if not metadata_file:
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        # Create temp directory
        temp_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(temp_dir, secure_filename(metadata_file.filename))
        metadata_file.save(file_path)

        segments = []
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.csv':
            df = pd.read_csv(file_path)
            df = df.fillna('')
            
            # Validate required columns
            required = ['start', 'end', 'speaker']
            missing = [col for col in required if col not in df.columns]
            if missing:
                return jsonify({
                    'success': False, 
                    'error': f'Missing columns: {", ".join(missing)}'
                }), 400
            
            for _, row in df.iterrows():
                segments.append({
                    'start': float(row['start']),
                    'end': float(row['end']),
                    'speaker': str(row['speaker']),
                    'text_en': str(row.get('en', '')),
                    'text_ro': str(row.get('ro', ''))
                })
        elif file_ext == '.srt':
            segments = alignment_utils.parse_srt(file_path)
        else:
            return jsonify({
                'success': False, 
                'error': 'Unsupported file format. Use CSV or SRT'
            }), 400

        if not segments:
            return jsonify({'success': False, 'error': 'No valid segments found'}), 400

        # Calculate statistics
        speakers = sorted(list(set(s['speaker'] for s in segments)))
        total_duration = max(s['end'] for s in segments)
        segments_per_speaker = {
            speaker: len([s for s in segments if s['speaker'] == speaker])
            for speaker in speakers
        }

        return jsonify({
            'success': True,
            'speakers': speakers,
            'segments': segments,
            'filename': metadata_file.filename,
            'stats': {
                'total_segments': len(segments),
                'total_duration': total_duration,
                'speaker_count': len(speakers),
                'segments_per_speaker': segments_per_speaker
            }
        })

    except Exception as e:
        logger.error(f"Upload metadata error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

@app.route('/api/cancel-job/<job_id>', methods=['POST'])
def cancel_job(job_id):
    """Cancel a running job"""
    with jobs_lock:
        if job_id in JOBS:
            JOBS[job_id]['status'] = '❌ Cancelled'
            JOBS[job_id]['success'] = False
            return jsonify({'success': True, 'message': 'Job cancelled'})
    return jsonify({'success': False, 'error': 'Job not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)