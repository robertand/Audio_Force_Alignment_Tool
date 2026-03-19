from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import pandas as pd
import numpy as np
import tempfile
import shutil
import threading
import uuid
import time
from werkzeug.utils import secure_filename
import torch
import torchaudio
import soundfile as sf
from utils.audio_processor import AudioProcessor
from utils.vad_processor import VADProcessor
from utils.alignment_utils import AlignmentUtils
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['OUTPUT_FOLDER'] = 'outputs'

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize processors
audio_processor = AudioProcessor()
vad_processor = VADProcessor()
alignment_utils = AlignmentUtils()

# Job management
JOBS = {}
jobs_lock = threading.Lock()

def cleanup_old_jobs():
    """Periodically remove jobs older than 1 hour and their associated files"""
    while True:
        try:
            time.sleep(600) # Run every 10 minutes
            now = time.time()
            to_delete = []
            with jobs_lock:
                for job_id, job in JOBS.items():
                    # If job was created more than 1 hour ago
                    if now - job.get('created_at', 0) > 3600:
                        to_delete.append(job_id)

            for job_id in to_delete:
                with jobs_lock:
                    job = JOBS.pop(job_id, None)
                    if job:
                        # Delete audio files
                        audio_info = job.get('audio_files_info', {})
                        for speaker, info in audio_info.items():
                            path = info.get('path')
                            if path and os.path.exists(path):
                                try:
                                    os.remove(path)
                                except:
                                    pass
        except Exception as e:
            print(f"Cleanup error: {e}")

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_jobs, daemon=True)
cleanup_thread.start()

# Lazy-loaded aligner
_aligner = None
aligner_lock = threading.Lock()

def get_aligner():
    global _aligner
    with aligner_lock:
        if _aligner is None:
            from utils.whisper_aligner import DialogueAligner
            _aligner = DialogueAligner()
        return _aligner

def background_alignment(job_id, segments, speakers_to_process, audio_files_info, use_whisper, model_name, initial_prompt):
    try:
        aligner = get_aligner() if use_whisper else None

        all_alignments = []
        total_steps = len(speakers_to_process)

        for idx, speaker in enumerate(speakers_to_process):
            with jobs_lock:
                if job_id in JOBS:
                    JOBS[job_id]['progress'] = int((idx / total_steps) * 100)
                    JOBS[job_id]['status'] = f'Aligning speaker: {speaker}'

            audio_info = audio_files_info.get(speaker)
            if not audio_info:
                continue

            audio_path = audio_info['path']

            # Filter segments for this character
            char_segments = [s for s in segments if s['speaker'] == speaker]

            if use_whisper and aligner:
                # Use Whisper to find where the dialogue actually is
                # Protect model transcription with lock to avoid race conditions on the same model instance
                with aligner_lock:
                    alignments = aligner.align_character_audio(
                        audio_path, char_segments,
                        model_name=model_name,
                        initial_prompt=initial_prompt
                    )

                for entry in alignments:
                    orig = entry['original']
                    aligned = entry['aligned']

                    alignment_info = {
                        'speaker': speaker,
                        'text_ro': orig.get('text_ro', ''),
                        'csv_start': orig['start'],
                        'csv_end': orig['end'],
                        'original_audio_filename': audio_info['filename']
                    }

                    if aligned:
                        alignment_info.update({
                            'source_start': aligned['start'],
                            'source_end': aligned['end'],
                            'confidence': aligned['confidence']
                        })
                    else:
                        alignment_info.update({
                            'source_start': 0,
                            'source_end': orig['end'] - orig['start'],
                            'confidence': 0
                        })
                    all_alignments.append(alignment_info)
            else:
                # No whisper, just mapping CSV timing to audio source starting at 0
                for orig in char_segments:
                    all_alignments.append({
                        'speaker': speaker,
                        'text_ro': orig.get('text_ro', ''),
                        'csv_start': orig['start'],
                        'csv_end': orig['end'],
                        'original_audio_filename': audio_info['filename'],
                        'source_start': 0,
                        'source_end': orig['end'] - orig['start'],
                        'confidence': 1.0
                    })

        with jobs_lock:
            if job_id in JOBS:
                JOBS[job_id]['progress'] = 100
                JOBS[job_id]['status'] = 'Completed'
                JOBS[job_id]['results'] = all_alignments
                JOBS[job_id]['success'] = True

    except Exception as e:
        traceback.print_exc()
        with jobs_lock:
            if job_id in JOBS:
                JOBS[job_id]['status'] = 'Failed'
                JOBS[job_id]['error'] = str(e)
                JOBS[job_id]['success'] = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_audio():
    try:
        # Get metadata and settings
        segments = json.loads(request.form.get('segments', '[]'))
        use_whisper = request.form.get('use_whisper') == 'true'
        speakers_to_process = request.form.getlist('speakers')
        model_name = request.form.get('whisper_model', 'base')
        initial_prompt = request.form.get('initial_prompt', '')
        
        if not segments or not speakers_to_process:
            return jsonify({'success': False, 'error': 'Missing metadata or speakers'}), 400

        job_id = str(uuid.uuid4())
        
        # Save audio files temporarily
        audio_files_info = {}
        for speaker in speakers_to_process:
            audio_key = f'audio_{speaker}'
            audio_file = request.files.get(audio_key)
            if audio_file:
                perm_filename = f"{job_id}_{secure_filename(speaker)}.wav"
                perm_path = os.path.join(app.config['UPLOAD_FOLDER'], perm_filename)
                audio_file.save(perm_path)
                audio_files_info[speaker] = {
                    'path': perm_path,
                    'filename': audio_file.filename
                }

        # Find total duration from ALL segments in metadata
        total_duration = 0
        if segments:
            total_duration = max(s['end'] for s in segments)

        with jobs_lock:
            JOBS[job_id] = {
                'progress': 0,
                'status': 'Starting...',
                'results': None,
                'success': None,
                'audio_files_info': audio_files_info, # Store for authorized lookup in /generate_final
                'total_duration': total_duration,
                'created_at': time.time()
            }

        # Start thread
        thread = threading.Thread(
            target=background_alignment,
            args=(job_id, segments, speakers_to_process, audio_files_info, use_whisper, model_name, initial_prompt)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'job_id': job_id
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/generate_final', methods=['POST'])
def generate_final():
    try:
        data = request.json
        job_id = data.get('job_id')
        adjusted_data = data.get('alignments', [])

        if not job_id or not adjusted_data:
            return jsonify({'success': False, 'error': 'Missing job_id or alignments'}), 400

        with jobs_lock:
            job = JOBS.get(job_id)
            if not job:
                return jsonify({'success': False, 'error': 'Job not found'}), 404
            audio_files_info = job['audio_files_info']
            total_duration = job['total_duration']

        # Group by speaker
        speakers_segments = {}
        for seg in adjusted_data:
            speaker = seg['speaker']
            if speaker not in speakers_segments:
                speakers_segments[speaker] = []
            speakers_segments[speaker].append(seg)

        output_tracks = []
        source_audio_cache = {}

        for speaker, speaker_segs in speakers_segments.items():
            aligned_for_reconstruction = []
            sr = 16000

            for seg in speaker_segs:
                # Security: Look up path from server-side state using speaker name
                # Do NOT trust path from client
                speaker_info = audio_files_info.get(speaker)
                if not speaker_info:
                    continue

                audio_path = speaker_info['path']
                if audio_path not in source_audio_cache:
                    source_audio_cache[audio_path] = audio_processor.load_audio(audio_path)

                audio, sr = source_audio_cache[audio_path]

                extracted = audio_processor.extract_segment(
                    audio, sr,
                    float(seg['source_start']),
                    float(seg['source_end']),
                    seg.get('text_ro', '')
                )

                aligned_for_reconstruction.append((
                    float(seg['csv_start']),
                    float(seg['csv_end']),
                    extracted
                ))

            if aligned_for_reconstruction:
                full_track = alignment_utils.reconstruct_character_track(
                    aligned_for_reconstruction, sr, total_duration=total_duration
                )

                output_filename = f"track_{job_id}_{secure_filename(speaker)}.wav"
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                sf.write(output_path, full_track, sr)

                output_tracks.append({
                    'speaker': speaker,
                    'file': output_filename
                })

        return jsonify({
            'success': True,
            'message': 'Tracks generated successfully',
            'output_files': output_tracks
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/download/<filename>')
def download_file(filename):
    # Security: Ensure filename is relative and doesn't contain path traversal
    filename = secure_filename(filename)
    return send_file(
        os.path.join(app.config['OUTPUT_FOLDER'], filename),
        as_attachment=True
    )

@app.route('/preview/<filename>')
def preview_file(filename):
    filename = secure_filename(filename)
    return send_file(
        os.path.join(app.config['OUTPUT_FOLDER'], filename),
        mimetype='audio/wav'
    )

@app.route('/get_original_audio/<job_id>/<speaker>')
def get_original_audio(job_id, speaker):
    with jobs_lock:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        audio_info = job['audio_files_info'].get(speaker)
        if not audio_info:
            return jsonify({'error': 'Speaker audio not found'}), 404

        return send_file(audio_info['path'], mimetype='audio/wav')

@app.route('/job_status/<job_id>')
def job_status(job_id):
    with jobs_lock:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        # Don't return internal audio_files_info to client
        client_job = job.copy()
        client_job.pop('audio_files_info', None)
        return jsonify(client_job)

@app.route('/upload_metadata', methods=['POST'])
def upload_metadata():
    temp_dir = None
    try:
        metadata_file = request.files.get('metadata')
        if not metadata_file:
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        temp_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(temp_dir, secure_filename(metadata_file.filename))
        metadata_file.save(file_path)

        segments = []
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            df = df.fillna('')
            for _, row in df.iterrows():
                segments.append({
                    'start': float(row['start']),
                    'end': float(row['end']),
                    'speaker': str(row['speaker']),
                    'text_en': str(row.get('en', '')),
                    'text_ro': str(row.get('ro', ''))
                })
        elif file_path.endswith('.srt'):
            segments = alignment_utils.parse_srt(file_path)
        else:
            return jsonify({'success': False, 'error': 'Unsupported file format'}), 400

        speakers = sorted(list(set(s['speaker'] for s in segments)))

        return jsonify({
            'success': True,
            'speakers': speakers,
            'segments': segments,
            'filename': metadata_file.filename
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
