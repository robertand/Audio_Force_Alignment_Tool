from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import pandas as pd
import numpy as np
import tempfile
import shutil
from werkzeug.utils import secure_filename
import torch
import torchaudio
import soundfile as sf
from utils.audio_processor import AudioProcessor
from utils.vad_processor import VADProcessor
from utils.alignment_utils import AlignmentUtils
import traceback
import threading
import uuid
import time

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

# Global job storage
jobs = {}

# Lazy-loaded aligners
_aligner = None
_aligner_type = None

def get_aligner(model_size="base", aligner_type="whisper"):
    global _aligner, _aligner_type

    # Reload if type changed OR if model size changed (for whisper)
    should_reload = (_aligner is None or _aligner_type != aligner_type)
    if not should_reload and aligner_type == "whisper" and getattr(_aligner, 'model_name', None) != model_size:
        should_reload = True

    if should_reload:
        if aligner_type == "mms":
            from utils.mms_aligner import MMSDialogueAligner
            _aligner = MMSDialogueAligner()
        else:
            from utils.whisper_aligner import DialogueAligner
            _aligner = DialogueAligner(model_name=model_size)
        _aligner_type = aligner_type

    return _aligner

@app.errorhandler(Exception)
def handle_exception(e):
    error_details = traceback.format_exc()
    status_code = 500
    if hasattr(e, 'code'):
        status_code = e.code
    return jsonify({
        'success': False,
        'error': str(e),
        'traceback': error_details
    }), status_code

@app.route('/')
def index():
    return render_template('index.html')

def run_job(job_id, segments, speakers_to_process, files_dict, use_whisper, model_size, aligner_type):
    temp_dir = None
    try:
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['total_speakers'] = len(speakers_to_process)
        
        abs_max_end = max(s['end'] for s in segments) if segments else 0
        temp_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])
        
        output_tracks = []
        failed_speakers = []
        
        aligner = None
        if use_whisper:
            aligner = get_aligner(model_size, aligner_type)

        for i, speaker in enumerate(speakers_to_process):
            try:
                jobs[job_id]['current_speaker'] = speaker
                jobs[job_id]['progress'] = int((i / len(speakers_to_process)) * 100)

                audio_content = files_dict.get(speaker)
                if not audio_content:
                    continue

                audio_path = os.path.join(temp_dir, f"input_{secure_filename(speaker)}.wav")
                with open(audio_path, 'wb') as f:
                    f.write(audio_content)

                char_audio, sr = audio_processor.load_audio(audio_path)
                char_segments = [s for s in segments if s['speaker'] == speaker]
                aligned_for_reconstruction = []

                if use_whisper and aligner:
                    alignments = aligner.align_character_audio(audio_path, char_segments)
                    for entry in alignments:
                        orig = entry['original']
                        aligned = entry['aligned']
                        if aligned:
                            extracted = audio_processor.extract_segment(
                                char_audio, sr, aligned['start'], aligned['end'], orig.get('text_ro', '')
                            )
                            aligned_for_reconstruction.append((orig['start'], orig['end'], extracted))
                else:
                    current_ptr = 0.0
                    total_dur = len(char_audio) / sr
                    for orig in char_segments:
                        dur = orig['end'] - orig['start']
                        start = min(current_ptr, total_dur)
                        end = min(current_ptr + dur, total_dur)
                        extracted = char_audio[int(start*sr):int(end*sr)]
                        aligned_for_reconstruction.append((orig['start'], orig['end'], extracted))
                        current_ptr = end

                if aligned_for_reconstruction:
                    full_track = alignment_utils.reconstruct_character_track(
                        aligned_for_reconstruction, sr, max_duration=abs_max_end
                    )
                    # Use a stable filename for the character track so it persists even if processing is repeated
                    output_filename = f"track_{secure_filename(speaker)}.wav"
                    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                    sf.write(output_path, full_track, sr)
                    output_tracks.append({'speaker': speaker, 'file': output_filename})
            except Exception as speaker_err:
                traceback.print_exc()
                failed_speakers.append({'speaker': speaker, 'error': str(speaker_err)})

        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['result'] = {
            'success': True,
            'output_files': output_tracks,
            'failed_speakers': failed_speakers
        }
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
        jobs[job_id]['traceback'] = traceback.format_exc()
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

@app.route('/process', methods=['POST'])
def process_audio():
    try:
        segments = json.loads(request.form.get('segments', '[]'))
        use_whisper = request.form.get('use_whisper') == 'true'
        speakers_to_process = request.form.getlist('speakers')
        model_size = request.form.get('model_size', 'base')
        aligner_type = request.form.get('aligner_type', 'whisper')

        if not segments or not speakers_to_process:
            return jsonify({'success': False, 'error': 'Missing metadata or speakers'}), 400

        files_dict = {}
        for speaker in speakers_to_process:
            audio_file = request.files.get(f'audio_{speaker}')
            if audio_file:
                files_dict[speaker] = audio_file.read()

        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            'id': job_id,
            'status': 'queued',
            'progress': 0,
            'created_at': time.time()
        }

        thread = threading.Thread(target=run_job, args=(
            job_id, segments, speakers_to_process, files_dict, use_whisper, model_size, aligner_type
        ))
        thread.start()

        return jsonify({'success': True, 'job_id': job_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/job/status/<job_id>')
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'success': False, 'error': 'Job not found'}), 404
    return jsonify(job)

def process_alignment(audio_path, df, timecode_data, temp_dir):
    """Main alignment processing function"""
    
    # Load audio
    audio, sr = audio_processor.load_audio(audio_path)
    
    # Process VAD to detect speech segments
    vad_segments = vad_processor.process_vad(audio, sr)
    
    # Get expected segments from CSV
    expected_segments = []
    for idx, row in df.iterrows():
        expected_segments.append({
            'start': row['start'],
            'end': row['end'],
            'speaker': row['speaker'],
            'text_en': row['en'],
            'text_ro': row['ro']
        })
    
    # Validate and adjust segments using VAD
    validated_segments = alignment_utils.validate_segments(
        expected_segments, vad_segments, sr
    )
    
    # Cut audio according to validated segments
    output_files = []
    details = []
    
    for i, segment in enumerate(validated_segments):
        # Extract segment audio
        start_sample = int(segment['start'] * sr)
        end_sample = int(segment['end'] * sr)
        segment_audio = audio[start_sample:end_sample]
        
        # Apply crossfade at boundaries to avoid clicks
        if i > 0:
            prev_end = validated_segments[i-1]['end']
            if segment['start'] - prev_end < 0.1:  # If segments are very close
                segment_audio = alignment_utils.apply_crossfade(
                    segment_audio, sr, 0.01
                )
        
        # Save segment
        output_filename = f"segment_{i:04d}_{segment['speaker']}.wav"
        output_path = os.path.join(temp_dir, output_filename)
        sf.write(output_path, segment_audio, sr)
        
        # Move to output folder
        final_output = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        shutil.move(output_path, final_output)
        output_files.append({
            'file': output_filename,
            'path': final_output,
            'start': segment['start'],
            'end': segment['end'],
            'speaker': segment['speaker'],
            'text_en': segment['text_en'],
            'text_ro': segment['text_ro']
        })
        
        details.append({
            'segment': i,
            'speaker': segment['speaker'],
            'original_start': segment['original_start'],
            'original_end': segment['original_end'],
            'adjusted_start': segment['start'],
            'adjusted_end': segment['end'],
            'duration': segment['end'] - segment['start'],
            'vad_confidence': segment.get('vad_confidence', 1.0)
        })
    
    # Create concatenated audio if needed
    if len(output_files) > 1:
        concat_audio = alignment_utils.concatenate_segments(
            [f['path'] for f in output_files]
        )
        concat_path = os.path.join(app.config['OUTPUT_FOLDER'], 'concatenated_output.wav')
        sf.write(concat_path, concat_audio, sr)
        output_files.append({
            'file': 'concatenated_output.wav',
            'path': concat_path,
            'type': 'concatenated'
        })
    
    return {
        'segments': output_files,
        'details': details
    }

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(app.config['OUTPUT_FOLDER'], filename),
        as_attachment=True
    )

@app.route('/preview/<filename>')
def preview_file(filename):
    return send_file(
        os.path.join(app.config['OUTPUT_FOLDER'], filename),
        mimetype='audio/wav'
    )

@app.route('/download_all')
def download_all():
    import zipfile
    zip_path = os.path.join(app.config['OUTPUT_FOLDER'], 'all_tracks.zip')
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(app.config['OUTPUT_FOLDER']):
            for file in files:
                if file.endswith('.wav'):
                    zipf.write(os.path.join(root, file), file)
    return send_file(zip_path, as_attachment=True)

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

        # Extract unique speakers
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

@app.route('/get_details')
def get_details():
    """Get processing details for current session"""
    details_file = os.path.join(app.config['OUTPUT_FOLDER'], 'details.json')
    if os.path.exists(details_file):
        with open(details_file, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'No details available'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)