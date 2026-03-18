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

# Lazy-loaded aligner
_aligner = None

def get_aligner():
    global _aligner
    if _aligner is None:
        from utils.whisper_aligner import DialogueAligner
        _aligner = DialogueAligner()
    return _aligner

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_audio():
    temp_dir = None
    try:
        # Get metadata and settings
        segments = json.loads(request.form.get('segments', '[]'))
        use_whisper = request.form.get('use_whisper') == 'true'
        speakers_to_process = request.form.getlist('speakers')
        
        if not segments or not speakers_to_process:
            return jsonify({'success': False, 'error': 'Missing metadata or speakers'}), 400

        temp_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])
        
        output_tracks = []
        
        # Optional Whisper Aligner
        aligner = None
        if use_whisper:
            aligner = get_aligner()

        for speaker in speakers_to_process:
            audio_key = f'audio_{speaker}'
            audio_file = request.files.get(audio_key)

            if not audio_file:
                continue # Skip if no audio provided for this selected speaker

            audio_path = os.path.join(temp_dir, f"input_{secure_filename(speaker)}.wav")
            audio_file.save(audio_path)

            # Load input audio
            char_audio, sr = audio_processor.load_audio(audio_path)

            # Filter segments for this character
            char_segments = [s for s in segments if s['speaker'] == speaker]

            aligned_for_reconstruction = []

            if use_whisper and aligner:
                # Use Whisper to find where the dialogue actually is
                alignments = aligner.align_character_audio(audio_path, char_segments)

                for entry in alignments:
                    orig = entry['original']
                    aligned = entry['aligned']

                    if aligned:
                        # Extract the segment from the character's audio based on Whisper's timing
                        extracted = audio_processor.extract_segment(
                            char_audio, sr, aligned['start'], aligned['end'], orig.get('text_ro', '')
                        )
                        aligned_for_reconstruction.append((orig['start'], orig['end'], extracted))
                    else:
                        # Fallback or skip
                        pass
            else:
                # Simple alignment: assume the character audio file matches the timeline
                # (this might not be what the user wants if they upload JUST the dialogue)
                # But the user said: "sa incarce audio doar dialogul acelui personaj"
                # If they upload just dialogue, we might need more complex matching
                # even without whisper, but whisper is the requested way.
                pass

            if aligned_for_reconstruction:
                # Build the full track for this character
                full_track = alignment_utils.reconstruct_character_track(aligned_for_reconstruction, sr)

                output_filename = f"track_{secure_filename(speaker)}.wav"
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                sf.write(output_path, full_track, sr)

                output_tracks.append({
                    'speaker': speaker,
                    'file': output_filename
                })

        return jsonify({
            'success': True,
            'message': 'Alignment completed successfully',
            'output_files': output_tracks
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        # Cleanup temp files after sending response
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

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