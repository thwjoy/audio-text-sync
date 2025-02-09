from flask import Flask, render_template, request, jsonify
from whisper_realtime import process_audio_with_text, load_audio_file
import numpy as np
import io
import os

app = Flask(__name__)

# Update allowed extensions to include mp3
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Check if both files are present
        if 'audio' not in request.files or not request.form.get('text'):
            return jsonify({'error': 'Missing audio file or text'}), 400

        audio_file = request.files['audio']
        text = request.form['text']

        # Validate audio file
        if audio_file.filename == '' or not allowed_file(audio_file.filename):
            return jsonify({'error': 'Invalid or missing audio file. Please upload a WAV or MP3 file.'}), 400

        # Read audio file directly into memory
        audio_bytes = io.BytesIO(audio_file.read())
        audio_data = load_audio_file(audio_bytes)  # Load audio data from the in-memory bytes

        # Process the audio and text
        transcript, processing_time, timing_df = process_audio_with_text(
            audio_data=audio_data,  # Pass audio data directly instead of file path
            text=text
        )

        # Convert DataFrame to dictionary for JSON response
        timing_data = timing_df.to_dict(orient='records')

        response = {
            'success': True,
            'transcript': transcript,
            'processing_time': f"{processing_time:.2f}",
            'timing_data': timing_data
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)