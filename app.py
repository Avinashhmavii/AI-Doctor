from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import base64
from huggingface_hub import InferenceClient
from gtts import gTTS
import speech_recognition as sr
from tempfile import NamedTemporaryFile
import re
from pydub import AudioSegment

app = Flask(__name__)

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY", "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")

# Initialize Hugging Face Inference client
hf_client = InferenceClient(model="unsloth/Llama-3.2-11B-Vision", token=HF_API_KEY)

# Global state (simulating session state)
state = {
    "diagnosis": None,
    "response_count": 0,
    "encoded_image": None
}

# Function to encode image to base64
def encode_image(image):
    return base64.b64encode(image.read()).decode('utf-8')

# Function to extract diagnosis from the analysis result
def extract_diagnosis(analysis_result):
    lines = analysis_result.split('\n')
    for line in lines:
        if "Primary Diagnosis" in line or "Diagnosis:" in line or "Diagnostic Assessment" in line:
            diagnosis = line.split(":", 1)[1].strip() if ":" in line else lines[lines.index(line) + 1].strip()
            diagnosis = re.sub(r'\(.*?confidence\)', '', diagnosis).strip()
            return diagnosis
    for i, line in enumerate(lines):
        if "Diagnostic" in line or "Key Findings" in line:
            for j in range(i + 1, len(lines)):
                if lines[j].strip() and not lines[j].startswith("- Alternative"):
                    return lines[j].strip()
    return "a medical condition based on the image analysis"

# Function to analyze image and voice input together
def analyze_image_and_voice(user_query, encoded_image, is_initial_analysis=True):
    if is_initial_analysis:
        medical_query = """
You are a highly skilled medical imaging expert. Analyze the uploaded medical image as follows:

### Image Type & Region
- Specify the type of image (e.g., CT scan, X-ray, MRI)
- Identify the anatomical region (e.g., chest, brain) and view (e.g., axial)
- Comment on image quality

### Key Findings
- List primary observations (e.g., normal lungs, no masses)
- Note any abnormalities with details (e.g., size, location)
- Rate severity: Normal/Mild/Moderate/Severe

### Diagnostic Assessment
- Provide a primary diagnosis with confidence level (e.g., 95% confidence)
- List differential diagnoses if applicable
- Support with evidence from the image

### Patient-Friendly Explanation
- Explain findings in simple language
- Address common patient concerns (e.g., "What should I do next?")

Format your response with markdown headers and bullet points.
"""
        full_query = medical_query + f"\n\nUser's additional query: {user_query}" if user_query else medical_query
    else:
        diagnosis = state["diagnosis"] if state["diagnosis"] else "a medical condition based on the image analysis"
        full_query = f"""
You are an AI doctor. The patient has been diagnosed with {diagnosis}. The patient has asked: "{user_query}". Provide a concise, direct answer in simple language, referencing the image findings if relevant. Offer general treatment advice confidently as a doctor would.
"""

    # Prepare the payload for Hugging Face Inference API
    response = hf_client.chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_query},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ],
            }
        ],
        max_tokens=500
    )
    response_text = response.choices[0].message.content
    if is_initial_analysis:
        state["diagnosis"] = extract_diagnosis(response_text)
    return response_text

# Function to generate AI response for text-only queries
def generate_ai_response(user_query):
    response = hf_client.chat_completion(
        messages=[{"role": "user", "content": user_query}],
        max_tokens=500
    )
    return response.choices[0].message.content

# Function to clean text for speech
def clean_text_for_speech(text):
    cleaned_text = re.sub(r'[,\;\*\(\)\[\]\{\}!?@#$%^&+=_"\'`~|]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# Function to convert text to speech
def text_to_speech(input_text):
    if not input_text or not isinstance(input_text, str):
        return None
    cleaned_text = clean_text_for_speech(input_text)
    if not cleaned_text:
        return None
    try:
        tts = gTTS(text=cleaned_text, lang='en')
        with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            tts.save(temp_file.name)
            with open(temp_file.name, "rb") as audio_file:
                audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            os.unlink(temp_file.name)
            return audio_base64
    except Exception:
        return None

# Function to transcribe uploaded audio
def transcribe_uploaded_audio(audio_file):
    temp_wav_path = None
    try:
        audio_segment = AudioSegment.from_file(audio_file)
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            temp_wav_path = temp_wav.name
            audio_segment.export(temp_wav_path, format="wav")
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_wav_path) as source:
                audio_data = recognizer.record(source)
                transcribed_text = recognizer.recognize_google(audio_data)
                os.unlink(temp_wav_path)
                return transcribed_text
    except Exception:
        return None
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    response = {}
    state["response_count"] += 1

    # Handle image upload
    if 'image' in request.files and request.files['image'].filename != '':
        image = request.files['image']
        state["encoded_image"] = encode_image(image)
        initial_query = "Describe the condition in this image."
        analysis_result = analyze_image_and_voice(initial_query, state["encoded_image"], is_initial_analysis=True)
        response["analysis"] = analysis_result
        response["audio"] = text_to_speech(analysis_result)

    # Handle text question
    text_input = request.form.get('text_input')
    if text_input:
        if state["encoded_image"]:
            ai_response = analyze_image_and_voice(text_input, state["encoded_image"], is_initial_analysis=False)
        else:
            ai_response = generate_ai_response(text_input)
        response["text_response"] = ai_response
        response["audio"] = text_to_speech(ai_response)

    # Handle audio upload
    if 'audio' in request.files and request.files['audio'].filename != '':
        audio_file = request.files['audio']
        transcribed_text = transcribe_uploaded_audio(audio_file)
        if transcribed_text:
            response["transcription"] = transcribed_text
            if state["encoded_image"]:
                ai_response = analyze_image_and_voice(transcribed_text, state["encoded_image"], is_initial_analysis=False)
            else:
                ai_response = generate_ai_response(transcribed_text)
            response["audio_response"] = ai_response
            response["audio"] = text_to_speech(ai_response)

    return jsonify(response)

if __name__ == '__main__':
    # Bind to 0.0.0.0 and use PORT from environment (default 10000 on Render)
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
