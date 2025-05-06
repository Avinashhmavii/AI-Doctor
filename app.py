from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import base64
import requests
import json
from gtts import gTTS
import speech_recognition as sr
from tempfile import NamedTemporaryFile
import re
from pydub import AudioSegment
import logging
from PIL import Image
import io
import time

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Validate API key
if not OPENROUTER_API_KEY:
    logger.error("OPENROUTER_API_KEY is not set in environment variables")
    raise ValueError("OpenRouter API key is missing. Set OPENROUTER_API_KEY in .env or environment variables.")

# Global state (simulating session state)
state = {
    "diagnosis": None,
    "response_count": 0,
    "encoded_image": None
}

# Function to resize and encode image to base64
def encode_image(image):
    try:
        # Open image with PIL
        img = Image.open(image)
        # Resize image to max 800x800 while maintaining aspect ratio
        img.thumbnail((800, 800), Image.Resampling.LANCZOS)
        # Convert to JPEG
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        # Encode to base64
        encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        logger.info(f"Encoded image size: {len(encoded)} bytes")
        return encoded
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        return None

# Function to extract diagnosis from the analysis result
def extract_diagnosis(analysis_result):
    try:
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
    except Exception as e:
        logger.error(f"Error extracting diagnosis: {str(e)}")
        return "a medical condition based on the image analysis"

# Function to analyze image and voice input together
def analyze_image_and_voice(user_query, encoded_image, is_initial_analysis=True):
    try:
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

        # Prepare the payload for OpenRouter API
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ai-doctor-me7g.onrender.com",
            "X-Title": "Medical Image Analysis App"
        }
        payload = {
            "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_query},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                        }
                    ]
                }
            ],
            "max_tokens": 500
        }
        start_time = time.time()
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=60  # Increased timeout to 60 seconds
        )
        response_time = time.time() - start_time
        logger.info(f"OpenRouter API request took {response_time:.2f} seconds")

        # Check Content-Type before parsing
        content_type = response.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            logger.error(f"Non-JSON response received: Content-Type={content_type}, Status={response.status_code}, Response={response.text[:500]}")
            return f"Error: Non-JSON response from OpenRouter API (Status: {response.status_code})"

        response_json = response.json()

        # Log the raw response
        logger.info(f"OpenRouter API response: {response_json}")

        # Handle common HTTP status codes
        if response.status_code == 401:
            logger.error("Unauthorized: Invalid or missing API key")
            return "Error: Invalid OpenRouter API key"
        elif response.status_code == 429:
            logger.error("Rate limit exceeded")
            return "Error: OpenRouter API rate limit exceeded. Please try again later."
        elif response.status_code != 200:
            logger.error(f"Unexpected status code: {response.status_code}, Response: {response_json}")
            return f"Error: OpenRouter API returned status {response.status_code}"

        # Validate response structure
        if "choices" not in response_json or not isinstance(response_json["choices"], list) or not response_json["choices"]:
            logger.error(f"Invalid response structure: {response_json}")
            return "Error: Invalid response from OpenRouter API (no valid choices found)"

        if "message" not in response_json["choices"][0] or "content" not in response_json["choices"][0]["message"]:
            logger.error(f"Missing message or content in response: {response_json}")
            return "Error: Invalid response from OpenRouter API (missing message or content)"

        # Extract the response text
        response_text = response_json["choices"][0]["message"]["content"]
        if is_initial_analysis:
            state["diagnosis"] = extract_diagnosis(response_text)
        return response_text
    except requests.exceptions.Timeout:
        logger.error("Request to OpenRouter API timed out")
        return "Error: OpenRouter API request timed out. Please try again later."
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error in OpenRouter API request: {str(e)} - Response: {response.text[:500] if 'response' in locals() else 'No response'}")
        return f"Error: HTTP error from OpenRouter API - {str(e)}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error in OpenRouter API request: {str(e)}")
        return f"Error: Failed to connect to OpenRouter API - {str(e)}"
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)} - Response: {response.text[:500] if 'response' in locals() else 'No response'}")
        return f"Error: Invalid JSON response from OpenRouter API - {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in analyze_image_and_voice: {str(e)}")
        return f"Error analyzing image and query: {str(e)}"

# Function to generate AI response for text-only queries
def generate_ai_response(user_query):
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ai-doctor-me7g.onrender.com",
            "X-Title": "Medical Image Analysis App"
        }
        payload = {
            "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
            "messages": [
                {"role": "user", "content": user_query}
            ],
            "max_tokens": 500
        }
        start_time = time.time()
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=60  # Increased timeout to 60 seconds
        )
        response_time = time.time() - start_time
        logger.info(f"OpenRouter API request took {response_time:.2f} seconds")

        # Check Content-Type before parsing
        content_type = response.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            logger.error(f"Non-JSON response received: Content-Type={content_type}, Status={response.status_code}, Response={response.text[:500]}")
            return f"Error: Non-JSON response from OpenRouter API (Status: {response.status_code})"

        response_json = response.json()

        # Log the raw response
        logger.info(f"OpenRouter API response: {response_json}")

        # Handle common HTTP status codes
        if response.status_code == 401:
            logger.error("Unauthorized: Invalid or missing API key")
            return "Error: Invalid OpenRouter API key"
        elif response.status_code == 429:
            logger.error("Rate limit exceeded")
            return "Error: OpenRouter API rate limit exceeded. Please try again later."
        elif response.status_code != 200:
            logger.error(f"Unexpected status code: {response.status_code}, Response: {response_json}")
            return f"Error: OpenRouter API returned status {response.status_code}"

        # Validate response structure
        if "choices" not in response_json or not isinstance(response_json["choices"], list) or not response_json["choices"]:
            logger.error(f"Invalid response structure: {response_json}")
            return "Error: Invalid response from OpenRouter API (no valid choices found)"

        if "message" not in response_json["choices"][0] or "content" not in response_json["choices"][0]["message"]:
            logger.error(f"Missing message or content in response: {response_json}")
            return "Error: Invalid response from OpenRouter API (missing message or content)"

        return response_json["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        logger.error("Request to OpenRouter API timed out")
        return "Error: OpenRouter API request timed out. Please try again later."
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error in OpenRouter API request: {str(e)} - Response: {response.text[:500] if 'response' in locals() else 'No response'}")
        return f"Error: HTTP error from OpenRouter API - {str(e)}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error in OpenRouter API request: {str(e)}")
        return f"Error: Failed to connect to OpenRouter API - {str(e)}"
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)} - Response: {response.text[:500] if 'response' in locals() else 'No response'}")
        return f"Error: Invalid JSON response from OpenRouter API - {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in generate_ai_response: {str(e)}")
        return f"Error generating response: {str(e)}"

# Function to clean text for speech
def clean_text_for_speech(text):
    try:
        cleaned_text = re.sub(r'[,\;\*\(\)\[\]\{\}!?@#$%^&+=_"\'`~|]', '', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text
    except Exception as e:
        logger.error(f"Error cleaning text for speech: {str(e)}")
        return text

# Function to convert text to speech
def text_to_speech(input_text):
    if not input_text or not isinstance(input_text, str):
        logger.warning("Invalid input text for text-to-speech")
        return None
    cleaned_text = clean_text_for_speech(input_text)
    if not cleaned_text:
        logger.warning("Cleaned text is empty for text-to-speech")
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
    except Exception as e:
        logger.error(f"Error in text_to_speech: {str(e)}")
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
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return None
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}")
        return jsonify({"error": "Failed to load index page"}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    response = {}
    state["response_count"] += 1

    try:
        # Handle image upload
        if 'image' in request.files and request.files['image'].filename != '':
            image = request.files['image']
            state["encoded_image"] = encode_image(image)
            if not state["encoded_image"]:
                return jsonify({"error": "Failed to encode image"}), 400
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
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    # Bind to 0.0.0.0 and use PORT from environment (default 10000 on Render)
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
