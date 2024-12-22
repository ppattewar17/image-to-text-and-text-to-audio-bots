import os
import string
from flask import json
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
#from dotenv import find_dotenv, load_dotenv
from gtts import gTTS
from transformers import pipeline
import language_tool_python


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['AUDIO_FOLDER'] = 'audio'

# Create the audio folder if it doesn't exist
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

def correct_grammar(text):
    # Initialize the language tool
    tool = language_tool_python.LanguageTool('en-US')

    # Perform grammar check
    matches = tool.check(text)

    # Apply corrections
    corrected_text = tool.correct(text)

    return corrected_text

def image2text(image_path, conditional_keyword):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    raw_image = Image.open(image_path).convert('RGB')

    # Conditional image captioning
    text = f"{conditional_keyword}" if conditional_keyword else "The"
    inputs = processor(raw_image, text, return_tensors="pt")
    min_tokens = 30
    # max_new_tokens = 3000
    out = model.generate(**inputs, min_length=min_tokens, max_length=min_tokens+10)
    conditional_description = processor.decode(out[0], skip_special_tokens=True)

    # Unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs, min_length=min_tokens, max_length=min_tokens+10)
    unconditional_description = processor.decode(out[0], skip_special_tokens=True)

    # return conditional_description.capitalize(), unconditional_description.capitalize()
    # Apply grammar correction
    corrected_conditional_description = correct_grammar(conditional_description.capitalize())
    corrected_unconditional_description = correct_grammar(unconditional_description.capitalize())

    return corrected_conditional_description, corrected_unconditional_description


def text2speech(text, filename):
    audio_path = os.path.join(app.config['AUDIO_FOLDER'], filename)
    tts = gTTS(text)
    tts.save(audio_path)
    return audio_path






############################################################################################


@app.route('/', methods=['GET', 'POST'])
def startPoint():
    return render_template('bootstrap_website.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if 'image' and 'conditional_keyword' are in request.files and request.form respectively
        if 'image' not in request.files or 'conditional_keyword' not in request.form:
            return jsonify({"conditional_audio_path": None, "unconditional_audio_path": None, 
                            "conditional_description": "No image or keyword uploaded", "unconditional_description": "No image uploaded"})

        image = request.files['image']
        conditional_keyword = request.form['conditional_keyword']

        if image.filename == '':
            return jsonify({"conditional_audio_path": None, "unconditional_audio_path": None, 
                            "conditional_description": "No selected image", "unconditional_description": "No selected image"})

        if image:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.png')
            image.save(image_path)
            conditional_description, unconditional_description = image2text(image_path, conditional_keyword)

            # Convert text to audio for both conditional and unconditional descriptions
            conditional_audio_path = text2speech(conditional_description, 'conditional_audio.mp3')
            unconditional_audio_path = text2speech(unconditional_description, 'unconditional_audio.mp3')

            response_data = {
                "conditional_description": conditional_description,
                "unconditional_description": unconditional_description,
                "conditional_audio_path": conditional_audio_path,
                "unconditional_audio_path": unconditional_audio_path,
            }

            return jsonify(response_data)

    return render_template('index.html')

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(app.config['AUDIO_FOLDER'], filename)

@app.route('/text_to_audio', methods=['GET', 'POST'])
def text_to_audio():
    if request.method == 'POST':
        text = request.form['text']
        tts = gTTS(text)
        tts.save('output.mp3')
        return send_file('output.mp3', as_attachment=True)
    return render_template('text_to_audio.html')




if __name__ == '__main__':
    app.run(debug=True)
