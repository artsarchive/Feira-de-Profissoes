from flask import Flask, request, send_file
from flask_cors import CORS
import requests
import io
import time
import os
from groq import Groq
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configurações das APIs, crie o arquivo .env com suas keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

# Clientes das APIs
groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-pro')
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

@app.route('/process_audio', methods=['POST'])
def process_audio():
    # Verifica se o arquivo de áudio foi enviado
    if 'audio' not in request.files:
        return {"error": "No audio file provided"}, 400
    
    audio_file = request.files['audio']
    
    # Salva temporariamente o áudio
    audio_path = f"temp_audio_{int(time.time())}.wav"
    audio_file.save(audio_path)
    
    try:
        # Transcreve o áudio com Groq
        transcript = transcribe_audio(audio_path)
        print(f"Transcrição: {transcript}")
        
        # Processa o texto com Gemini
        processed_text = process_text(transcript)
        print(f"Texto processado: {processed_text}")
        
        # Gera áudio com ElevenLabs
        audio_response = generate_audio(processed_text)
        
        # Salva o áudio gerado
        output_path = f"output_audio_{int(time.time())}.mp3"
        with open(output_path, "wb") as f:
            for chunk in audio_response:
                if chunk:
                    f.write(chunk)
        
        # Retorna o áudio gerado
        return send_file(output_path, as_attachment=True, download_name="response.mp3")
    
    except Exception as e:
        print(f"Erro: {str(e)}")
        return {"error": str(e)}, 500
    
    # finally:
    #     # Limpa os arquivos temporários
    #     if os.path.exists(audio_path):
    #         os.remove(audio_path)
    #     if 'output_path' in locals() and os.path.exists(output_path):
    #         os.remove(output_path)

def transcribe_audio(audio_path):
    with open(audio_path, 'rb') as file:
        transcription = groq_client.audio.transcriptions.create(
            file=(audio_path, file.read()),
            model="whisper-large-v3",
            response_format="text"
        )
    return transcription

def process_text(text):
    response = gemini_model.generate_content(
        f"Responda de forma natural e conversacional, você se Chama Ada e é uma veterana do curso de Ciência da Computação da Universidade Estadual de Santa Cruz (UESC), está no 5º período e estagia no BTG, seja amigavel com os calouros, segue o texto que te mandaram, responda em ate 50 palavras: {text}"
    )
    return response.text

def generate_audio(text):
    audio = elevenlabs_client.text_to_speech.convert(
        text=text,
        voice_id=ELEVENLABS_VOICE_ID,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    )
    return audio

if __name__ == '__main__':
    app.run(debug=True, port=5000)