from flask import Flask, request, send_file
from flask_cors import CORS
import requests
import io
import time
import os
from groq import Groq
import google.generativeai as genai
from dotenv import load_dotenv

# Bibliotecas novas para usar o áudio clonado
import torch
import torch.serialization
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig

from TTS.tts.models.xtts import XttsAudioConfig 

_original_torch_load = torch.load

def patched_torch_load(f, *args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)

torch.load = patched_torch_load

# torch.load aceita a classe XttsConfig
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig])

# Carregar variáveis de ambiente
load_dotenv()

app = Flask(__name__)
CORS(app)

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
try:
    TTS_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)

    print(f"Modelo Coqui TTS (XTTSv2) carregado com sucesso no dispositivo: {DEVICE}")
except Exception as e: 
    TTS_model = None
    import traceback
    print(f"Aviso: Falha ao carregar o modelo Coqui TTS. Erro {e}")
    traceback.print_exc()
    print("-------------------------")
    print("Coqui TTS Model falhou no carregamento. Confira dependências, a configuração da CUDA e os arquivos do modelo.")
    print("-------------------------")


CLONED_VOICE_REF = "nathalie.wav"

# Configurações das APIs, crie o arquivo .env com suas keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Clientes das APIs
groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-pro')

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
        
        # Gera áudio com Coqui TTS local
        output_path = generate_audio(processed_text)

        return send_file(output_path, as_attachment=True, download_name="response.wav")
    
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
    if not TTS_model: 
        raise Exception("Modelo Coqui TTS não carregado.")
    
    output_path = f"output_coqui_{int(time.time())}.wav"

    TTS_model.tts_to_file(
        text=text,
        file_path=output_path,
        speaker_wav=CLONED_VOICE_REF,
        language="pt"
    )

    return output_path

if __name__ == '__main__':
    app.run(debug=True, port=5000)