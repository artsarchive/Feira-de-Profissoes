# test_elevenlabs.py
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import os

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

try:
    # Inicializa o cliente
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    
    # Gera o áudio
    response = client.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        output_format="mp3_44100_128",
        text="Olá, isso é um teste da API ElevenLabs",
        model_id="eleven_multilingual_v2",
        voice_settings={
            "stability": 0.7,
            "similarity_boost": 0.8
        }
    )
    
    # Salva o áudio
    with open("test_audio.mp3", "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)
    
    print("Áudio gerado com sucesso! Arquivo salvo como test_audio.mp3")
    
except Exception as e:
    print(f"Erro: {e}")