import sounddevice as sd
import numpy as np
from openai import OpenAI
from pydub import AudioSegment
import io
from llm import ask_llama

client = OpenAI(
    base_url="http://localhost:8880/v1", api_key="not-needed"
)

SAMPLE_RATE = 24000  # Model's audio rate

def stream_and_play_live(input_text):
    with client.audio.speech.with_streaming_response.create(
        model="kokoro",
        voice="af_sarah",
        input=input_text
    ) as response:
        for chunk in response.iter_bytes():
            if not chunk:
                continue
            try:
                # Decode MP3 chunk
                audio = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
                samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
                # Play with sounddevice
                sd.play(samples, samplerate=audio.frame_rate, blocking=True)
            except Exception as e:
                print("Chunk decode error:", e)

while True:
    user_input = input("You: ")
    llama_answer = ask_llama(user_input)
    print(llama_answer)
    if user_input.lower() in ["exit", "quit"]:
        break
    stream_and_play_live(llama_answer)
