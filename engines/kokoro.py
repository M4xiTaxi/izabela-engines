import io
import soundfile as sf

import torch
import json

from engines.base import BaseEngine
from models import Voice
import logging
from kokoro import KPipeline, KModel
from pathlib import Path

logging.basicConfig(level=logging.INFO)

heart_fallback = {
    "af_heart": {
        "name": "Heart",
        "gender": "Female",
        "lang_code": {
            "espeak": "en-us",
            "misaki": "a"
        }
    }
}

class KokoroEngine(BaseEngine):
    def __init__(self):
        self.loaded_voice = ""
        self.voice_dict = None
        voice_dict_path = Path(Path.cwd() / "resources/kokoro_voices.json")

        if voice_dict_path.exists():
            with open(voice_dict_path, "r") as f:
                self.voice_dict = json.load(f)
        else:
            self.voice_dict = heart_fallback

        logging.info("Loading Kokoro model...")
        # TODO: find out how to stop it from downloading the model every single time
        self._model = KModel(repo_id='hexgrad/Kokoro-82M').eval()
        self._pipeline = None

        # TODO: Add cpu and gpu resolver

        # TODO: figure out how to get triton working on Windows so we can compile models
        # logging.info("Compiling Kokoro model. This might take some time, but will make the model run much faster.")
        # self._model.compile()

    def list_voices(self) -> list[Voice]:
        voice_list = []
        
        for id in self.voice_dict:
            voice_list.append(Voice(
                id=id,
                name=f"{self.voice_dict[id]["name"]} {self.voice_dict[id]["gender"]} - {self.voice_dict[id]["lang_code"]["espeak"]}",
                category=self.__class__.__name__,
                languageCode=self.voice_dict[id]["lang_code"]["espeak"]
            ))

        return voice_list

    def synthesize_voice(self, voice: Voice, text: str) -> bytes:
        if self.loaded_voice != voice.id:
            logging.info(f'Loading pipeline for language {voice.languageCode}...')
            self._pipeline = KPipeline(lang_code=self.voice_dict[voice.id]["lang_code"]["misaki"], model=self._model)
            self.loaded_voice = voice.id

        samples = None

        for result in self._pipeline(text, voice=voice.id):
            if result.audio is not None:
                if samples is None:
                    samples = result.audio
                else:
                    samples = torch.cat((samples, result.audio))
        
        mp3_bytes = io.BytesIO()
        sf.write(file=mp3_bytes, samplerate=24000, data=samples, format='MP3', bitrate_mode='VARIABLE', compression_level=0.25)
        return mp3_bytes.getvalue() 
