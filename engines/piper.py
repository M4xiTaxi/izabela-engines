import io

from engines.base import BaseEngine
from models import Voice
from dataclasses import asdict

from urllib.request import urlopen
from piper import download_voices
from piper import PiperVoice
from piper import SynthesisConfig
from pathlib import Path

import pydub
import json

import logging
logging.basicConfig(level=logging.INFO)

class PiperEngine(BaseEngine):
    def __init__(self):
        self.loaded_voice = ""
        self.piper_voice = None
        self.synth_config = SynthesisConfig()

        logging.info("Loading piper model...")


    def list_voices(self) -> list[Voice]:
        voices_list = []
        json_voices = None

        logging.debug("Downloading voices.json file: '%s'", download_voices.VOICES_JSON)
        with urlopen(download_voices.VOICES_JSON) as response:
            json_voices = json.load(response)

        for voice in json_voices:
            voices_list.append(Voice(
                id=voice,
                name=f"{json_voices[voice]["name"]} {json_voices[voice]["language"]["code"]} - {json_voices[voice]["quality"]}",
                category=self.__class__.__name__,
                languageCode=f"{json_voices[voice]["language"]["code"]}"
            ))

        return voices_list
    
    def _load_voice(self, voice: Voice):
        logging.debug(f"Voice not loaded: {voice.id}")
        download_path = Path(Path.cwd() / "models")

        if download_path.exists() is False:
            download_path.mkdir()

        model_path = download_path / f"{voice.id}.onnx"
        
        if model_path.exists() is False:
            logging.debug(f"Downloading model: {voice.id}")
            download_voices.download_voice(voice.id, download_path)
        elif model_path.stat().st_size == 0:
            logging.debug(f"Downloading model: {voice.id}")
            download_voices.download_voice(voice.id, download_path)
        else:
            logging.debug(f"Model is downloaded: {voice.id}")

        self.piper_voice = PiperVoice.load(model_path)

        config_path = download_path / f"{voice.id}_config.json"
        
        if config_path.exists():
            logging.debug(f"Loading synthesis config for {voice.id}")
            with open(config_path, 'r') as f:
                self.synth_config = SynthesisConfig(**json.load(f))
        else:
            logging.debug(f"Creating synthesis config for {voice.id}")
            self.synth_config = SynthesisConfig()
            with open(config_path, 'w') as f:
                json.dump(asdict(self.synth_config), f)
        
        logging.debug(self.synth_config)
                                
        logging.debug(f"Voice loaded: {voice.id}")

    def synthesize_voice(self, voice: Voice, text: str) -> bytes:
        if self.loaded_voice != voice.id:
            self._load_voice(voice)
            self.loaded_voice = voice.id

        logging.debug(f"Synthesizing: {text}")
        audio_chunks = self.piper_voice.synthesize(text, syn_config=self.synth_config)
        audio_stream = pydub.AudioSegment.empty()

        for chunk in audio_chunks:
            sound = pydub.AudioSegment(
                data=chunk.audio_int16_bytes, 
                sample_width=chunk.sample_width,
                frame_rate=chunk.sample_rate,
                channels=chunk.sample_channels
            )

            audio_stream += sound

        mp3_bytes = io.BytesIO()

        audio_stream.export(mp3_bytes, format="mp3")

        return mp3_bytes.getvalue()
