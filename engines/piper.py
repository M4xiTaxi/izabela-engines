import io

from engines.base import BaseEngine
from models import Voice

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
        # self.synth_config = SynthesisConfig(
        #     volume=0.5,  # half as loud
        #     length_scale=1.0,
        #     noise_scale=1.0,  # more audio variation
        #     noise_w_scale=1.0,  # more speaking variation
        #     normalize_audio=False, # use raw audio from voice
        # )

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

    def synthesize_voice(self, voice: Voice, text: str) -> bytes:
        if self.loaded_voice != voice.id:
            logging.debug(f"Voice not loaded: {voice.id}")
            voices_path = Path(Path.cwd() / "voices")

            if voices_path.exists() is False:
                voices_path.mkdir()
            
            download_voices.download_voice(voice.id, voices_path)
            
            self.loaded_voice = voice.id
            self.piper_voice = PiperVoice.load(voices_path / f"{self.loaded_voice}.onnx")
            logging.debug(f"Voice loaded: {voice.id}")

        logging.debug(f"Synthesizing: {text}")
        # TODO: Make this use a config per voice which Liv can edit
        audio_chunks = self.piper_voice.synthesize(text) #, syn_config=self.synth_config)
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
