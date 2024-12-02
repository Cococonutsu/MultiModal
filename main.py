import torch
from model.Trainer import Trainer

from config import MOSI_config

if __name__ == "__main__":
    text_trainer = Trainer("text", MOSI_config)
    audio_trainer = Trainer("audio", MOSI_config)
    vision_trainer = Trainer("vision", MOSI_config)

    text_trainer.train()
    audio_trainer.train()
    vision_trainer.train()
