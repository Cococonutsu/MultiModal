import torch
from model.T_trainer import Text_trainer
from model.A_trainer import Audio_trainer
from model.V_trainer import Vision_trainer

if __name__ == "__main__":
    # text_train = Text_trainer()
    # text_train.train()

    audio_train = Audio_trainer()
    audio_train.train()

    # vision_train = Vision_trainer()
    # vision_train.train()
