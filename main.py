import torch
from model.Trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train("text")
    # trainer.train("audio")
    # trainer.train("vision")
