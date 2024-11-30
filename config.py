import torch

class MOSI_config:

    class train_param:
        epochs = 20
        batch_size = 16
        DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
        lr = 1e-4
        num_warm_up = 5
        annotation_cls = {
            "Positive" : 0,
            "Neutral" : 1,
            "Negative" : 2
        }



    class Path:
        data_path = "./data/MOSI/label.csv"
        unaligned_data_path = 'data/MOSI/Processed/unaligned_50.pkl'

    class Text:
        ROBERTA = "roberta-large"
        BERT = "bert-base-uncased"
        BERT = "roberta-base"

    class Video:
        VIT = "google/vit-base-patch16-224-in21k"
        XCLIP = "microsoft/xclip-base-patch32"
        VIDEOMAE = "MCG-NJU/videomae-base"

    class Audio:
        DATA2VEC = "facebook/data2vec-audio-base"


    