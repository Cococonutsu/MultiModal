import torch

class MOSI_config:

    class train_param:
        DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

        annotation_cls = {
            "Positive" : 0,
            "Neutral" : 1,
            "Negative" : 2
        }



    class Path:
        data_path = "./data/MOSI/label.csv"
        unaligned_data_path = 'data/MOSI/Processed/unaligned_50.pkl'

    class Text:
        lr = 1e-4
        weight_deacy = 1e-3
        amsgrad = True
        num_warm_up = 5

        epochs = 100
        batch_size = 128

        ROBERTA = "roberta-large"
        BERT = "bert-base-uncased"
        # BERT = "roberta-base"

    class Vision:
        lr = 1e-4
        weight_deacy = 1e-3
        amsgrad = True
        num_warm_up = 5

        epochs = 100
        batch_size = 128


    class Audio:
        lr = 1e-4
        weight_deacy = 1e-3
        amsgrad = True
        num_warm_up = 5

        epochs = 50
        batch_size = 32
        DATA2VEC = "facebook/data2vec-audio-base"



    