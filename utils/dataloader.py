import os
import numpy as np
import pandas as pd
from PIL import Image
import _pickle as pk
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoImageProcessor, ViTModel, Wav2Vec2FeatureExtractor

from config import MOSI_config

class MOSI_Dataset(Dataset):
    def __init__(self, mode, config):
        super().__init__()
        #加载对应数据集train；test；valid
        mosi_data = pd.read_csv(config.Path.data_path)
        self.mosi_data = mosi_data[mosi_data["mode"] == mode].reset_index()

        self.video_ids = self.mosi_data["video_id"]
        self.clip_ids = self.mosi_data["clip_id"]
        self.labels = self.mosi_data["label"]


        #加载文本和文本特征提取器
        self.texts = self.mosi_data["text"]
        self.texts = [text[0].capitalize() + text[1:].lower() for text in self.texts]
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(config.Text.BERT)

        # #加载人脸和图片特征提取器
        self.mp4_file_paths = []
        for i in range(0, len(self.video_ids)):
            imgs_path = os.path.join("data", "MOSI", "Raw", str(self.video_ids[i]), str(self.clip_ids[i])+".mp4")
            self.mp4_file_paths.append(imgs_path)
        

        #加载音频和音频特征提取器
        self.audio_file_paths = []
        for i in range(0, len(self.video_ids)):
            audio_path = os.path.join("data", "MOSI", "wav", str(self.video_ids[i]), str(self.clip_ids[i]) + ".wav")
            self.audio_file_paths.append(audio_path)

        self.audio_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)   


    def __getitem__(self, index):
        label = self.labels[index]

        text = self.texts[index]
        tokenized_text = self._text(text)

        mp4_file_path = self.mp4_file_paths[index]


        audio = self.audio_file_paths[index]
        audio_feature, audio_mask = self._audio(audio)


           
        return { # text
                "texts_tokens": torch.tensor(tokenized_text["input_ids"], dtype=torch.long),
                "text_masks": torch.tensor(tokenized_text["attention_mask"], dtype=torch.long),

                # # image
                "mp4_file_paths": mp4_file_path,

                # audio
                "audio_features": audio_feature,
                "audio_masks": audio_mask,

                 # labels
                "labels": torch.tensor(label, dtype=torch.float32)
                } 


    def _text(self, text):
        tokenized_text = self.roberta_tokenizer(
            text, 
            max_length=96, 
            padding= "max_length", 
            truncation=True,
            return_attention_mask = True    
        )
        return tokenized_text
        


    def _audio(self, audio):
        sound,_ = torchaudio.load(audio)
        soundData = torch.mean(sound, dim=0, keepdim=False)
        features = self.audio_extractor(soundData, sampling_rate=16000, max_length=96000,return_attention_mask=True,truncation=True, padding="max_length")
        audio_features = torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze()
        audio_masks = torch.tensor(np.array(features['attention_mask']), dtype=torch.long).squeeze()
        return audio_features, audio_masks

    def __len__(self):
        return len(self.mosi_data)
    
# def collate_fn(batch):
#     images_list = [item["images"] for item in batch]
#     max_length = max(len(sublist) for sublist in images_list)
#     for i in range(len(images_list)):
#         if len(images_list[i]) < max_length:
#             images_list[i].extend([None] * (max_length - len(images_list[i])))

#     texts = [item["texts"] for item in batch]
#     audio_features = [torch.tensor(item["audio_features"], dtype=torch.long) for item in batch]
#     audio_masks = [torch.tensor(item["audio_masks"], dtype=torch.long) for item in batch]
#     labels = [torch.tensor(item["labels"], dtype=torch.float32) for item in batch]

#     return {
#         "texts": texts,
#         "images": images_list,
#         "audio_features": torch.stack(audio_features),
#         "audio_masks": torch.stack(audio_masks),
#         "labels": torch.stack(labels)
#     }



def get_dataloader(dataset_name):
    if dataset_name == "MOSI":
        train_data = MOSI_Dataset("train", MOSI_config)
        test_data = MOSI_Dataset("test", MOSI_config)
        valid_data = MOSI_Dataset("valid", MOSI_config)

        train_loader = DataLoader(train_data, batch_size=MOSI_config.train_param.batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=MOSI_config.train_param.batch_size, shuffle=False)
        valid_loader = DataLoader(valid_data, batch_size=MOSI_config.train_param.batch_size, shuffle=False)

        return train_loader, test_loader, valid_loader
        

if __name__ == "__main__":
    a = MOSI_Dataset("train")
    for batch in a:
        pass
    






        
        


        