import os
import numpy as np
import pandas as pd
from PIL import Image
import _pickle as pk
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor

from config import MOSI_config

class MOSI_Dataset(Dataset):
    def __init__(self, mode, config):
        super().__init__()
        #加载对应数据集train；test；valid
        self.mosi_data, self.unaligned_data = self._processed_data(
            config.Path.data_path, 
            config.Path.unaligned_data_path, 
            mode
        )


        self.video_ids = self.mosi_data["video_id"]
        self.clip_ids = self.mosi_data["clip_id"]
        self.labels = self.mosi_data["label"]
        self.annotations = [config.train_param.annotation_cls[i] for i in self.mosi_data["annotation"]]


        #加载文本和文本特征提取器
        self.texts = self.mosi_data["text"]
        self.texts = [text[0].capitalize() + text[1:].lower() for text in self.texts]
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(config.Text.BERT)

        self.visions = self.unaligned_data["vision"]
        self.vision_masks =  self.unaligned_data["vision_padding_mask"]

        # # #加载人脸和图片特征提取器
        # self.mp4_file_paths = []
        # for i in range(0, len(self.video_ids)):
        #     imgs_path = os.path.join("data", "MOSI", "Raw", str(self.video_ids[i]), str(self.clip_ids[i])+".mp4")
        #     self.mp4_file_paths.append(imgs_path)
        

        #加载音频和音频特征提取器
        self.audio_file_paths = []
        for i in range(0, len(self.video_ids)):
            audio_path = os.path.join("data", "MOSI", "wav", str(self.video_ids[i]), str(self.clip_ids[i]) + ".wav")
            self.audio_file_paths.append(audio_path)

        self.audio_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)   


    def __getitem__(self, index):
        label = self.labels[index]
        annotation = self.annotations[index]

        text = self.texts[index]
        tokenized_text = self._text(text)

        vision = self.visions[index]
        vision_mask = self.vision_masks[index]


        audio = self.audio_file_paths[index]
        audio_feature, audio_mask = self._audio(audio)


           
        return { # text
                "texts_tokens": torch.tensor(tokenized_text["input_ids"], dtype=torch.long),
                "text_masks": torch.tensor(tokenized_text["attention_mask"], dtype=torch.long),

                # # image
                "visions": torch.tensor(vision, dtype=torch.float32),
                "vision_masks": vision_mask,

                # audio
                "audio_features": audio_feature,
                "audio_masks": audio_mask,

                 # labels
                "labels": torch.tensor(label, dtype=torch.float32),

                 # annotations
                "annotations": torch.tensor(annotation)
                } 
    
    def _processed_data(self, raw_data_path, unaligned_data_path, mode):
        mosi_data = pd.read_csv(raw_data_path)
        self.mosi_data = mosi_data[mosi_data["mode"] == mode].reset_index()

        with open(unaligned_data_path, 'rb') as f:
            self.unaligned_data = pk.load(f)[mode]
        self.unaligned_data = {
            key: self.unaligned_data[key] 
            for key in ['vision', 'vision_lengths', 'id']
        }

        processed_data = []
        for full_id in self.unaligned_data["id"]:
            # Split the ID into the video ID and clip ID
            video_id, clip_id = full_id.split("$_$")
            processed_data.append({"video_id": video_id, "clip_id": int(clip_id)})

        # Update self.data with the processed attributes
        self.unaligned_data["video_id"] = [item["video_id"] for item in processed_data]
        self.unaligned_data["clip_id"] = [item["clip_id"] for item in processed_data]

        self._check_data_consistency(self.mosi_data, self.unaligned_data)

        vision_tmp = torch.sum(torch.tensor(self.unaligned_data['vision']), dim=-1)
        vision_mask = (vision_tmp == 0)

        for i in range(len(self.unaligned_data["vision"])):
            vision_mask[i][0] = False
        vision_mask = torch.cat((vision_mask[:, 0:1], vision_mask), dim=-1)

        self.unaligned_data['vision_padding_mask'] = vision_mask
        return self.mosi_data, self.unaligned_data


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
    
    def _check_data_consistency(self, mosi_data, unaligned_data):
        """
        检查 self.mosi_data 和 self.unaligned_data 的 video_id 和 clip_id 是否对应。
        
        Args:
            mosi_data: 包含 'video_id' 和 'clip_id' 的 pandas DataFrame。
            unaligned_data: 包含 'video_id' 和 'clip_id' 的列表或字典形式。

        Returns:
            bool: 如果所有的 video_id 和 clip_id 都对应，返回 True，否则返回 False。
        """
        # 提取 mosi_data 的 video_id 和 clip_id
        mosi_video_ids = mosi_data["video_id"].tolist()
        mosi_clip_ids = mosi_data["clip_id"].tolist()

        # 提取 unaligned_data 的 video_id 和 clip_id
        unaligned_video_ids = unaligned_data["video_id"]
        unaligned_clip_ids = unaligned_data["clip_id"]

        # 检查长度是否一致
        if len(mosi_video_ids) != len(unaligned_video_ids):
            print("Mismatch in number of entries!")
            return False

        # 检查每个 video_id 和 clip_id 是否对应
        for idx, (mosi_vid, mosi_clip) in enumerate(zip(mosi_video_ids, mosi_clip_ids)):
            if mosi_vid != unaligned_video_ids[idx] or mosi_clip != unaligned_clip_ids[idx]:
                print(f"Mismatch at index {idx}:")
                print(f"MOSI - video_id: {mosi_vid}, clip_id: {mosi_clip}")
                print(f"Unaligned - video_id: {unaligned_video_ids[idx]}, clip_id: {unaligned_clip_ids[idx]}")
                return False

        print("All video_id and clip_id pairs are consistent!")
        return True

    def __len__(self):
        return len(self.mosi_data)
    


def get_dataloader(dataset_name):
    if dataset_name == "MOSI":
        train_data = MOSI_Dataset("train", MOSI_config)
        test_data = MOSI_Dataset("test", MOSI_config)
        valid_data = MOSI_Dataset("valid", MOSI_config)

        train_loader = DataLoader(train_data, batch_size=MOSI_config.train_param.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=MOSI_config.train_param.batch_size, shuffle=False)
        valid_loader = DataLoader(valid_data, batch_size=MOSI_config.train_param.batch_size, shuffle=False)

        return train_loader, test_loader, valid_loader
        

if __name__ == "__main__":
    a = MOSI_Dataset("train")
    for batch in a:
        pass
    






        
        


        