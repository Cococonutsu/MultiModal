import av
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from transformers import AutoProcessor, AutoModel
from model.projector import Projector

from utils.dataloader import DataLoader


class Vision_Encoder(nn.Module):
    def __init__(self, config):
        super(Vision_Encoder, self).__init__()
    # 加载 ViCLIP 模型
        self.device = config.train_param.DEVICE
        self.processor = AutoProcessor.from_pretrained(config.Video.VIDEOMAE)
        self.model = AutoModel.from_pretrained(config.Video.VIT)

        self.input_dim = 500
        self.output_dim = 256

        self.projector = Projector(
            input_dim = self.input_dim,
            output_dim = self.output_dim,
        )

        self.classifier_reg = Video_Classifier_reg(
            input_dim = 512,
        )

        self.mse = nn.MSELoss()
        self.corss_entropy = nn.CrossEntropyLoss()


    def forward(self, visions, labels):
        # V_feature = self._get_video_features(mp4_file_paths)
        V_feature = visions.mean(dim=-1)
        V_feature = self.projector(V_feature)
        reg_result = self.classifier_reg(V_feature)
        # reg_loss = self.mse(reg_result.view(-1), labels)
        reg_loss = self.corss_entropy(reg_result, labels)
        return reg_result, reg_loss
        pass




    def _get_video_features(self, mp4_file_paths):
        features = []
        for mp4_file in mp4_file_paths:
            container = av.open(mp4_file)
            indices = self._sample_frame_indices(container.streams.video[0].frames, 4)
            video = self._read_video_pyav(container, indices)


            pixel_values = self.processor(images=list(video), return_tensors="pt").pixel_values
            pixel_values = pixel_values.mean(dim=1).to(self.device)

            features.append(pixel_values)
        V_feature = torch.cat(features, dim=0)
        raw_output = self.model(V_feature)
        return raw_output["pooler_output"]
    
    def _read_video_pyav(self, container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        container.seek(0)
        indices = list(indices)
        for i, frame in enumerate(container.decode(video=0)):
            while i in indices:
                frames.append(frame)
                indices.remove(i)  
            if not indices:
                break
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])
    


    def _sample_frame_indices(self, seg_len, sample_num):
        # Initialize the indices list with the first and last frame
        indices = [0, seg_len - 1]

        # If the sample_num is greater than 2, calculate intermediate frames
        if sample_num > 2:
            step = (seg_len - 1) / (sample_num - 1)
            for i in range(1, sample_num - 1):
                indices.insert(-1, round(i * step))

        return indices
    

class Video_Classifier_reg(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super(Video_Classifier_reg, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x
    