import av
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from transformers import AutoProcessor, XCLIPVisionModel
from model.projector import Projector

from utils.dataloader import DataLoader


class Vision_Encoder(nn.Module):
    def __init__(self, config):
        super(Vision_Encoder, self).__init__()
    # 加载 ViCLIP 模型
        self.device = config.train_param.DEVICE
        self.processor = AutoProcessor.from_pretrained(config.Video.VIDEOMAE)
        self.model = XCLIPVisionModel.from_pretrained(config.Video.VIDEOMAE)

        self.input_dim = 768
        self.output_dim = 256

        self.projector = Projector(
            input_dim = self.input_dim,
            output_dim = self.output_dim,
        )

        self.classifier_reg = Video_Classifier_reg(
            input_dim = 512,
        )

        self.mse = nn.MSELoss()


    def forward(self, mp4_file_paths, labels):
        V_feature = self._get_video_features(mp4_file_paths)
        V_feature = self.projector(V_feature)
        reg_result = self.classifier_reg(V_feature).mean(dim=1)
        reg_loss = self.mse(reg_result.view(-1), labels)
        return reg_result, reg_loss
        pass




    def _get_video_features(self, mp4_file_paths):
        features = []
        for mp4_file in mp4_file_paths:
            container = av.open(mp4_file)
            indices = self._sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
            video = self._read_video_pyav(container, indices)


            pixel_values = self.processor(images=list(video), return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            batch_size, num_frames, num_channels, height, width = pixel_values.shape
            pixel_values = pixel_values.reshape(-1, num_channels, height, width)

            outputs = torch.mean(self.model(pixel_values)["last_hidden_state"], dim=0)
            features.append(outputs.unsqueeze(0))#添加一个batch维度
        return torch.cat(features, dim=0)
    
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
    


    def _sample_frame_indices(self, clip_len, frame_sample_rate, seg_len):
        '''
        从视频中采样指定数量的帧索引。
        参数:
            clip_len (`int`): 需要采样的帧数。
            frame_sample_rate (`int`): 每隔多少帧采样。
            seg_len (`int`): 视频的总帧数。
        返回:
            indices (`List[int]`): 采样的帧索引列表。
        '''
        converted_len = clip_len * frame_sample_rate

        if seg_len < clip_len:
            # 如果视频帧数不足，按顺序重复帧补齐到 clip_len 帧
            print(f"视频帧数不足！seg_len={seg_len}, 需要帧数={clip_len}。按顺序重复采样补齐到 {clip_len} 帧。")

            # 从头采样所有帧
            indices = np.arange(0, seg_len)  # 从头开始采样所有帧

            # 计算需要重复的次数
            repeat_count = (clip_len + seg_len - 1) // seg_len  # 计算需要重复几轮
            full_indices = np.tile(indices, repeat_count)  # 按顺序重复采样帧

            # 截取前 clip_len 个帧
            indices = full_indices[:clip_len]

            return np.sort(indices.astype(np.int64))


        # 正常采样：视频帧数足够，按要求采样
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        return np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    

class Video_Classifier_reg(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(Video_Classifier_reg, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x
    