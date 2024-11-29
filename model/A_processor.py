import torch 
import torch.nn as nn
import numpy as np

from transformers import AutoModel, AutoTokenizer, Data2VecAudioModel
from model.projector import Projector

class Audio_Encoder(nn.Module):
    def __init__(self, config):
        super(Audio_Encoder, self).__init__()

        self.device = config.train_param.DEVICE

        self.input_dim = 768
        self.output_dim = 256
        self.category_num = 3

        self.projector = Projector(
            input_dim = self.input_dim,
            output_dim = self.output_dim,
        )

        self.classifier_reg = Audio_Classifier_reg(
            input_dim = 512,
        )
        self.mse = nn.MSELoss()

        self.data2vec_model = Data2VecAudioModel.from_pretrained(config.Audio.DATA2VEC).to(self.device)

    def forward(self, test_audio_features, test_audio_masks, labels, mode):
        audio_out = self.data2vec_model(test_audio_features, test_audio_masks, output_attentions=True)
        if mode == "single_cls":
            A_hidden_states = audio_out.last_hidden_state
            A_feature = self.projector(A_hidden_states)
            reg_result = self.classifier_reg(A_feature).mean(dim=1)
            reg_loss = self.mse(reg_result.view(-1), labels)
            return reg_result, reg_loss
        else:
            A_features = []
            audio_mask_idx_new = []
            for batch in range(A_hidden_states.shape[0]):
                layer = 0
                while layer<12:
                    try:
                        padding_idx = sum(audio_out.attentions[layer][batch][0][0]!=0)
                        audio_mask_idx_new.append(padding_idx)
                        break
                    except:
                        layer += 1
                truncated_feature = torch.mean(A_hidden_states[batch][:padding_idx],0) #Shape is [768]
                A_features.append(truncated_feature)
            A_features = torch.stack(A_features,0).to(self.device)
            ## create new audio mask
            audio_mask_new = torch.zeros(A_hidden_states.shape[0], A_hidden_states.shape[1]).to(self.device)
            for batch in range(audio_mask_new.shape[0]):
                audio_mask_new[batch][:audio_mask_idx_new[batch]] = 1
            audio_inputs, audio_attn_mask = self.prepend_cls(A_hidden_states, audio_mask_new, 'audio')
            return audio_inputs, audio_attn_mask
        
    def prepend_cls(self, inputs, masks):

        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = self.text_cls_emb(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, inputs.size(2))
        outputs = torch.cat((cls_emb, inputs), dim=1)
        
        cls_mask = torch.ones(inputs.size(0), 1).to(device=inputs.device)
        masks = torch.cat((cls_mask, masks), dim=1)
        return outputs, masks

    
class Audio_Classifier_reg(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(Audio_Classifier_reg, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x
    