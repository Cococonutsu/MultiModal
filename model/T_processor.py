import torch 
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer
from model.projector import Projector

from config import MOSI_config


class Text_Encoder(nn.Module):
    def __init__(self, config):
        super(Text_Encoder, self).__init__()

        self.device = config.train_param.DEVICE

        self.bert = AutoModel.from_pretrained(config.Text.BERT)
        self.text_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)

        self.input_dim = 768
        self.output_dim = 256

        self.projector = Projector(
            input_dim = self.input_dim,
            output_dim = self.output_dim,
        )

        self.classifier_reg = Text_Classifier_reg(
            input_dim = 512,
        )

        self.mse = nn.MSELoss()

    def forward(self, texts_tokens, text_masks, labels, mode):
        raw_output = self.bert(texts_tokens, text_masks)
        if mode == "single_cls":
            T_pooler_output = raw_output["pooler_output"]
            T_feature = self.projector(T_pooler_output)

            reg_result = self.classifier_reg(T_feature)
            
            reg_loss = self.mse(reg_result.view(-1), labels)
            
            return reg_result,  reg_loss
        else:
            T_hidden_states = raw_output.last_hidden_state
            text_inputs, text_attn_mask = self.prepend_cls(T_hidden_states, text_masks)
            return text_inputs, text_attn_mask

    def prepend_cls(self, inputs, masks):

        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = self.text_cls_emb(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, inputs.size(2))
        outputs = torch.cat((cls_emb, inputs), dim=1)
        
        cls_mask = torch.ones(inputs.size(0), 1).to(device=inputs.device)
        masks = torch.cat((cls_mask, masks), dim=1)
        return outputs, masks
            

    
class Text_Classifier_reg(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(Text_Classifier_reg, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x
    