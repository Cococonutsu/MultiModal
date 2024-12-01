import torch
import torch.nn as nn

class Vision_Encoder(nn.Module):
    def __init__(self, config):
        super(Vision_Encoder, self).__init__()
    
        self.device = config.train_param.DEVICE
        self.encoder = TfEncoder(
            d_model=768,
            nhead=8, 
            dim_feedforward=768,
            num_layers=2,
            dropout=0.5, 
            activation='gelu',
        )
        self.layernorm = nn.LayerNorm(768)


        self.classifier_reg = Video_Classifier_reg(
            input_size = 768,
            hidden_size = [768//2, 768//4, 768//8],
            output_size= 1
        )

        self.mse = nn.MSELoss()


    def forward(self, visions, vision_masks, labels):
        V_feature = self.encoder(visions, src_key_padding_mask=vision_masks)
        V_feature = self.layernorm(V_feature)
        V_feature = torch.mean(V_feature, dim=1, keepdim=True)
        reg_result = self.classifier_reg(V_feature).squeeze()
        reg_loss = self.mse(reg_result, labels)
        return reg_result, reg_loss
            
class Video_Classifier_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_out=0.3, name=None):
        super(Video_Classifier_reg, self).__init__()

        ModuleList = []
        for i, h in enumerate(hidden_size):
            if i == 0:
                ModuleList.append(nn.Linear(input_size, h))
                ModuleList.append(nn.GELU())
            else:
                ModuleList.append(nn.Linear(hidden_size[i - 1], h))
                ModuleList.append(nn.GELU())
        ModuleList.append(nn.Linear(hidden_size[-1], output_size))

        self.MLP = nn.Sequential(*ModuleList)

    def forward(self, x):
        x = self.MLP(x)
        return x
    

class PositionEncodingTraining(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, fea_size=20, tf_hidden_dim=768, drop_out=0.3):
        super().__init__()

        self.cls_token = nn.Parameter(torch.ones(1, 1, tf_hidden_dim))
        num_patches = 500
        self.proj = nn.Linear(fea_size, tf_hidden_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, tf_hidden_dim))
        self.dropout = nn.Dropout(drop_out)

    def forward(self, embeddings):
        batch_size = embeddings.shape[0]
        embeddings = self.proj(embeddings)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class TfEncoder(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.2, activation='gelu'):
        super(TfEncoder, self).__init__()

        self.pos_encoder = PositionEncodingTraining()
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src, src_key_padding_mask=None):
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        return output.transpose(0, 1)