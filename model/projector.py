import torch 
import torch.nn as nn

from config import MOSI_config     


class Projector(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers = 3, drop_out = 0.1, config = MOSI_config):
        super(Projector, self).__init__()

        self.device = config.train_param.DEVICE
        self.feed_forward_size = output_dim * 2
        self.project_size = output_dim

        self.proj1 = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.GELU(),
            nn.Linear(input_dim*2, output_dim)
        )


        self.proj2 = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.proj2.append(nn.Linear(input_dim, self.project_size, bias = False))
            else:
                self.proj2.append(nn.Linear(self.project_size, self.project_size, bias = False))
            self.proj2.append(nn.GELU())

        self.layernorm_ff = nn.LayerNorm(output_dim)
        self.layernorm = nn.LayerNorm(output_dim)
        self.MLP = nn.Sequential(*self.proj2)

        self.dropout = nn.Dropout(p=drop_out)
                


    def forward(self, batch):
        # input: list of data samples with different seq length
        dropped = self.dropout(batch)
        ff = self.proj1(dropped)
        x = self.MLP(dropped)
        x = torch.cat([self.layernorm(x), self.layernorm_ff(ff)], dim=-1)
        # return x.transpose(0, 1)  # return shape: [seq,batch,fea]
        return x 
        


