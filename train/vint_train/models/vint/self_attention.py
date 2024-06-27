import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#!定义了一个 PositionalEncoding 类，它是一个神经网络模块，用于向输入的特征添加位置信息。这个类是 nn.Module 的一个子类，
#!通常在处理序列数据时使用，特别是在使用 Transformer 模型时非常关键，因为 Transformer 架构本身不具备处理序列中元素位置关系的能力。
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # Compute the positional encoding once
        #!创建一个全零的张量 pos_enc，形状为 [max_seq_len, d_model]，用于存储每个位置的编码
        pos_enc = torch.zeros(max_seq_len, d_model)
        #!pos 是一个形状为 [max_seq_len, 1] 的张量，表示各个位置的索引
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        #!div_term 是一个衰减因子，用于在不同的维度上给出不同的波长，通过 exp 和 log 的组合计算得到，其基本形式是一个几何级数。
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #!使用正弦函数和余弦函数交替填充 pos_enc 的不同列，其中偶数列用 sin 填充，奇数列用 cos 填充。这种设置使得每个位置的编码包含频率不同的正弦波和余弦波，帮助模型学习到位置信息。
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)
         
        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        #!将计算得到的位置编码 pos_enc 注册为一个缓冲区，这意味着它不会被视为模型参数，在训练过程中不会被更新，但是在模型保存和加载时会被保留。
        #!缓冲区（buffer）属于模型中不会被更新的部分参数。它们在模型的训练过程中不会通过反向传播进行更新，而是用来保存模型需要的额外固定数据。这些数据在模型的前向传播过程中可能会用到，但它们不会被优化。
        #!注册到缓冲区的变量本质上是模型的一部分，并保存在模型的状态字典中。因此，即使重新启动机器，也可以通过加载保存的模型状态来恢复这些缓冲区。
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x

class MultiLayerDecoder(nn.Module):
    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64], nhead=8, num_layers=8, ff_dim_factor=4):
        super(MultiLayerDecoder, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        self.sa_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor*embed_dim, activation="gelu", batch_first=True, norm_first=True)
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)
        self.output_layers = nn.ModuleList([nn.Linear(seq_len*embed_dim, embed_dim)])
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers)-1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i+1]))

    def forward(self, x):
        if self.positional_encoding: x = self.positional_encoding(x)
        x = self.sa_decoder(x)
        # currently, x is [batch_size, seq_len, embed_dim]
        x = x.reshape(x.shape[0], -1)
        for i in range(len(self.output_layers)):
            x = self.output_layers[i](x)
            x = F.relu(x)
        return x
