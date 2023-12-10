"""
-*- coding: utf-8 -*-

@Author : Aoran,Li
@Time : 2023/12/8 2:07
@File : model.py
"""
from train.init import *


class CNNALSTM(nn.Module):
    def __init__(self, out_put=2):
        super().__init__()

        self.cnn = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(10, 2), dilation=(2, 1)),
                                 nn.LeakyReLU(negative_slope=0.01, inplace=True),
                                 nn.MaxPool2d((2, 2)),
                                 nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(6, 2)),
                                 nn.LeakyReLU(negative_slope=0.01, inplace=True))

        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True)

        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)

        self.dropout = nn.Dropout(p=0.5)
        self.fc_sequence = nn.Sequential(nn.Linear(64, 32),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(32, 16),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(16, out_put),
                                         nn.Softmax(dim=-1))

    def forward(self, x):
        # Input shape: (batch size, 60, 5)
        # CNN
        x = x.unsqueeze(1)  # (bs, 1, 60, 5)
        output = self.cnn(x)  # (bs, 32, 16, 1)
        output = output.squeeze(-1).transpose(1, 2)  # (bs, 16, 32)

        # LSTM
        h0 = torch.zeros(2, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(2, x.size(0), self.lstm.hidden_size).to(x.device)
        output, _ = self.lstm(output)  # (bs, 16, 64)

        # Attention
        output, _ = self.attention(output, output, output)  # (bs, 16, 64)

        # Linear
        output = self.dropout(output)
        output = self.fc_sequence(output)  # (bs, 16, 2)
        return output[:, -1, :]  # (bs, 2)


class LSTMModel:
    def __init__(self, out_put=2):
        super().__init__()

        self.lstm = nn.LSTM(input_size=5,
                            hidden_size=32,
                            num_layers=2,
                            batch_first=True)
        self.fc_sequence = nn.Sequential(nn.Linear(32, 8),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(8, out_put),
                                         nn.Softmax(dim=-1))

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc_sequence(output[:,-1,:])
        return output

