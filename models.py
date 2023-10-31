import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size=20, hidden_size=512, act=F.tanh, dropout_p=0.1):
        super().__init__()
        # self.bn = nn.BatchNorm1d(input_size)
        self.rnn_type = nn.GRU
        self.lstm1 = self.rnn_type(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout_p,
            num_layers=5,
            batch_first=True,
            bidirectional=True,
        )
        hidden_size_d = hidden_size * 2
        hidden_size_dd = hidden_size * 2 * 2
        hidden_size_ddd = hidden_size * 2 * 2 * 2
        self.drop1 = nn.Dropout(dropout_p)
        self.lstm2 = self.rnn_type(
            input_size=hidden_size_d,
            hidden_size=hidden_size_d,
            # dropout=0.25,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop2 = nn.Dropout(dropout_p)
        self.lstm3 = self.rnn_type(
            input_size=hidden_size_dd,
            hidden_size=hidden_size_dd,
            # dropout=0.25,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=hidden_size_ddd, out_features=hidden_size_ddd // 8),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(
                in_features=hidden_size_ddd // 8, out_features=hidden_size_ddd // 16
            ),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(in_features=hidden_size_ddd // 16, out_features=input_size),
        )
        self.act = act

    def forward(self, x):
        # x = self.bn(x)
        x = x.squeeze()  # bs, 20, 1 -> bs, 20
        x_1, _ = self.lstm1(x)
        # print(x_1.shape)
        x = self.act(self.drop1(x_1))
        # print(x.shape, x_1.shape)
        x_2, _ = self.lstm2(x)  # + x_1)
        # print(x.shape, x_2.shape)
        x = self.act(self.drop2(x_2))
        x, _ = self.lstm3(x)  # + x_2)  # bs, 20
        # print(x.shape)
        x = self.act(x)
        x = self.fc(x)
        return x.unsqueeze(-1)
