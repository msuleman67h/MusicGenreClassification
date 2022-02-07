import torch
import torch.nn as nn


class MyLSTM(nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(MyLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.my_lstm_1 = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size * 2, 128)
        self.fc_2 = nn.Linear(128, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(MyLSTM.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(MyLSTM.device)

        out, _ = self.my_lstm_1(x, (h0, c0))
        out = out[:, -1, :]

        out = self.fc_1(out)
        out = torch.relu(out)
        out = self.fc_2(out)
        return out
