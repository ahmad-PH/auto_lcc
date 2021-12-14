import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence

class GRU(nn.Module):
    def __init__(self, input_dims: int, num_class: int, hidden_dim:int, num_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim        
        self.lstm_layers = num_layers

        self.EMB_SIZE = input_dims
        self.title_gru = nn.GRU(input_dims, self.hidden_dim, self.lstm_layers, batch_first=True)
        self.synop_gru = nn.GRU(input_dims, self.hidden_dim, self.lstm_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)

        self.final = nn.Linear(self.hidden_dim * 2, num_class)
        self.relu = nn.ReLU()

    # Expect 1 row for now
    def forward(self, x, x_lengths, synop):
        h_0 = Variable(torch.zeros(self.lstm_layers, x_lengths, self.hidden_dim)).to(dev) #hidden state

        title_gru_out, _ = self.title_gru(x, h_0)
        seq_unpacked, lens_unpacked = pad_packed_sequence(title_gru_out, batch_first=True)
        title_outputs = self.last_timestep(seq_unpacked, lens_unpacked)

        h_1 = Variable(torch.zeros(self.lstm_layers, x_lengths, self.hidden_dim)).to(dev) #hidden state
        synop_gru_out, _ = self.synop_gru(synop, h_1)
        seq_unpacked, lens_unpacked = pad_packed_sequence(synop_gru_out, batch_first=True)
        synop_outputs = self.last_timestep(seq_unpacked, lens_unpacked)

        output = torch.cat((title_outputs, synop_outputs), dim=1)
        output = self.final(output)
        return output


    def last_timestep(self, unpacked, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(unpacked.size(0),
                                               unpacked.size(2)).unsqueeze(1).to(dev)
        return unpacked.gather(1, idx).squeeze()

    def predict(self, X):
        return torch.argmax(self.forward(**X), dim=1)
