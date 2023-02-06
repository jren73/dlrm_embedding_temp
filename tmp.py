import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, bidirectional=True)
        
    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)
        return output, hidden, cell
    
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim*2, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)
        
    def forward(self, hidden, encoder_outputs):
        hidden = hidden.unsqueeze(2)
        attn_energies = self.attn(torch.tanh(self.fc(encoder_outputs)))
        attn_weights = F.softmax(attn_energies, dim=1)
        context = attn_weights * encoder_outputs
        context = context.sum(dim=1)
        return context, attn_weights
    
class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(hidden_dim*2, hidden_dim, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.attention = Attention(hidden_dim)
        
    def forward(self, x, hidden, cell, encoder_outputs):
        context, attn_weights = self.attention(hidden, encoder_outputs)
        x = torch.cat((x, context), dim=1)
        x = x.unsqueeze(0)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        output = self.fc(output.squeeze(0))
        return output, hidden, cell, attn_weights
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, x, target, teacher_forcing_ratio=0.5):
        batch_size = target.shape[1]
        target_len = target.shape[0

https://hannibunny.github.io/mlbook/transformer/attention.html#attention