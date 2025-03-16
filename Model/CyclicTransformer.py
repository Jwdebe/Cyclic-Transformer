import numpy as np
import torch 

from torch import nn
from torch.nn import MultiheadAttention, LayerNorm
import math
import copy 
from typing import List, Union, Dict

class Classifier(nn.Module):
    def __init__(self, config, layer_norm_eps: float = 1e-5) -> None:
        super().__init__()

        reduction_latents = config['reduction_eeg']
        dimension_bottleneck = config['dimension_bottleneck']

        self.bottleneck = nn.Linear(in_features = reduction_latents[-1], out_features = dimension_bottleneck)
        self.bottleactivation = nn.PReLU(init = 0.25)
        self.norm = LayerNorm(dimension_bottleneck, layer_norm_eps) 
        self.fc = nn.Linear(in_features = dimension_bottleneck, out_features = 1)
        self.activation = nn.Sigmoid()
    
    def forward(self, inputs):

        x = self.bottleneck(inputs)
        x = self.norm(x)
        x = self.bottleactivation(x)
        x = self.fc(x)
        pred = self.activation(x)

        return pred

class FeatLayer(nn.Module):
    def __init__(self, d_input: int, d_output: int, dropout: float = 0.25, layer_norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.fc = nn.Linear(d_input, d_output)
        self.norm = nn.BatchNorm1d(d_output, layer_norm_eps)
        self.activation = nn.PReLU(init=0.25)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.fc(inputs)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_input, nhead, dim_feedforward = 1024, dropout = 0.1, layer_norm_eps = 1e-5) -> None:
        super(EncoderLayer, self).__init__()

        self.attn = MultiheadAttention(d_input, nhead, dropout=dropout, batch_first = True)
        self.feed_forward = FeedForward(d_input, dim_feedforward, dropout)
        self.norm1 = LayerNorm(d_input, layer_norm_eps)
        self.norm2 = LayerNorm(d_input, layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """Forward through one encoder layer: multi-head attn => add & norm
           => feed forward => add & norm.
        Args:
            x: embeddings or output of the last layer.
                [batch_size, seq_len, d_model]
            mask: [batch_size, (1 or seq_len), seq_len]
        """
        # multihead attn & norm

        a, _ = self.attn(x, x, x)
        t = self.norm1(x + self.dropout1(a))

        # feed forward & norm
        z = self.feed_forward(t)  # linear(dropout(act(linear(x)))))
        y = self.norm2(t + self.dropout2(z))

        return y

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_input: int, nhead: int, dim_feedforward: int = 1024, dropout: float = 0.1, layer_norm_eps: float = 1e-5) -> None:
        super().__init__()

        self.norm1 = LayerNorm(d_input, layer_norm_eps)
        self.norm2 = LayerNorm(d_input, layer_norm_eps)
        self.attn = MultiheadAttention(d_input, nhead, dropout = dropout, batch_first = True)
        self.feed_forward = FeedForward(d_input, dim_feedforward, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x1, x2):

        a, _ = self.attn(x1, x2, x2)
        t = self.norm1(x1 + self.dropout1(a))
        
        z = self.feed_forward(t)
        y = self.norm2(t + self.dropout2(z))

        return y

class UnimodalEncoder(nn.Module):
    def __init__(self, d_input: int, reduction_latents: List[int], num_heads: int, num_encoders: int) -> None:
        super().__init__()
        dimensions = [d_input] + reduction_latents

        self.feat_encoding_layers = nn.ModuleList([
            FeatLayer(d_input=dimensions[i], d_output=dimensions[i + 1])
            for i in range(len(reduction_latents))
        ])

        self.pos_encoder = PositionalEncoding(d_model=dimensions[-1], dropout=0.1)
        self.layers = nn.ModuleList([
            EncoderLayer(d_input=dimensions[-1], nhead=num_heads)
            for _ in range(num_encoders)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mod in self.feat_encoding_layers:
            x = mod(x)
        x = self.pos_encoder(x)
        for mod in self.layers:
            x = mod(x)
        return x


class Encoder(nn.Module):
    def __init__(self, config) -> None:
        super(Encoder, self).__init__()

        d_eeg = config['dimension_eeg']
        d_acc = config['dimension_acc']

        reduction_eeg = config['reduction_eeg']
        reduction_acc = config['reduction_acc']
        nheads = config['number_heads']
        num_eeg_encoders = config['eeg_encoders']
        num_acc_encoders = config['acc_encoders']
        num_cross_encoders = config['num_cross_encoders']

        d_latent = reduction_eeg[-1]
        
        self.eeg_encoder = UnimodalEncoder(d_input = d_eeg, reduction_latents = reduction_eeg, 
                                            num_heads = nheads, num_encoders = num_eeg_encoders)

        self.acc_encoder = UnimodalEncoder(d_input = d_acc, reduction_latents = reduction_acc, 
                                            num_heads = nheads, num_encoders = num_acc_encoders)    

        cross_attn = _get_clones(CrossAttentionLayer(d_input = d_latent, nhead = nheads), num_cross_encoders)        
        self.cross_attn = cross_attn    
    
    def forward(self, x1, x2):

        x1 = self.eeg_encoder(x1)
        x2 = self.acc_encoder(x2)

        for mod in self.cross_attn:
            x = mod(x1, x2)
    
        return x

class Decoder(nn.Module):

    def __init__(self, config):
        super(Decoder, self).__init__()
        d_output = config['dimension_emg']
        num_heads = config['number_heads']
        num_decoders = config['num_decoders']
        encoder_layers = _get_clones(EncoderLayer(d_input = d_output, nhead = num_heads), num_decoders)    
        self.encoder_layers = encoder_layers

    def forward(self, x):

        for mod in self.encoder_layers:
            x = mod(x)

        return x
    
class Reconstructor(nn.Module):
    def __init__(self, config, d_input, recons_latents) -> None:
        super().__init__()
        
        num_heads = config['number_heads']
        num_decoders = config['num_recons_decoders']
        
        dimensions = [d_input] + recons_latents
        
        feat_decoding_layers = nn.ModuleList([FeatLayer(d_input = dimensions[i], d_output = dimensions[i + 1])
                                                for i in range(len(recons_latents))])
        
        decoder_layers = _get_clones(EncoderLayer(d_input = d_input, nhead = num_heads), num_decoders)
        
        self.decoding_layers = decoder_layers

        self.feat_decoding_layers = feat_decoding_layers


    def forward(self, x):
        
        for mod in self.decoding_layers:
            x = mod(x)

        for mod in self.feat_decoding_layers:
            x = mod(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout) -> None:
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)        

class DynamicFeedForward(nn.Module):
    def __init__(self, d_input, dim_feedforward, d_output, dropout) -> None:
        super(DynamicFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_input, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_output)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)      

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[0, :x.size(1), :]
        return self.dropout(x)

class Translator(nn.Module):
    def __init__(self, config):
        super(Translator, self).__init__()

        d_emg = config['dimension_emg']
        recons_eeg_latents = config['recons_eeg_latents']
        recons_acc_latents = config['recons_acc_latents']

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.reconstrutor_eeg = Reconstructor(config, d_input = d_emg, recons_latents = recons_eeg_latents)

        self.classifier = Classifier(config)

    def forward(self, src1, src2, tgt):
        
        src1 = src1.to(torch.float32)  
        src2 = src2.to(torch.float32) 
        tgt = tgt.to(torch.float32)  
          
        encoded = self.encoder(src1, src2)
        decoded = self.decoder(encoded)

        cycled_decoded_eeg = self.reconstrutor_eeg(decoded)

        pred = self.classifier(encoded)

        return pred, encoded, decoded, cycled_decoded_eeg

    def encoding(self, src1, src2):

        encoded = self.encoder(src1, src2)

        return encoded
    
    def predict(self, src1, src2):
        
        encoded = self.encoding(src1, src2)
        pred = self.classifier(encoded)

        return pred

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
