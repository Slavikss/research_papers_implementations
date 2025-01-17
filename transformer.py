import math
import torch
import torch.nn as nn

# Transformer model for *Translation task*
# It applies cross-attention of source tokens and target tokens 

class MultiHeadAttentionLayer(nn.Module):
    """
    Multi-head Attention layer(other implementation)
    """
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        # assert if n_heads is not divisor of hid_dim
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
           
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
    
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
     
        # Q * K^T
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
                
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(energy, dim = 1)

        x = torch.matmul(self.dropout(attention), V)
                
        x = x.permute(0, 2, 1, 3).contiguous()
                
        x = x.view(batch_size, -1, self.hid_dim)
        
        x = self.fc_o(x)
                
        return x, attention
    
class PositionwiseFeedforwardLayer(nn.Module):
    """
    Fully-connected Linear layer 
    """
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # applying relu and dropout after first layer
        x = self.dropout(torch.relu(self.fc_1(x)))

        x = self.fc_2(x)

        return x

class EncoderLayer(nn.Module):
    """
    Encoder layer of transformer
    """
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        # FF layer
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        
        # layernorm after FFN 
        self.ff_layer_norm = nn.LayerNorm(hid_dim)

         # attention
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        
        # layernorm after attention
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        # ignoring attention as 2 parameter 
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src


class Encoder(nn.Module):
    """
    Encoder of Transformer, contains:
     - EncoderLayer's,
     - Token Embeddings
     - Position Embedding(encodings)

     Encodes all our tokens into hidden dim
    """
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()

        self.device = device

        # init token embeddings
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)

        # init positional encodings(may be simple embedding table)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        # init encoder layers for n_layers times
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(hid_dim)
        
    def forward(self, src, src_mask):
        batch_size, src_len = src.shape

        # init tokens position 
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # getting token embeddings, normalizing, summing with position embeddings
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # applying through all encoder layers
        for layer in self.layers:
            src = layer(src, src_mask)

        return src    
    
class DecoderLayer(nn.Module):
    """
    Decoder Layer of transformer 
    """
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        # attention for decoder layer
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)

        # layernorm for decoder layer
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)

        # encoder attention
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
            
        # encoder layer norm
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)

        # FF layer
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        
        # layernorm after FF
        self.ff_layer_norm = nn.LayerNorm(hid_dim)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        # applying all this layers sequientially

        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        _trg = self.positionwise_feedforward(trg)
        
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
 
        return trg, attention
    
class Decoder(nn.Module):
    """"
    Transformer's Decoder

    Encoder of Transformer, contains:
     - DecoderLayer's,
     - Token Embeddings
     - Position Embedding(encodings)

    Decodes hidden dim into token
    """
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)

        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(hid_dim)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):   
        batch_size, trg_len= trg.shape
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
    
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
    
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        output = self.fc_out(trg)
                    
        return output, attention
    
class Transformer(nn.Module):
    """
    Transformer 
    """
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        # our encoder
        self.encoder = encoder

        # our decoder
        self.decoder = decoder

        # padding indexes 
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.device = device
        
    def make_src_mask(self, src):

        # make src bool mask 
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask
    
    def make_trg_mask(self, trg):
        trg_len = trg.shape[1]

        # make trg pad mask
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        # making tril tensor [trg_len, trg_len]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool() 

        # applying both masks
        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask

    def forward(self, src, trg):  

        # making masks
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # encode tokens 
        enc_src = self.encoder(src, src_mask)

        # decode encoded pert with attention return
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention