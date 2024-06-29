import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
import math
import yfinance as yf
from math import sqrt

from torch import Tensor
from typing import Optional, Any, Union, Callable
import torch.nn.functional as F


class RobertaEncoder(nn.Module):
    def __init__(self, device='cuda:0'):
        super().__init__()
        self.device = device
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base') 
        self.model = RobertaModel.from_pretrained('roberta-base')

    def forward(self, text):
        e = self.tokenizer(text, return_tensors='pt').to(self.device)
        output = self.model(**e)
        return output.last_hidden_state[:,0,:]
 
class c_transformer_layer(nn.Module):
    def __init__(self, static_dim, seq_dim, act_fn, data_dim, nhead, dim_ff, dropout):
        super(c_transformer_layer, self).__init__()
        self.static_dim = static_dim
        self.seq_dim = seq_dim
        self.data_dim = data_dim
        self.nhead = nhead
        self.dim_ff = dim_ff
        self.act_fn = act_fn
        self.seq_len = int(self.seq_dim/data_dim)
        self.lin_inp = nn.Sequential(
            nn.Linear(self.static_dim+self.seq_dim, self.seq_dim),
            self.act_fn(),
            #nn.Linear(self.seq_dim, self.seq_dim),
            #self.act_fn()
        )
        self.tran_layer = nn.TransformerEncoderLayer(d_model=self.data_dim, nhead=nhead, dim_feedforward=dim_ff, 
                                    activation=self.act_fn(), batch_first = True, dropout=dropout)
        
        
    def forward(self, x, sum):
        batch_size = x.shape[0]
        x = self.tran_layer(x)
        res = torch.reshape(x, (batch_size, self.seq_dim))
        x = torch.cat((torch.reshape(x, (batch_size, self.seq_dim)), sum), dim=1)
        x = self.lin_inp(x)
        x = x + res
        x = torch.reshape(x, (batch_size, self.seq_len, self.data_dim))
        return x
    
class base_transformer_layer(nn.Module):
    def __init__(self, act_fn, data_dim, nhead, dim_ff, dropout):
        super(base_transformer_layer, self).__init__()
        self.tran_layer = nn.TransformerEncoderLayer(d_model=data_dim, nhead=nhead, dim_feedforward=dim_ff, 
                                    activation=act_fn(), batch_first=True, dropout=dropout)
        
        
    def forward(self, x):
        x = self.tran_layer(x)
        return x

class Dist_Pred(nn.Module):
    def __init__(self,seq_len=350, data_dim=5, num_bins=21, num_days=5, nhead=5, ff=15000, layers=72, sum_emb=76, scale=1, s_scale=0, num_cls_layers=6, dropout=0.1):
        super(Dist_Pred, self).__init__()
        self.num_bins = num_bins
        self.seq_len = seq_len
        self.dim = data_dim
        self.num_preds = num_days-1
        self.act_fn = nn.GELU
        self.act = nn.GELU()
        self.scale = scale
        self.s_scale = s_scale
        self.sum_emb = sum_emb
        self.seq_dim = self.seq_len*self.dim
        self.num_lin_layers = num_cls_layers
        self.dropout = nn.Dropout(dropout)
        # Transformer Layers
        self.layers = nn.ModuleList([base_transformer_layer(act_fn=self.act_fn,data_dim=self.dim, nhead=nhead, 
                                            dim_ff=ff, dropout=0.1) for i in range(layers)])  
        linear_in_dim = 100
        # Classification Head
        '''
        self.linear_layers = nn.ModuleList([nn.Linear(int(self.scale*self.seq_len*data_dim), int(self.scale*self.seq_len*data_dim)) for i in range(self.num_lin_layers)]) 
        self.linear_out = nn.Linear(int(self.scale*self.seq_len*data_dim), num_bins*self.num_preds)
        self.linear_in = nn.Sequential(
            nn.Linear(self.seq_len*self.dim+self.sum_emb, int(self.scale*self.seq_len*data_dim)),
            self.act_fn(),)
        '''
        self.linear_in = nn.Sequential(
            nn.Linear(self.seq_len*self.dim+self.sum_emb+156, linear_in_dim),
            self.act_fn(),
            nn.Linear(linear_in_dim, num_bins*self.num_preds),
            )

        print('Linear dim: ', int(self.scale*self.seq_len*data_dim))
        print('Transformer params ', sum(param.numel() for param in self.layers.parameters()))
        
        
        # Summary Module
        #self.summary_in = nn.Sequential(nn.Linear(sum_emb, int(self.s_scale*sum_emb)),
        #                                self.act_fn())
        #self.summary_layers = nn.ModuleList([nn.Linear(int(self.s_scale*sum_emb), int(self.s_scale*sum_emb))])
        self.pos_encoding = PositionalEncoding(data_dim, seq_len)
        self._encoding = nn.Parameter(self.pos_encoding.encoding, requires_grad=False)

    # For use in forward()
    def pos_encode(self, x):
        batch_size, seq_len, data_dim = x.size()
        return self._encoding[:seq_len, :]
    
    def forward(self, x, s):
        batch_size = x.shape[0]
        x = torch.flip(x,[1])
        x = x + self.pos_encode(x)
        
        # x shape is batch, 400, 52.  We need to add new entries for the summary emb, and pad it to match the length
        # It will take 15 entries, and needs to have 12 elements padded so it is divisible by the data dim of 52
        #pad = torch.randn((1, 12), device='cuda')*1e-8
        #pad = pad.repeat(batch_size, 1)
        
        # s is shape (batch, 768) -> (batch, 780)
        #s = torch.cat((s, pad), dim=1) 

        # Reshape this to (batch, 15, 52)
        s = torch.reshape(s, (batch_size, 19, 52))
        
        # Add these data points to existing seqence
        x = torch.cat((x, s), dim=1)

        # Send the final data through transformer layers
        res_i = 0
        res_2 = 0
        i = 0
        for layer in self.layers:
            x = layer(x) + res_i + res_2
            if i == 0:
                res_i = x
            if i == 1:
                res_2 = x
            i += 1
        # Send transformer activation through linear classification head
        x = torch.reshape(x, (batch_size, self.seq_dim+self.sum_emb+156))

        x = self.linear_in(x)
        #for lin_layer in self.linear_layers:
        #    x = lin_layer(x)
        #    x = self.act(x)
            # x = self.dropout(x)
        #x = self.linear_out(x)

        # Return reshaped output
        x = torch.reshape(x, (batch_size, self.num_bins, self.num_preds))
        return x
    
    def transformer(self, x, s):
        batch_size = x.shape[0]
        x = torch.flip(x, dim=1)
        x = x + self.pos_encode(x)
        x = torch.cat((x, s), dim=1)
        res_i = 0
        res_2 = 0
        i = 0
        for layer in self.layers:
            x = layer(x) + res_i + res_2
            if i == 0:
                res_i = x
            if i == 1:
                res_2 = x
            i += 1
        return x

class Dist_DirectPredSumTran(nn.Module):
    def __init__(self,seq_len=350, data_dim=5, num_bins=21, num_days=5, nhead=5, ff=15000, layers=72, sum_emb=76, scale=1):
        super(Dist_DirectPredSumTran, self).__init__()
        self.num_bins = num_bins
        self.seq_len = seq_len
        self.dim = data_dim
        self.num_preds = num_days-1
        self.act_fn = nn.GELU
        self.scale = scale
        self.sum_emb = sum_emb
        self.seq_dim = self.seq_len*self.dim

        self.layers = nn.ModuleList([c_transformer_layer(static_dim=sum_emb, seq_dim=self.seq_dim, act_fn=self.act_fn,data_dim=self.dim, nhead=nhead, 
                                            dim_ff=ff, dropout=0.1) for i in range(layers)])  
        self.linear = nn.Sequential(
            nn.Linear(seq_len*data_dim, int(self.scale*seq_len*data_dim)),
            nn.GELU(),
            nn.Linear(int(self.scale*seq_len*data_dim), int(self.scale*seq_len*data_dim)),
            nn.GELU(),
            nn.Linear(int(self.scale*seq_len*data_dim), int(self.scale*seq_len*data_dim)),
            nn.GELU(),
            nn.Linear(int(self.scale*seq_len*data_dim), int(self.scale*seq_len*data_dim)),
            nn.GELU(),
            nn.Linear(int(self.scale*seq_len*data_dim), num_bins*self.num_preds),
            )
        self.pos_encoding = PositionalEncoding(data_dim, seq_len)
        self._encoding = nn.Parameter(self.pos_encoding.encoding, requires_grad=False)   

    def pos_encode(self, x):
        batch_size, seq_len, data_dim = x.size()
        return self._encoding[:seq_len, :]
    

    def encode(self, x, s):
        batch_size = x.shape[0]
        x = x + self.pos_encode(x)
        res = 0
        init_res = 0
        res_2 = 0
        res_3 = 0
        res_4 = 0
        res_5 = 0
        res_6 = 0
        res_7 = 0
        res_8 = 0
        res_9 = 0
        res_10 = 0
        res_11 = 0
        res_12 = 0
        res_13 = 0
        res_14 = 0
        i = 0
        for layer in self.layers:
            n = i+3
            x = (layer(x, s) + (res + init_res + res_2 + res_3+res_4+res_5+res_6+res_7+res_8+res_9+res_10+
                                res_11+res_12+res_13+res_14))/(sqrt(n))
            res = x 
            if i == 0:
                init_res = x
            if i == 1:
                res_2 = x
            if i == 2:
                res_3 = x
            if i == 3:
                res_4 = x
            if i == 4:
                res_5 = x
            if i == 5:
                res_6 = x
            if i == 6:
                res_7 = x
            if i == 7:
                res_8 = x
            if i == 8:
                res_9 = x
            if i == 9:
                res_10 = x
            if i == 10:
                res_11 = x
            if i == 11:
                res_12 = x
            if i == 12:
                res_13 = x
            if i == 13:
                res_14 = x
            i += 1
        transformer_activation = x
        x = torch.reshape(x, (batch_size, self.seq_dim))
        x = self.linear(x) # 1 160 4
        #print(x.shape)
        softmax = nn.Softmax(dim=1)
        x = softmax(x)
        linear_output = x.flatten()
        return (transformer_activation, linear_output) # Shape (200, 52), ()
    
    def forward(self, x, s):
        #print(x.shape, s.shape)
        #print(self.pos_enc.encoding.device, x.device)
        batch_size = x.shape[0]
        #print(x.shape, self.pos_encode(x).shape)
        x = x + self.pos_encode(x)
        res = 0
        init_res = 0
        res_2 = 0
        res_3 = 0
        res_4 = 0
        res_5 = 0
        res_6 = 0
        res_7 = 0
        res_8 = 0
        res_9 = 0
        res_10 = 0
        res_11 = 0
        res_12 = 0
        res_13 = 0
        res_14 = 0
        res_15 = 0
        res_16 = 0
        res_17 = 0
        res_18 = 0
        res_19 = 0
        #res_20 = 0
        #res_21 = 0
        #res_22 = 0
        #res_23 = 0
        #res_24 = 0
        #res_25 = 0
        i = 0
        for layer in self.layers:
            n = i+3
            x = (layer(x, s) + (res + init_res + res_2 + res_3+res_4+res_5+res_6+res_7+res_8+res_9+res_10+
                                res_11+res_12+res_13+res_14))/(sqrt(n))
            res = x 
            if i == 0:
                init_res = x
            if i == 1:
                res_2 = x
            if i == 2:
                res_3 = x
            if i == 3:
                res_4 = x
            if i == 4:
                res_5 = x
            if i == 5:
                res_6 = x
            if i == 6:
                res_7 = x
            if i == 7:
                res_8 = x
            if i == 8:
                res_9 = x
            if i == 9:
                res_10 = x
            if i == 10:
                res_11 = x
            if i == 11:
                res_12 = x
            if i == 12:
                res_13 = x
            if i == 13:
                res_14 = x
            i += 1
        x = torch.reshape(x, (batch_size, self.seq_dim))
        x = self.linear(x)
        x = torch.reshape(x, (batch_size, self.num_bins, self.num_preds))
        return x

class Layer_Dist_DirectPredSumTran(nn.Module):
    def __init__(self,seq_len=350, data_dim=5, num_bins=21, num_days=5, nhead=5, ff=15000, layers=72, sum_emb=76, scale=1,num_lin_layers=8):
        super(Layer_Dist_DirectPredSumTran, self).__init__()
        self.num_lin_layers = num_lin_layers
        self.num_bins = num_bins
        self.seq_len = seq_len*2
        self.dim = data_dim
        #self.dim = data_dim*2
        self.num_preds = num_days-1
        self.act_fn = nn.GELU
        self.act = nn.GELU()
        self.scale = scale
        self.sum_emb = sum_emb
        self.seq_dim = self.seq_len*self.dim
        self.layer_act_dim = 160*4
        self.layers = nn.ModuleList([c_transformer_layer(static_dim=self.layer_act_dim, seq_dim=self.seq_dim, act_fn=self.act_fn,data_dim=self.dim, nhead=nhead, 
                                            dim_ff=ff, dropout=0.1) for i in range(layers)])  
        self.summary_module_dim = 300
        self.sum_scale = 3
        self.linear_in = nn.Sequential(
            nn.Linear(self.seq_len*self.dim+self.summary_module_dim+4*160, int(self.scale*self.seq_len*data_dim)),
            nn.GELU(),
            )
        self.linear_layers = nn.ModuleList([nn.Linear(int(self.scale*self.seq_len*data_dim), int(self.scale*self.seq_len*data_dim)) for i in range(self.num_lin_layers)]) 
        self.linear_out = nn.Linear(int(self.scale*self.seq_len*data_dim), num_bins*self.num_preds)
        self.summary_module = nn.Sequential(
            nn.Linear(sum_emb, int(self.sum_scale*sum_emb)),
            nn.GELU(),
            nn.Linear(int(self.sum_scale*sum_emb), int(self.sum_scale*sum_emb)),
            nn.GELU(),
            nn.Linear(int(self.sum_scale*sum_emb), int(self.sum_scale*sum_emb)),
            nn.GELU(),
            nn.Linear(int(self.sum_scale*sum_emb), self.summary_module_dim),
            nn.GELU()
        )
        self.pos_encoding = PositionalEncoding(self.dim, int(self.seq_len/2))
        self._encoding = nn.Parameter(self.pos_encoding.encoding, requires_grad=False)   

    def pos_encode(self, x):
        batch_size, seq_len, data_dim = x.size()
        return self._encoding[:seq_len, :]

    def forward(self, x, s, a_lin):
        # x -> previous 200 days
        # s -> summary embedding
        # a_seq -> first 200 days from previous model
        # a_lin -> flattened linear activation from  previous model
        #print(x.shape, s.shape)
        #print(self.pos_enc.encoding.device, x.device)
        batch_size = x.shape[0]
        #x = x + self.pos_encode(x)
        #print(x[:,200:,:].shape, self.pos_encode(x[:,200:,:]).shape)
        x[:,200:,:] = x[:,200:,:] + self.pos_encode(x[:,200:,:])
        i = 0
        #res = 0
        for layer in self.layers:
            x = layer(x, a_lin)
        x = torch.reshape(x, (batch_size, self.seq_dim))
        s = self.summary_module(s.squeeze(1))
        #print(x.shape, s.shape)
        x = torch.cat((x, s, a_lin), dim=-1)
        x = self.linear_in(x)
        #print(x, type(x))
        l_res = 0
        l_res_0 = 0
        i = 0
        for layer in self.linear_layers:
            x = layer(x) + l_res + l_res_0
            x = self.act(x)
            l_res = x
            if i == 0:
                l_res_0 = x
                i += 1
        x = self.linear_out(x)
        x = torch.reshape(x, (batch_size, self.num_bins, self.num_preds))
        return x
    
# The following is from hyunwoongko's implementation of the Transformer model: https://github.com/hyunwoongko/transformer
class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient
        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        if d_model % 2 == 0:
            self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        else:
            self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i[:d_model//2] / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]
        #print(x.size())
        batch_size, seq_len, data_dim = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]         

   