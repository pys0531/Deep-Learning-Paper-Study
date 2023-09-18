import torch
import torch.nn as nn

from config import cfg

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, hidden_dim, drop_rate):
        super(PatchEmbedding, self).__init__()
        nw, nh = int(cfg.input_shape[0]/patch_size), int(cfg.input_shape[1]/patch_size) # number of patch
        seq_len = nw * nh

        self.patch_embd = nn.Conv2d(3, hidden_dim, patch_size, stride = patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embd = nn.Parameter(torch.zeros(1, seq_len+1, hidden_dim)) # Add class token => cls + seq vec
        self.dropout = nn.Dropout(drop_rate)


    def forward(self, x):
        # b, 3, h, w ==> b, dim, h/p, w/p ==> .view and .permute => b, nw*nh, dim
        x = self.patch_embd(x).flatten(2).transpose(1, 2)
        # b, nw*nh+1, dim
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim = 1)
        x = x + self.pos_embd
        x = self.dropout(x)
        return x

class MHSA(nn.Module):
    def __init__(self, hidden_dim, num_heads, drop_rate, attention_drop_rate):
        super(MHSA, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = int(hidden_dim / num_heads) # => D_h is typically set to D/k

        #self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias = False)
        self.q = nn.Linear(hidden_dim, hidden_dim) # 197, 786 => 197, 16*12 => num_head를 12로 놓고 12개의 self-attention 계산 => multi-head self-attention
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(attention_drop_rate)

        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size = x.size(0)

        #q, k, v = self.qkv(x).chunk(3, dim = -1)
        q = self.q(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = self.k(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,3,1)
        v = self.v(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        A = torch.softmax(q @ k / (self.head_dim**0.5), dim = -1) # => Attention Score
        x = self.dropout(A) @ v

        x = x.permute(0,2,1,3)
        x = x.reshape(batch_size, -1, self.hidden_dim)
        x = self.dropout(self.proj(x))
        return x, A