import torch.nn as nn
from transformers import ViTModel, ViTForImageClassification

from networks.modules.transformer import PatchEmbedding, MHSA
from config import cfg

## Accuracy with CosineAnnealingWarmupRestarts ##
## Pretrained ##
# Model | Type   | Dataset | Acc
# ViT    base    'cifar10': 98.77 (sgd / lr: 3e-2)

##

class MLP(nn.Module):
    def __init__(self, hidden_dim, mlp_hidden_dim, drop_rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(mlp_hidden_dim, hidden_dim)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Transformer(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_hidden_dim, drop_rate, attention_drop_rate):
        super(Transformer, self).__init__()
        self.attn = MHSA(hidden_dim, num_heads, drop_rate, attention_drop_rate)
        self.mlp = MLP(hidden_dim, mlp_hidden_dim, drop_rate)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=1e-6) # => 각 d차원에 대해서 정규화
        self.ln2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        res = x
        x, A = self.attn(self.ln1(x))
        x = x + res
        
        res = x
        x = self.mlp(self.ln2(x))
        x = x + res
        return x, A

class ViT(nn.Module):
    def __init__(
        self,
        patch_size = 16,
        drop_rate = 0.1, # 0.1
        attention_drop_rate = 0.1, # 0.1
        pretrained = True
    ):
        super(ViT, self).__init__()

        network_type = {
            "base": [12, 768, 3072, 12], # 86M Param
            "large": [24, 1024, 4096, 16], # 307M Param
            "huge": [32, 1280, 5120, 16] # 632M Param
        }
        num_layers, hidden_dim, mlp_hidden_dim, num_heads = network_type[cfg.network_type]

        self.patch_embd = PatchEmbedding(patch_size, hidden_dim, drop_rate)  # b, 3, 32, 32 => b, w*h*p, 3
        transformer = [Transformer(hidden_dim, num_heads, mlp_hidden_dim, drop_rate, attention_drop_rate) for _ in range(num_layers)]
        self.transformer = nn.Sequential(*transformer)

        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp_head = nn.Linear(hidden_dim, cfg.class_num)            

    def forward(self, x, attn_vis = False):
        x = self.patch_embd(x)
        attn_weight = []
        for transformer in self.transformer:
            x, A = transformer(x)
            attn_weight.append(A)
        x = self.mlp_norm(x)
        x = self.mlp_head(x[:,0])
        if attn_vis:
            return x, attn_weight
        else:
            return x


    def init_weights(self):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained
        # ViT: 'google/vit-base-patch16-224-in21k'
        # SwinT: 'microsoft/swin-tiny-patch4-window7-224'
        # pretrained_state_dict = ViTModel.from_pretrained(f'google/vit-{cfg.network_type}-patch16-224-in21k').state_dict()
        pretrained_state_dict = ViTForImageClassification.from_pretrained(f'google/vit-{cfg.network_type}-patch16-224').state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
            
        ## Transfer parameters from pretrained model to current model
        for i, param in enumerate(param_names[:-2]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]
        
        ## Classifier Head Init
        ## => we remove the pre-trained prediction head and attach a zero-initialized D × K feedforward layer,
        ##    where K is the number of downstream classes.
        nn.init.zeros_(self.mlp_head.weight)
        nn.init.zeros_(self.mlp_head.bias)
            
        self.load_state_dict(state_dict)

        print(f"Initialize ViT-{cfg.network_type}-Patch16-224 from pretrained model")