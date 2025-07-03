# transformer encoder 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 

# l layer of transformer encoder 
class EncoderLayer(nn.Module):
    
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim) 
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        # dimension of input, number of heads 
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
        # hidden_dim -> pf_dim 
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim, pf_dim, dropout_ratio)
        self.dropout = nn.Dropout(dropout_ratio)
    
    # encoder layer have input of src and padding mask(to ignore the impact of pad tokens)
    # encoder layer have both src and padding_mask
    def forward(self, src, padding_mask):
        
        # input: query, key, value 
        # output attentioned query (_src), attention 
        # _src.shape: (batch_size, src_len, hidden_dim)
        _src, attention = self.self_attention(src, src, src, padding_mask) 
        # add * layer norm 
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # feedforward network 
        _src = self.positionwise_feedforward(src)
        # after feedforward network, add and norm 
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        # src.shape: [batch_size, src_len, hidden_dim]
        return src 
    
class Encoder(nn.Module):
    def __init__(self, input_features, n_layers, n_heads, pf_dim, dropout_ratio, device, max_length):
        super().__init__()
        
        self.device = device 
        
        # when training sequence of embedding, input is sequence of embedding, not the index of word 
        # so, we don't need token embedding 
        # size of embedding is fixed = size of input embedding 
        # max_length = max length of sequence 
        self.pos_embedding = nn.Embedding(max_length, input_features)
        
        self.layers = nn.ModuleList([EncoderLayer(input_features, n_heads, pf_dim, dropout_ratio, device) for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout_ratio)
        self.scale = torch.sqrt(torch.FloatTensor([input_features])).to(device)
        
    # shape of src: (batch_size, seq_length(padded), # of features)
    # shape of padding_mask: (batch_size, seq_length(padded))
    # padding_mask shows that each item is pad_token or not. 
    def forward(self, src, padding_mask):
            
        batch_size = src.shape[0]
        # length of sequence (after padding, max_seq_length)
        src_len = src.shape[1]
        src_feature = src.shape[2]
        
        # pos.shape: [batch_size, src_len]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = src.to(self.device)
        
        # batch_size: 128, max_seq_length: 512, hidden_dim: 128 
        #t = self.tok_embedding(src) * self.scale
        #pos_embed = self.pos_embedding(pos)
        #print(t.shape, pos_embed.shape)
        #src = self.dropout((self.tok_embedding(src)* self.scale) + self.pos_embedding(pos)).to(self.device)
        
        # add position embedding to src(sequence of embedding)
        src = self.dropout((src* self.scale + self.pos_embedding(pos).to(self.device)))
        
        # src.shape: [batch_size, max_length, hidden_dim]
        # embedding is continuously updated 
        # give padding mask with src when using layers 
        for layer in self.layers:
            src = layer(src, padding_mask)
        
        # expected: (batch_size, seq_length, # feature)
        #print(src.shape)
        #print(src)
        cls = src[:, 0, :]
        end = src[:, -1, :]
        #print(cls.shape)
         
        #cls = cls.unsqueeze(dim = 1)
        #end = end.unsqueeze(dim = 1)        
        '''# src.shape(batch_size, src_len, hidden_dim)
        # if only return first token of embedding,
        if self.return_cls == True:
            # first token for each sequence in batch (size:512)
            # src.shape = (batch_size, 1, hidden_dim)
            cls = cls.unsqueeze(dim=1)
            end = end.unsqueeze(dim=1)'''                      
        return cls, end 
    
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hidden_dim, pf_dim, dropout_ratio):
        super().__init__()
        
        self.fc_1 = nn.Linear(hidden_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout_ratio)
        
    def forward(self, x):
        
        # input x.shape: [batch_size, q_len, hidden_dim]
        # output x.shape: [batch_size, q_len, pf_dim]
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        # input x.shape: [batch_size, q_len, pf_dim]
        # output x.shape: [batch_size, q_len, hidden_dim]
        x = self.fc_2(x)

        return x     
    
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()
        
        assert hidden_dim % n_heads == 0
        
        self.hidden_dim = hidden_dim 
        self.n_heads = n_heads 
        self.head_dim = hidden_dim // n_heads 
        
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout_ratio)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
                
    def forward(self, query, key, value, padding_mask = None):
        
        batch_size = query.shape[0]
        
        # size of Q, K, V: (batch_size, q_len, # of feature)
        # the dtype of nn.Linear is torch.float32 
        # the dtype of query, key, value should be torch.float32
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        # 1. change size of Q, K, V to (batch_size, q_len, n_heads, dim_head)
        # 2. permute the size of tensor (batch_size, n_heads, q_len, dim_head)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) 
            
        # for K, change the size of tensor into (batch_size, n_heads, dim_head, seq_len)
        # energy.shape: (batch_size, n_heads, q_len, k_len) 
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale 
        #print(energy)
        #print(energy.shape)
        
        if padding_mask is not None:
            energy = energy.masked_fill(padding_mask == 1, -1e10)
        #print("-" *30)    
        #print(energy)
        
        #print("-" *30)
        #print(padding_mask.shape)
        #print(padding_mask)
        # if mask is not none, convert the index to -1e10 where mask is 1(mask == 1 means the embedding is padded embedding)
        # change energy value into probability 
        # attention.shape: (batch_size, n_heads, q_len, k_len)
        # attetion between (query words) and (key words)
        attention = torch.softmax(energy, dim = -1)
        
        # add attention to original query 
        # k_len is equal to q_len
        # attention.shape: (batch_size, n_heads, q_len, q_len)
        # V.shape: (batch_size, n_heads, q_len, dim_head)
        # x.shape: (batch_size, n_heads, q_len, dim_head)
        x = torch.matmul(self.dropout(attention), V)
        
        # x.shape: (batch_size, q_len, n_heads, dim_heads)
        x = x.permute(0, 2, 1, 3).contiguous()
        
        # x.shape: (batch_size, q_len, hidden_dim)
        # n_heads * dim_heads = hidden_dim 
        x = x.view(batch_size, -1, self.hidden_dim)
        
        x = self.fc_o(x)
        
        # x.shape: (batch_size, q_len, hidden_dim) (same as input shape)
        # attention.shape: (batch_size, n_heads, q_len, k_len)
        return x, attention
    
class TransformerEncoder(nn.Module):
    
    def __init__(self, hidden_dim, num_labels, encoder, device):
        super().__init__()
        
        self.encoder = encoder 
        self.device = device  
        self.hidden_dim = hidden_dim 
        self.num_labels = num_labels
        # input of classifier: embedding from transformer (embedding which contain relation between workflows)
        # input.shape: [batch_size, hidden_dim] (only input the first token hidden dimension)
        # output_shape: [batch_size, num_labels]
        self.classifier = nn.Linear(hidden_dim, num_labels)
        # size of padding_index = (batch_size, seq_length)
        # if padding_index[][i] == 1, the embedding is padded embedding 

    # make padding_mask (after make padding_mask, send it to encoder)
    # size of src: (batch_size, seq_length, # of feature)
    # padded_index : (batch_size, seq_length) -> if padded then 1, else 0
    def forward(self, src, padded_index): 
        
        padding_mask = self.make_padding_mask_selfattention(src, padded_index)
        
        # embedding (output of encoder)
        # enc_src.shape: [batch_size, max_seq_length, hidden_dim]
        src_cls, src_end = self.encoder(src, padding_mask) 
        # enc_src_cls: hidden_dim for first token (e.g [128, 64, 256] -> get the embedding of first token -> [32, 256])
        
        # label (conf type) of this input 
        # label.shape: [batch_size, num_labels]
        label = self.classifier(src_cls)

        # enc_src.shape: [batch_size, max_seq_length, hidden_dim]
        # label.shape: [batch_size, num_labels]
        return src_cls, label 
    
    def make_padding_mask_selfattention(self, q, padded_index):
        
        # size of q, k = (batch_size, seq_length, # of feature)
        _, q_len, _ = q.size()
        #_, k_len, _ = k.size()

        # size of padded_index: (batch_size, seq_length)
        padded_index = padded_index.unsqueeze(1).unsqueeze(3)
        padded_index = padded_index.repeat(1, 1, 1, q_len)

        mask = padded_index & padded_index 
        return mask
        
class LinearClassifier(nn.Module):
    
    def __init__(self, hidden_dim, num_labels):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(hidden_dim, num_labels)
        
    def forward(self, features):
        return self.fc(features)