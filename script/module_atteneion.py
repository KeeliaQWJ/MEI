import pandas as pd
import pickle
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import argparse

def load_esm(lookup):
    esm = format_esm(torch.load('app/data/esm_data/' + str(lookup) + '.pt'))
    return esm.unsqueeze(0)


# Define the function to format ESM data
def format_esm(a):
    if type(a) == dict:
        a = a['mean_representations'][33]
    return a

def load_mol_emz(file_path):
    df = pd.read_pickle(file_path)
    #df = pd.read_csv(file_path, sep='\t')
    # Load mol_embeddings from pickle
    with open('3k_mol_embeddings.pkl', 'rb') as f:
        mol_embeddings = pickle.load(f)

    # Load enzyme_embeddings for each UniProt_ID
    enzyme_embeddings = []
    valid_indices = []  # 用于存储有效的索引
    for i, id in enumerate(df['UniProt_ID']):
        if pd.notnull(id):  # 检查UniProt_ID是否为NaN
            enzyme_embedding = load_esm(id)
            enzyme_embeddings.append(enzyme_embedding)
            valid_indices.append(i)
    enzyme_embeddings = torch.cat(enzyme_embeddings)

    # 根据有效索引获取相应的mol_embeddings和labels
    mol_embeddings = torch.tensor(mol_embeddings[valid_indices])
    labels = torch.tensor(df['label'].values[valid_indices])

    # Concatenate mol_embeddings and enzyme_embeddings
    # combined_features = torch.cat((mol_embeddings, enzyme_embeddings), dim=1)

    return df, mol_embeddings, enzyme_embeddings, labels


def load_data(file_path):

    df = pd.read_pickle(file_path)
    enzyme_embeddings = torch.tensor(df['emb_test'])
    mol_embeddings = torch.tensor(df['mol_embeddings'])
    labels = torch.tensor(df['label'])
    combined_features = torch.cat((mol_embeddings, enzyme_embeddings), dim=1)

    return df, mol_embeddings, enzyme_embeddings, labels

class smi_encoder(nn.Module):
    def __init__(self, smi_dim, hid_dim, n_layers, kernel_size, dropout=0.2):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size"
        self.input_dim = 128
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        # self.device = torch.device('cuda:0')
        self.scale = torch.sqrt(torch.FloatTensor([0.5]))#.to(torch.device('cuda:0'))
        self.convs = nn.ModuleList(
            [nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)]
        )   # convolutional layers
        self.do = nn.Dropout(0.2)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim)

    '''
    Input: protein (batch_size, sequence_length, input_dim)
    Output: conved (batch_size, sequence_length, hid_dim)
    '''
    def forward(self, smi):
        print(smi.shape)
        smi = smi.permute(0, 2, 1)
        conv_input = self.fc(smi)
        conv_input = conv_input.permute(0, 2, 1)
        for i, conv in enumerate(self.convs):
            conved = conv(self.do(conv_input))
            conved = F.glu(conved, dim=1)
            conved = conved + conv_input * self.scale
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        conved = self.ln(conved)
        return conved


class AttentionBlock(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))#.cuda() #.cuda()

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.f_q(query)
        K = self.f_k(key)
        V = self.f_v(value)
    
        Q = Q.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        K_T = K.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3).transpose(2, 3)
        V = V.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)

        energy = torch.matmul(Q, K_T) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        weighter_matrix = torch.matmul(attention, V)

        weighter_matrix = weighter_matrix.permute(0, 2, 1, 3).contiguous()

        weighter_matrix = weighter_matrix.view(batch_size, self.n_heads * (self.hid_dim // self.n_heads))

        weighter_matrix = self.do(self.fc(weighter_matrix))

        return weighter_matrix


class MEI(torch.nn.Module):
    def __init__(self, n_output=2, projection_dim=256, dropout=0.2):
        super(MEI, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        # 1D convolution on protein sequence
        self.projection_dim = projection_dim
        self.mol_linear = nn.Linear(1024, projection_dim)#.to(device) 1024
        self.enzyme_linear = nn.Linear(128, projection_dim)#.to(device) 128
        
        self.fc1_xt = nn.Linear(64, projection_dim)
        self.ln = nn.LayerNorm(projection_dim)
        # cross attention
        self.att = AttentionBlock(projection_dim, 1, 0.1)
        self.enzyme_linear_proj = nn.Linear(128, projection_dim)  # 128维调整为1024维
        self.att_block = AttentionBlock(projection_dim, 1, 0.1)  # 使用调整后的维度
        self.ln2 = nn.LayerNorm(projection_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(768, 2560), 
            nn.ReLU(),
            nn.Dropout(),
            
            nn.Linear(2560, 2560),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(2560, 1024),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 2)
        )


    def forward(self, mol_embeddings, enzyme_embeddings):
        mol_projection = self.mol_linear(mol_embeddings)
        enzyme_projection = self.enzyme_linear(enzyme_embeddings) #enzyme_linear_proj
        
        att = self.att(mol_projection, enzyme_projection, enzyme_projection)
        combined_embeddings = mol_projection + att
        combined_embeddings = self.ln2(combined_embeddings)
        feature = torch.cat((mol_projection, combined_embeddings, enzyme_projection), dim=1)
        output = self.fc(feature)

        return output
    
    def extract_features(self, mol_embeddings, enzyme_embeddings):
        mol_projection = self.mol_linear(mol_embeddings)
        enzyme_projection = self.enzyme_linear_proj(enzyme_embeddings)
        att = self.att(mol_projection, enzyme_projection, enzyme_projection)
        combined_embeddings = mol_projection + att
        combined_embeddings = self.ln2(combined_embeddings)
        feature = torch.cat((mol_projection, combined_embeddings, enzyme_projection), dim=1)

        # for layer in self.fc[:-1]:
        #     feature = layer(feature)

        return feature

