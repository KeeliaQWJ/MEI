# import os
# os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
import os
import pandas as pd
import torch
import pickle
import dgl
import pysmiles
from mol_ra.src.model import GNN
from clean.src.CLEAN.model import LayerNormNet
from torch.utils.data import DataLoader

# Set up environment for MKL to use GNU thread layer for compatibility with libgomp.so.1
os.environ["MKL_THREADING_LAYER"] = "GNU"

# Class to process SMILES strings into DGL graphs
class GraphDataset(dgl.data.DGLDataset):
    def __init__(self, path_to_model, smiles_list, gpu):
        self.path = path_to_model
        self.smiles_list = smiles_list
        self.gpu = gpu
        super().__init__(name='graph_dataset')

    def process(self):
        with open(self.path + '/feature_enc.pkl', 'rb') as f:
            feature_encoder = pickle.load(f)
        self.graphs = []
        for i, smiles in enumerate(self.smiles_list):
            try:
                raw_graph = pysmiles.read_smiles(smiles, zero_order_bonds=False)
                dgl_graph = dgl.graph_data.graph_networkx_to_dgl(raw_graph, feature_encoder)
                self.graphs.append(dgl_graph)
            except Exception as e:
                print(f'ERROR: No. {i} smiles is not parsed successfully: {e}')

        if torch.cuda.is_available() and self.gpu is not None:
            self.graphs = [graph.to(f'cuda:{self.gpu}') for graph in self.graphs]

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)

# Function to load model embeddings
def load_model_embeddings(path, ids, device, dtype):
    model = LayerNormNet(512, 128, device, dtype)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    embeddings = [torch.load(f'app/data/esm_data/{id}.pt')['mean_representations'][33].unsqueeze(0) for id in ids]
    return torch.cat(embeddings).to(device=device, dtype=dtype)

# Molecular feature featurizer using a GNN
class MolEFeaturizer:
    def __init__(self, path_to_model, gpu=0):
        self.path_to_model = path_to_model
        self.gpu = gpu
        with open(os.path.join(path_to_model, 'hparams.pkl'), 'rb') as f:
            hparams = pickle.load(f)
        self.mole = GNN(hparams['gnn'], hparams['layer'], hparams['feature_len'], hparams['dim'])

        model_path = os.path.join(path_to_model, 'model.pt')
        if torch.cuda.is_available() and gpu is not None:
            self.mole.load_state_dict(torch.load(model_path))
            self.mole.cuda(gpu)
        else:
            self.mole.load_state_dict(torch.load(model_path, map_location='cpu'))

    def transform(self, smiles_list):
        data = GraphDataset(self.path_to_model, smiles_list, self.gpu)
        dataloader = DataLoader(data, batch_size=32, shuffle=False, num_workers=4)
        embeddings = []

        with torch.no_grad():
            self.mole.eval()
            for graphs in dataloader:
                graph_embeddings = self.mole(graphs)
                embeddings.extend(graph_embeddings.cpu().numpy())

        return embeddings

# Main execution function
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    dtype = torch.float32

    df_path = 'mei_data/final/mei_test.pkl'
    df = pd.read_pickle(df_path)
    print(df.columns)

    model_path = 'app/data/pretrained/split100.pth'
    protein_embeddings = load_model_embeddings(model_path, df['UniProt_ID'], device, dtype)
    smiles_list = df['react'].tolist()  # Assuming SMILES data is in 'SMILES' column
    molecular_embeddings = MolEFeaturizer('MolR/saved_3/gcn_1024', device).transform(smiles_list)

    df['emb_test'] = protein_embeddings.tolist()
    df['molgcn_embeddings'] = molecular_embeddings

    output_path = 'mei_data/final/mei_test.pkl'
    df.to_pickle(output_path)
    print(f"Updated DataFrame with 'emb_test' and 'mol_embeddings' saved to {output_path}")

if __name__ == "__main__":
    main()







