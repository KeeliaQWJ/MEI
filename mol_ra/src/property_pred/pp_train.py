import torch
import pickle
from model import GNN
import pandas as pd
from dgl.dataloading import GraphDataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.neural_network import MLPClassifier
import numpy as np

def train(args, data):
    path = '../saved_3/' + args.pretrained_model + '/'
    print('loading hyperparameters of pretrained model from ' + path + 'hparams.pkl')
    with open(path + 'hparams.pkl', 'rb') as f:
        hparams = pickle.load(f)

    print('loading pretrained model from ' + path + 'model.pt')
    mole = GNN(hparams['gnn'], hparams['layer'], hparams['feature_len'], hparams['dim'])
    if torch.cuda.is_available():
        mole.load_state_dict(torch.load(path + 'model.pt'))
        mole = mole.cuda(args.gpu)
    else:
        mole.load_state_dict(torch.load(path + 'model.pt', map_location=torch.device('cpu')))

    dataloader = GraphDataLoader(data, batch_size=args.batch_size, shuffle=False)
    all_features = []
    all_labels = []
    with torch.no_grad():
        mole.eval()
        for graphs, labels in dataloader:
            graph_embeddings = mole(graphs)
            all_features.append(graph_embeddings)
            all_labels.append(labels)
        all_features = torch.cat(all_features, dim=0).cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

    print('splitting dataset')
    all_features_shape = np.array(all_features).shape
    #all_features.pd.to_pickle('df_features.pkl')
    print(all_features_shape) # (11050, 1024)
    s = pd.Series(all_features.tolist()) # Series是pandas中的一种数据结构，可以将一维数组转换为DataFrame中的一列
    # 创建一个空的DataFrame
    df = pd.DataFrame()
     # 将Series添加到DataFrame中
    df['smi_features'] = s
    df.to_pickle("df_train_with_ESM1b_GCN.pkl")
    df.to_csv('esp_features_gcn.csv', index=False)

    train_features = all_features[: int(0.8 * len(data))]
    train_labels = all_labels[: int(0.8 * len(data))]
    valid_features = all_features[int(0.8 * len(data)): int(0.9 * len(data))]
    valid_labels = all_labels[int(0.8 * len(data)): int(0.9 * len(data))]
    test_features = all_features[int(0.9 * len(data)):]
    test_labels = all_labels[int(0.9 * len(data)):]

    print('training the classification model\n')
    pred_model = MLPClassifier(hidden_layer_sizes=(1024, 256, 2), max_iter=1000, learning_rate_init=0.0001) #, warm_start=True
    early_stopping = EarlyStopping(patience=20)

    for epoch in range(30):
        pred_model.fit(train_features, train_labels)
        train_proba = pred_model.predict_proba(train_features)
        valid_proba = pred_model.predict_proba(valid_features)
        train_auc = roc_auc_score(train_labels, train_proba[:, 1])
        valid_auc = roc_auc_score(valid_labels, valid_proba[:, 1])
        print(f"Epoch {epoch + 1}: Train AUROC: {train_auc}, Valid AUROC: {valid_auc}")

        if early_stopping.step(pred_model, valid_auc):
            break

    best_model = early_stopping.best_model
    run_classification(best_model, 'train', train_features, train_labels)
    run_classification(best_model, 'valid', valid_features, valid_labels)
    run_classification(best_model, 'test', test_features, test_labels)


def run_classification(model, mode, features, labels):
    acc = model.score(features, labels)
    auc = roc_auc_score(labels, model.predict_proba(features)[:, 1])
    mcc = matthews_corrcoef(labels, model.predict(features))
    print('%s acc: %.4f   auc: %.4f   mcc: %.4f' % (mode, acc, auc, mcc))

# 自定义EarlyStopping回调
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_score = -np.inf
        self.counter = 0
        self.best_model = None

    def step(self, model, score):
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            self.best_model = model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


