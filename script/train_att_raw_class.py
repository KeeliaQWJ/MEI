import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import math
import logging
import time
import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, f1_score
from utils import *
from module_atteneion import MEI
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb=256"

def load_data(file_path):
    df = pd.read_pickle(file_path)
    enzyme_embeddings = df['emb_test'].apply(lambda x: torch.tensor(x)).tolist()
    mol_embeddings = df['mol_embeddings'].apply(lambda x: torch.tensor(x)).tolist()
    print(mol_embeddings.shape)
    labels = df['label'].apply(lambda x: torch.tensor(x)).tolist()
    
    return df, mol_embeddings, enzyme_embeddings, labels

def val(model, criterion, test_mol_embeddings, test_enzyme_embeddings, test_labels, device, batch_size=128):
    
    model.eval()
    running_loss = AverageMeter()
    test_iterator = range(0, len(test_mol_embeddings), batch_size)
    for batch_start in test_iterator:
    
        batch_mol_embeddings = test_mol_embeddings[batch_start:batch_start + batch_size].to(device)
        batch_enzyme_embeddings = test_enzyme_embeddings[batch_start:batch_start + batch_size].to(device)
        batch_labels = test_labels[batch_start:batch_start + batch_size].to(device)
        batch_labels = batch_labels.float()
        with torch.no_grad():
            pred = model(batch_mol_embeddings, batch_enzyme_embeddings)
            loss = criterion(pred, batch_labels)
            # loss = custom_loss(pred, test_labels)
            label = batch_labels
            running_loss.update(loss.item(), label.size(0))

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()

    return epoch_loss

class BasicLogger(object):
    def __init__(self, path):
        self.logger = logging.getLogger(path)
        self.logger.setLevel(logging.DEBUG)
        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                        "%Y-%m-%d %H:%M:%S")

        if not self.logger.handlers:
            file_handler = logging.FileHandler(path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)

            print_handler = logging.StreamHandler()
            print_handler.setLevel(logging.DEBUG)
            print_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(print_handler)

    def noteset(self, message):
        self.logger.noteset(message)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

def create_dir(dir_list):
    assert  isinstance(dir_list, list) == True
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)

class TrainLogger(BasicLogger):
    def __init__(self, args):
        self.args = args

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if args.get('mark') == None:
            savetag = timestamp + '_' + args.get('dataset')

        save_dir = args.get('save_dir')
        if save_dir == None:
            raise Exception('save_dir can not be None!')
        train_save_dir = os.path.join(save_dir, savetag)
        self.log_dir = os.path.join(train_save_dir, 'log', 'train')
        self.model_dir = os.path.join(train_save_dir, 'model')
        create_dir([self.log_dir, self.model_dir])

        print(self.log_dir)

        log_path = os.path.join(self.log_dir, 'Train.log')
        super().__init__(log_path)

    def get_log_dir(self):
        if hasattr(self, 'log_dir'):
            return self.log_dir
        else:
            return None

    def get_model_dir(self):
        if hasattr(self, 'model_dir'):
            return self.model_dir
        else:
            return None

def evaluate_model(model, test_mol_embeddings, test_enzyme_embeddings, test_labels, device):
    model.eval()
    
    with torch.no_grad():
        pred = model(test_mol_embeddings, test_enzyme_embeddings)
        pred = pred.cpu()
        binary_pred = np.argmax(pred.detach().numpy(), axis=1)
        
        binary_labels = np.argmax(test_labels, axis=1)
        
        acc = accuracy_score(binary_labels, binary_pred)
        mcc = matthews_corrcoef(binary_labels, binary_pred)
        roc_auc = roc_auc_score(binary_labels, pred[:, 1])
        f1 = f1_score(binary_labels, binary_pred)
        
        return acc, mcc, roc_auc, f1


def main():
    device = torch.device('cuda:0')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_file', type=str, default='mei_data/final/mei_train_mol.pkl', help='path to data file')
    parser.add_argument('--save_model', default=True, action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch_size')
    args = parser.parse_args()

    params = dict(
        data_root="data",
        save_dir="save",
        dataset=args.data_file,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size
    )
    batch_size = args.batch_size
    logger = TrainLogger(params)
    logger.info(__file__)

    df, mol_embeddings, enzyme_embeddings, labels = load_data(args.data_file)
    train_df, test_df, train_mol_embeddings, test_mol_embeddings, train_enzyme_embeddings, test_enzyme_embeddings, train_labels, test_labels = train_test_split(
        df, mol_embeddings, enzyme_embeddings, labels, test_size=0.1, random_state=42)
    
    train_mol_embeddings = torch.tensor(train_mol_embeddings, dtype=torch.float32).to(device)
    train_enzyme_embeddings = torch.tensor(train_enzyme_embeddings, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
    train_labels = train_labels.float()
    train_labels=torch.eye(2)[train_labels.long(), :]

    test_mol_embeddings = torch.tensor(test_mol_embeddings, dtype=torch.float32).to(device)
    test_labels = test_labels.to(torch.long).to(device)
    test_labels = test_labels.float()
    test_enzyme_embeddings = torch.tensor(test_enzyme_embeddings, dtype=torch.float32).to(device)
    test_labels=torch.eye(2)[test_labels.long(), :]
    model = MEI().to(device)
    running_best_acc = BestMeter('max')
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_iterator = range(len(train_mol_embeddings))

    epochs = 10000
    steps_per_epoch = 2000
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_df))
    break_flag = False

    global_step = 0
    global_epoch = 0
    early_stop_epoch = 200

    running_loss = AverageMeter()
    running_best_mse = BestMeter("min")
    # running_best_acc = BestMeter("max")

    model.train()

    batch_size = 1024

    train_iterator = range(0, len(train_mol_embeddings), batch_size)

    for i in range(num_iter):
        if break_flag:
            break

        for batch_start in train_iterator:
            batch_mol_embeddings = train_mol_embeddings[batch_start:batch_start + batch_size].to(device)
            batch_enzyme_embeddings = train_enzyme_embeddings[batch_start:batch_start + batch_size].to(device)
            batch_labels = train_labels[batch_start:batch_start + batch_size].to(device)
            batch_labels = batch_labels.float()

            global_step += 1
            pred = model(batch_mol_embeddings, batch_enzyme_embeddings)
            # pred = pred.squeeze(1)
            loss = criterion(pred, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item(), batch_labels.size(0)) 
            if global_step % steps_per_epoch == 0:
                global_epoch += 1

                epoch_loss = running_loss.get_average()
                running_loss.reset()

                test_loss = val(model, criterion, test_mol_embeddings, test_enzyme_embeddings, test_labels, device)
                test_accuracy, test_mcc, test_aucroc, test_f1 = evaluate_model(model, test_mol_embeddings, test_enzyme_embeddings, test_labels, device)

                msg = "epoch-%d, loss-%.4f, test_loss-%.4f, accuracy-%.4f, mcc-%.4f, aucroc-%.4f, f1-%.4f" % (global_epoch, epoch_loss, test_loss, test_accuracy, test_mcc, test_aucroc, test_f1)
                print(msg)
                logger.info(msg)

                if test_loss < running_best_mse.get_best():
                    running_best_mse.update(test_loss)
                    save_model_dict(model, logger.get_model_dir(), msg)
                
                else:
                    count = running_best_mse.counter() 
                    if count > early_stop_epoch:
                        logger.info(f"early stop in epoch {global_epoch}")
                        break_flag = True
                        break

if __name__ == "__main__":
    main()

