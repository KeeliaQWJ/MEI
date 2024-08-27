import csv
import random
import os
import math
from re import L
import torch
import numpy as np
import subprocess
import tqdm
import pickle
import torch.nn.functional as F
from clean.src.CLEAN.distance_map import get_dist_map
from clean.src.CLEAN.utils import * 
from clean.src.CLEAN.model import LayerNormNet
from clean.src.CLEAN.distance_map import *
from clean.src.CLEAN.evaluate import *
import pandas as pd
import warnings

def format_esm(a):
    if type(a) == dict:
        a = a['mean_representations'][33]
    return a

def load_esm(lookup):
    esm = format_esm(torch.load('./data/esm_data/' + lookup + '.pt'))
    return esm.unsqueeze(0)

def retrive_esm1b_embedding(fasta_name):
    esm_script = "app/esm/scripts/extract.py"
    esm_out = "app/data/esm_data"
    esm_type = "esm1b_t33_650M_UR50S"
    fasta_name = "app/data/datasets/" + fasta_name + ".fasta"
    command = ["python", esm_script, esm_type, 
              fasta_name, esm_out, "--include", "mean"]
    subprocess.run(command)

def esm_embedding(ec_id_dict, device, dtype):
    '''
    Loading esm embedding in the sequence of EC numbers
    prepare for calculating cluster center by EC
    '''
    esm_emb = []
    # for ec in tqdm(list(ec_id_dict.keys())):
    for ec in list(ec_id_dict.keys()):
        ids_for_query = list(ec_id_dict[ec])
        esm_to_cat = [load_esm(id) for id in ids_for_query]
        esm_emb = esm_emb + esm_to_cat
    return torch.cat(esm_emb).to(device=device, dtype=dtype)

retrive_esm1b_embedding("enzyme_nondup")