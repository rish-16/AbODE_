import torch

import math, random, sys, argparse, os, json, csv, time
import numpy as np
import MDAnalysis as mda
from pprint import pprint
from tqdm import tqdm
import pickle as pkl
from rdkit import Chem

import peptide_utils
import featurise_cremp

CREMP_PATH = "/data/rishabh/pickle/"
pdb_sequences = OS.listdir(CREMP_PATH) 

for pdb in pdb_sequences:
    fp = CREMP_PATH + pdb
    features = featurise_cremp.featurize_macrocycle_atoms_from_file(path=fp) # return_mol = False
    print (features)