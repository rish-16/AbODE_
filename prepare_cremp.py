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
pdb_sequences = os.listdir(CREMP_PATH) 

for pdb in pdb_sequences:
    fp = CREMP_PATH + pdb
    residues_in_mol = [aa.strip("[]") for aa in pdb.replace("Sar", "MeG").split(".")["-1"]] # ignore 'pickle' at the end
    print (residues_in_mol)
    features = featurise_cremp.featurize_macrocycle_atoms_from_file(path=fp, residues_in_mol=residues_in_mol) # return_mol = False
    print (features)

    break
