import MDAnalysis as mda
import numpy as np
import os
from rdkit.Chem import Descriptors3D
import pickle as pkl

# PATH = "generated_cremp_peptides_lossv3/"
# contents = os.listdir(PATH)

# # for pdb_path in contents:
#     # fp = PATH + pdb_path
#     # uni = mda.Universe(fp, format="PDB")
#     # rg = uni.atoms.radius_of_gyration()
#     # print (rg)

# # RadiusOfGyration
# with open("P.Men.A.MeW.pickle", "rb") as f:
#     mol = pkl.load(f)['rd_mol']
#     rg = Descriptors3D.RadiusOfGyration(mol)
#     print (rg)

cremp_data = torch.load("cremp_data_ca_only.pt")
print ("Loaded dataset ...")
n_instances = len(cremp_data)
train_size = int(0.8 * n_instances)
peptide_data_train, peptide_data_test = cremp_data[:30000], cremp_data[train_size:][:70] # test size of 50 peptides

rog_ca = {
    4: [],
    5: [],
    6: []
}
for peptide in peptide_data_train:
    carbon_ones = torch.ones(x.size(0), 1)
    pooled_carbon_ones = pooled_carbon_ones.view(-1,1).item()
    coords = peptide.y[:, 55:58]
    N_atoms = int(peptide.x.size(0))
    rog = radgyr(coords, N_atoms)

    if N_atoms in rog:
        rog[N_atoms].append(rog)
    else:
        rog[N_atoms] = [rog]

for key, val in rog_ca:
    val = np.array(val).reshape(-1, 1).mean(axis=0)[0]
    print (key, val)