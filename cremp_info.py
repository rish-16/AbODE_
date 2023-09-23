import os
import matplotlib.pyplot as plt
import numpy as np

CREMP_PATH = "/data/rishabh/pickle/"
pdb_sequences = os.listdir(CREMP_PATH) 

sizes = []

unique_res = set()

for pdb in pdb_sequences[1:]:
    fp = CREMP_PATH + pdb
    residues_in_mol = [aa.strip("[]") for aa in pdb.replace("Sar", "MeG").split(".")[:-1]] # ignore 'pickle' at the end
    sizes.append(len(residues_in_mol))
    for res in residues_in_mol:
        unique_res.add(res)

# fig = plt.figure()
# plt.hist(sizes, color="blue")    
# plt.xlabel("Macrocyclic peptide # residues", fontsize=15)
# plt.ylabel("Frequency", fontsize=15)
# plt.savefig("cremp_protein_sizes.pdf")

print (unique_res)
print (len(unique_res))