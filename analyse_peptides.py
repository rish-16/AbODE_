import MDAnalysis as mda
import os
from rdkit.Chem import Descriptors3D
import pickle as pkl

PATH = "generated_cremp_peptides_lossv3/"
contents = os.listdir(PATH)

# for pdb_path in contents:
    # fp = PATH + pdb_path
    # uni = mda.Universe(fp, format="PDB")
    # rg = uni.atoms.radius_of_gyration()
    # print (rg)

# RadiusOfGyration
with open("P.Men.A.MeW.pickle", "rb") as f:
    mol = pkl.load(f)['rd_mol']
    rg = Descriptors3D.RadiusOfGyration(mol)
    print (rg)