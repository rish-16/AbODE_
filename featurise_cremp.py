import torch
import pickle as pkl
from pprint import pprint
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdchem import ChiralType, HybridizationType

from cremp_extra import chem, peptides

ATOMIC_NUMS = list(range(1, 100))
PEPTIDE_CHIRAL_TAGS = {
    "L": 1,
    "D": -1,
    None: 0,
}
CHIRAL_TAGS = {
    ChiralType.CHI_TETRAHEDRAL_CW: -1,
    ChiralType.CHI_TETRAHEDRAL_CCW: 1,
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_OTHER: 0,
}
HYBRIDIZATION_TYPES = [
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
]
DEGREES = [0, 1, 2, 3, 4, 5]
VALENCES = [0, 1, 2, 3, 4, 5, 6]
NUM_HYDROGENS = [0, 1, 2, 3, 4]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
RING_SIZES = [3, 4, 5, 6, 7, 8]
NUM_RINGS = [0, 1, 2, 3]

ATOMIC_NUM_FEATURE_NAMES = [f"anum{anum}" for anum in ATOMIC_NUMS] + ["anumUNK"]
CHIRAL_TAG_FEATURE_NAME = "chiraltag"
AROMATICITY_FEATURE_NAME = "aromatic"
HYBRIDIZATION_TYPE_FEATURE_NAMES = [f"hybrid{ht}" for ht in HYBRIDIZATION_TYPES] + ["hybridUNK"]
DEGREE_FEATURE_NAMES = [f"degree{d}" for d in DEGREES] + ["degreeUNK"]
VALENCE_FEATURE_NAMES = [f"valence{v}" for v in VALENCES] + ["valenceUNK"]
NUM_HYDROGEN_FEATURE_NAMES = [f"numh{nh}" for nh in NUM_HYDROGENS] + ["numhUNK"]
FORMAL_CHARGE_FEATURE_NAMES = [f"charge{c}" for c in FORMAL_CHARGES] + ["chargeUNK"]
RING_SIZE_FEATURE_NAMES = [f"ringsize{rs}" for rs in RING_SIZES]  # Don't need "unknown" name
NUM_RING_FEATURE_NAMES = [f"numring{nr}" for nr in NUM_RINGS] + ["numringUNK"]

AMINO_ACID_RESNAMES = ['A', 'K', 'S', 'E', 'MeI', 'Mei', 'l', 'Mev', 'MeV', 'W', 'MeW', 'Mey', 'L', 'c', 'MeQ', 'MeN', 'q', 'MeC', 'Mef', 'Y', 'MeY', 'a', 'i', 'MeF', 'Meq', 'w', 'p', 'G', 'D', 'Met', 'Mew', 'N', 'Mec', 'Mes', 'Mel', 'P', 'F', 'I', 'MeL', 'f', 'Q', 'T', 'MeS', 'n', 'MeA', 'y', 'MeG', 'MeT', 's', 'Men', 't', 'C', 'V', 'H']
res_mapper = {sym : i for i, sym in enumerate(AMINO_ACID_RESNAMES)}

def onehot_encoder(sym, mapper):
    vec = [0 for _ in range(len(mapper) + 1)] # extra +1 for unknown element
    if sym not in mapper:
        vec[-1] = 1 # set final component as 1 if unknown element
    else:
        idx = mapper[sym]
        vec[idx] = 1
    return vec

def one_k_encoding(value: Any, choices: List[Any], include_unknown: bool = True) -> List[int]:
    """Create a one-hot encoding with an extra category for uncommon values.

    Args:
        value: The value for which the encoding should be one.
        choices: A list of possible values.
        include_unknown: Add the extra category for uncommon values.

    Returns:
        A one-hot encoding of the `value` in a list of length len(`choices`) + 1.
        If `value` is not in `choices, then the final element in the encoding is 1
        (if `include_unknown` is True).
    """
    encoding = [0] * (len(choices) + include_unknown)
    try:
        idx = choices.index(value)
    except ValueError:
        if include_unknown:
            idx = -1
        else:
            raise ValueError(
                f"Cannot encode '{value}' because it is not in the list of possible values {choices}"
            )
    encoding[idx] = 1

    return encoding

def featurize_macrocycle_atoms(
    mol: Chem.Mol,
    macrocycle_idxs: Optional[List[int]] = None,
    use_peptide_stereo: bool = True,
    residues_in_mol: Optional[List[str]] = None,
    include_side_chain_fingerprint: bool = True,
    radius: int = 3,
    size: int = 2048,
) -> pd.DataFrame:
    """Create a sequence of features for each atom in `macrocycle_idxs`.

    Args:
        mol: Macrocycle molecule.
        macrocycle_idxs: Atom indices for atoms in the macrocycle.
        use_peptide_stereo: Use L/D chiral tags instead of RDKit tags.
        residues_in_mol: Residues the mol is composed of. Speeds up determining L/D tags.
        include_side_chain_fingerprint: Add Morgan count fingerprints.
        radius: Morgan fingerprint radius.
        size: Morgan fingerprint size.

    Returns:
        DataFrame where each row is an atom in the macrocycle and each column is a feature.
    """

    # mol = Chem.RemoveHs(mol)

    if macrocycle_idxs is None:
        macrocycle_idxs = chem.get_macrocycle_idxs(mol, n_to_c=True)
        if macrocycle_idxs is None:
            raise ValueError(
                f"Couldn't get macrocycle indices for '{Chem.MolToSmiles(Chem.RemoveHs(mol))}'"
            )

    assert len(macrocycle_idxs) % 3 == 0

    BACKBONE_ATOM_LABELS = ["N", "Calpha", "CO"]
    BACKBONE_ATOM_LABELS_ = ["N", "C", "C"]
    BACKBONE_ATOM_IDS = [0, 1, 2]
    bb_reps = len(macrocycle_idxs) // 3
    backbone_atom_labels = BACKBONE_ATOM_LABELS * bb_reps
    backbone_atom_labels_ = BACKBONE_ATOM_LABELS_ * bb_reps
    backbone_atom_ids = BACKBONE_ATOM_IDS * bb_reps 

    only_backbone_atoms = np.array(list(mol.GetAtoms()))[macrocycle_idxs].tolist() # grouped into triplets of (N, Ca, C)

    if use_peptide_stereo:
        residues = peptides.get_residues(
            mol, residues_in_mol=residues_in_mol, macrocycle_idxs=macrocycle_idxs
        )
        atom_to_residue = {
            atom_idx: symbol for atom_idxs, symbol in residues.items() for atom_idx in atom_idxs
        }

    """        
    conformers = mol.GetConformers()
    print (len(conformers))

    print (mol.GetNumAtoms())
    for atom, pos in zip(mol.GetAtoms(), conformers[0].GetPositions()):
        print (atom.GetSymbol(), pos)    
    """

    res_ohe = [onehot_encoder(res, res_mapper) for res in residues_in_mol]

    all_conformers = mol.GetConformers()
    # print ("Num conformers:", len(all_conformers))
    all_conformer_coords = []
    shape_corrector = lambda x,i : x[i:][::3]
    for cix, conformer in enumerate(all_conformers):
        only_backbone_positions = np.array(list(conformer.GetPositions()))[macrocycle_idxs] # grouped into triplets of (N, Ca, C)
        # print (only_backbone_positions.shape)
        # for aid, pos in zip(only_backbone_atoms, only_backbone_positions):
            # print (aid.GetSymbol(), pos)
        bb_pos_n = shape_corrector(only_backbone_positions, 0)
        bb_pos_ca = shape_corrector(only_backbone_positions, 1)
        bb_pos_c = shape_corrector(only_backbone_positions, 2)

        # bb_coords_concat = np.concatenate([bb_pos_n, bb_pos_ca, bb_pos_c], axis=1) # (x_N, x_Ca, x_C)

        all_conformer_coords.append([bb_pos_n, bb_pos_ca, bb_pos_c])

        if cix > 5:
            break

    return res_ohe, all_conformer_coords


def featurize_macrocycle_atoms_from_file(
    path,
    use_peptide_stereo = True,
    residues_in_mol = None,
    include_side_chain_fingerprint = True,
    radius = 3,
    size = 2048,
    return_mol = False,
):
    
    with open(path, "rb") as f:
        ensemble_data = pkl.load(f)

    print (ensemble_data)
    if ensemble_data:
        mol = ensemble_data["rd_mol"]

        features = featurize_macrocycle_atoms(
            mol,
            use_peptide_stereo=use_peptide_stereo,
            residues_in_mol=residues_in_mol,
            include_side_chain_fingerprint=include_side_chain_fingerprint,
            radius=radius,
            size=size,
        )

        if return_mol:
            return mol, features
        return features
    else:
        if return_mol:
            return None, None
        return None
