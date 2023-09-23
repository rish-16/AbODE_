import pickle, torch
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

    atom_features = {}
    ring_info = mol.GetRingInfo()
    morgan_fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=size, includeChirality=True
    )
    fingerprint_feature_names = [f"fp{i}" for i in range(size)]

    if use_peptide_stereo:
        residues = peptides.get_residues(
            mol, residues_in_mol=residues_in_mol, macrocycle_idxs=macrocycle_idxs
        )
        atom_to_residue = {
            atom_idx: symbol for atom_idxs, symbol in residues.items() for atom_idx in atom_idxs
        }

        # pprint (atom_to_residue)

    """        
    conformers = mol.GetConformers()
    print (len(conformers))

    print (mol.GetNumAtoms())
    for atom, pos in zip(mol.GetAtoms(), conformers[0].GetPositions()):
        print (atom.GetSymbol(), pos)    
    """

    all_conformers = mol.GetConformers()
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

        bb_coords_concat = np.concatenate([bb_pos_n, bb_pos_ca, bb_pos_c], axis=1) # (x_N, x_Ca, x_C)

        all_conformer_coords.append(bb_coords_concat)

    all_conformer_coords = np.concatenate(all_conformer_coords, axis=0)
    all_conformer_coords = torch.from_numpy(all_conformer_coords)
    print (all_conformer_coords.shape)

    for atom_idx in macrocycle_idxs:
        atom_feature_dict = {}
        atom = mol.GetAtomWithIdx(atom_idx)

        atomic_num_onehot = one_k_encoding(atom.GetAtomicNum(), ATOMIC_NUMS)
        atom_feature_dict.update(dict(zip(ATOMIC_NUM_FEATURE_NAMES, atomic_num_onehot)))

        # chiral_feature = CHIRAL_TAGS[atom.GetChiralTag()]
        # if use_peptide_stereo:
        #     # Only label an atom with the residue L/D tag if the atom is a chiral center
        #     if chiral_feature != 0:
        #         chiral_feature = PEPTIDE_CHIRAL_TAGS[
        #             peptides.get_amino_acid_stereo(atom_to_residue[atom_idx])
        #         ]
        # atom_feature_dict[CHIRAL_TAG_FEATURE_NAME] = chiral_feature

        # atom_feature_dict[AROMATICITY_FEATURE_NAME] = 1 if atom.GetIsAromatic() else 0

        # hybridization_onehot = one_k_encoding(atom.GetHybridization(), HYBRIDIZATION_TYPES)
        # atom_feature_dict.update(dict(zip(HYBRIDIZATION_TYPE_FEATURE_NAMES, hybridization_onehot)))

        # degree_onehot = one_k_encoding(atom.GetTotalDegree(), DEGREES)
        # atom_feature_dict.update(dict(zip(DEGREE_FEATURE_NAMES, degree_onehot)))

        # valence_onehot = one_k_encoding(atom.GetTotalValence(), VALENCES)
        # atom_feature_dict.update(dict(zip(VALENCE_FEATURE_NAMES, valence_onehot)))

        # num_hydrogen_onehot = one_k_encoding(
        #     atom.GetTotalNumHs(includeNeighbors=True), NUM_HYDROGENS
        # )
        # atom_feature_dict.update(dict(zip(NUM_HYDROGEN_FEATURE_NAMES, num_hydrogen_onehot)))

        # charge_onehot = one_k_encoding(atom.GetFormalCharge(), FORMAL_CHARGES)
        # atom_feature_dict.update(dict(zip(FORMAL_CHARGE_FEATURE_NAMES, charge_onehot)))

        # in_ring_sizes = [int(ring_info.IsAtomInRingOfSize(atom_idx, size)) for size in RING_SIZES]
        # atom_feature_dict.update(dict(zip(RING_SIZE_FEATURE_NAMES, in_ring_sizes)))

        # num_rings_onehot = one_k_encoding(int(ring_info.NumAtomRings(atom_idx)), NUM_RINGS)
        # atom_feature_dict.update(dict(zip(NUM_RING_FEATURE_NAMES, num_rings_onehot)))

        # if include_side_chain_fingerprint:
        #     # Fingerprint includes atom in ring that side chain starts at
        #     side_chain_idxs = chem.dfs(atom_idx, mol, blocked_idxs=macrocycle_idxs)
        #     fingerprint = morgan_fingerprint_generator.GetCountFingerprintAsNumPy(
        #         mol, fromAtoms=side_chain_idxs
        #     )
        #     fingerprint = np.asarray(fingerprint.astype(np.int64), dtype=int)
        #     atom_feature_dict.update(dict(zip(fingerprint_feature_names, fingerprint)))

        atom_features[atom_idx] = atom_feature_dict

    # atom_features = pd.DataFrame(atom_features).T
    # atom_features.index.name = "atom_idx"

    return atom_features


def featurize_macrocycle_atoms_from_file(
    path: Union[str, Path],
    use_peptide_stereo: bool = True,
    residues_in_mol: Optional[List[str]] = None,
    include_side_chain_fingerprint: bool = True,
    radius: int = 3,
    size: int = 2048,
    return_mol: bool = False,
) -> Union[pd.DataFrame, Tuple[Chem.Mol, pd.DataFrame]]:
    
    with open(path, "rb") as f:
        if "?" not in path:
            ensemble_data = pickle.load(f)
        else:
            ensemble_data = None

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
