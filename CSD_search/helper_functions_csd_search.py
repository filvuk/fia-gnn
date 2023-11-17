import random
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def check_bonds(mol):
    """
    Check if an unwanted (hetero atom - hetero atom) bond is in the molecule
    """
    forbidden_bonds = [
        ["O", "O"],
        ["O", "S"],
        ["O", "N"],
        ["S", "S"],
        ["S", "N"],
        ["N", "N"],
        ["O", "F"],
        ["O", "Cl"],
        ["O", "Br"],
        ["O", "I"],
        ["S", "F"],
        ["S", "Cl"],
        ["S", "Br"],
        ["S", "I"],
        ["N", "F"],
        ["N", "Cl"],
        ["N", "Br"],
        ["N", "I"],
    ]
    forbidden_bonds = [sorted(b) for b in forbidden_bonds]

    reject = False
    for bond in mol.GetBonds():
        b = [bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()]
        b.sort()

        if b in forbidden_bonds:
            reject = True
            break

    return reject


def check_neighbors(mol, neighbor_count, central_atom_symbol):
    """
    Check if the neighbor count matches the expected number of neighbors
    """
    reject = False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == central_atom_symbol:
            n = len(atom.GetNeighbors())
            if n != neighbor_count:
                reject = True

    return reject, n


def check_unpaired_electrons(mol, central_atom_symbol):
    """
    Check if unpaired electrons are in the molecule
    """
    message = "pass"
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() != 0:
            if atom.GetSymbol() == central_atom_symbol:
                message = "Unpaired electron(s) at central atom contained."
            else:
                message = "Unpaired electron(s) contained (not at central atom)."
    return message


def check_isotope(mol):
    """
    Check if unwanted isotopes are in the molecule
    """
    reject = False
    for atom in mol.GetAtoms():
        if atom.GetIsotope() != 0:
            reject = True
            break
    return reject


def get_picks(subset, n_picks):
    """
    Pick a subset of molecules from a given set with a MinMax picking strategy
    based on the Tanimoto index calculated from Morgan fingerprints.
    The first pick is randomly chosen.
    """

    def get_tanimoto_similarities(smiles, all_smiles):
        """
        Calculate the Tanimoto similarities of a molecule (provided as SMILES string) with all other molecules
        """
        similarities = []

        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3)

        for smiles_2 in all_smiles:
            mol_2 = Chem.MolFromSmiles(smiles_2)
            mol_2 = Chem.AddHs(mol_2)
            fp_2 = AllChem.GetMorganFingerprintAsBitVect(mol_2, radius=3)

            similarities.append(DataStructs.TanimotoSimilarity(fp, fp_2))

        return similarities

    all_smiles = [subset[identifier]["rdkit_smiles"] for identifier in subset]

    first = random.choice(list(subset.keys()))
    similarities = get_tanimoto_similarities(subset[first]["rdkit_smiles"], all_smiles)
    pick_df = pd.DataFrame(
        {"smiles": all_smiles, first: similarities}, index=list(subset.keys())
    )

    while len(list(pick_df.columns)) - 2 < n_picks:
        pick_df["sum"] = pick_df.sum(axis=1)
        pick = pick_df["sum"].idxmin()
        pick_df = pick_df.drop(pick, axis=0)
        pick_df[pick] = get_tanimoto_similarities(
            subset[pick]["rdkit_smiles"], list(pick_df["smiles"])
        )

    pick_df = pick_df.drop(["sum", "smiles"], axis=1)
    return list(pick_df.columns)
