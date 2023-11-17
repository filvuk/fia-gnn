# pylint: disable=invalid-name, global-variable-not-assigned, unused-import, line-too-long, missing-function-docstring, missing-module-docstring, missing-class-docstring, no-member, c-extension-no-member, unspecified-encoding
import json
import numpy as np
import tensorflow as tf
from nfp.preprocessing.mol_preprocessor import MolPreprocessor
from rdkit import Chem

path_to_periodic_table = 'periodic_table_all_118.json'

with open(path_to_periodic_table, 'r') as f:
    my_periodic_table = json.load(f)

def atom_featurizer(atom):
    return str((atom.GetSymbol(), atom.GetDegree(), ))

def bond_featurizer(bond, flipped=False):
    if not flipped:
        atoms = f"{bond.GetBeginAtom().GetSymbol()}-{bond.GetEndAtom().GetSymbol()}"
    else:
        atoms = f"{bond.GetEndAtom().GetSymbol()}-{bond.GetBeginAtom().GetSymbol()}"
    btype = str((bond.GetBondType(), ))
    return " ".join([atoms, btype]).strip()

def get_nfp_preprocessor(path):
    preprocessor = MolPreprocessor(
        atom_features=atom_featurizer,
        bond_features=bond_featurizer,
        output_dtype="int64",)
    preprocessor.from_json(path)
    return preprocessor

def get_output_signature(nfp_preprocessor):
    output_signature = {
        **nfp_preprocessor.output_signature,
        **{"dist_to_central_atom": tf.TensorSpec(shape=(None, ), dtype=tf.int64)},
        **{"global_features": tf.TensorSpec(shape=(None, ), dtype=tf.float32)},
    }
    return output_signature

def get_padding_values(nfp_preprocessor):
    padding_values = {
        **nfp_preprocessor.padding_values,
        **{"dist_to_central_atom": tf.constant(0, dtype=tf.int64)},
        **{"global_features": tf.constant(0, dtype=tf.float32)},
    }
    return padding_values


class fiaGnnPreprocessor:
    def __init__(self, mol, fia_gas_preprocessor=None, fia_solv_preprocessor=None, ca_idx=None) -> None:
        self.periodic_table = my_periodic_table
        self.fia_gas_preprocessor = fia_gas_preprocessor
        self.fia_solv_preprocessor = fia_solv_preprocessor
       
        self.mol = Chem.AddHs(mol)
        self.smiles = self.canonical_smiles()

        if ca_idx:
            self.ca_idx = ca_idx
        else:
            self.ca_idx = self.get_ca_idx()
        self.ca_symbol = self.mol.GetAtomWithIdx(self.ca_idx).GetSymbol()

        self.global_features = self.get_global_features()
        self.distance_features = self.get_distance_features()

        self._fia_gas_input = None # dict: is set if self.fia_gas_input property is called
        self._fia_solv_input = None # dict: is set if self.fia_solv_input property is called
        self._unknown_atom_token = None # bool: is set if self.fia_gas_input or self.fia_solv_input property is called
        self._unknown_bond_token = None # bool: is set if self.fia_gas_input or self.fia_solv_input property is called
        return

    @property
    def fia_gas_input(self):
        # check if _fia_gas_input is not already calculatet:
        if self._fia_gas_input is None:
            if self.fia_gas_preprocessor is None:
                raise ValueError('NFP preprocessor for fia gas model is not specified.')
            input_dict = self.fia_gas_preprocessor(self.mol, train=False)
            input_dict['dist_to_central_atom'] = self.distance_features
            input_dict['global_features'] = self.global_features
            self._fia_gas_input = input_dict
            self._unknown_atom_token = (1 in input_dict['atom'])
            self._unknown_bond_token = (1 in input_dict['bond'])
        return self._fia_gas_input

    @property
    def fia_solv_input(self):
        # check if _fia_solv_input is not already calculatet:
        if self._fia_solv_input is None:
            if self.fia_gas_preprocessor is None:
                raise ValueError('NFP preprocessor for fia solv model is not specified.')
            input_dict = self.fia_solv_preprocessor(self.mol, train=False)
            input_dict['dist_to_central_atom'] = self.distance_features
            input_dict['global_features'] = self.global_features
            self._fia_solv_input = input_dict
            self._unknown_atom_token = (1 in input_dict['atom'])
            self._unknown_bond_token = (1 in input_dict['bond'])
        return self._fia_solv_input

    @property
    def unknown_atom_token(self):
        # if _unknown_atom_token is not set, fia_gas_input (or fia_solv_input) needs to be calculated:
        if self._unknown_atom_token is None:
            _ = self.fia_gas_input
        return self._unknown_atom_token

    @property
    def unknown_bond_token(self):
        # if _unknown_bond_token is not set, fia_gas_input (or fia_solv_input) needs to be calculated:
        if self._unknown_bond_token is None:
            _ = self.fia_gas_input
        return self._unknown_bond_token

    def canonical_smiles(self):
        mol = Chem.Mol(self.mol)
        Chem.RemoveStereochemistry(mol)
        smiles = Chem.MolToSmiles(Chem.RemoveHs(self.mol))
        return smiles

    def get_ca_idx(self):
        ca_smarts = Chem.MolFromSmarts('[#5X3,AlX3,GaX3,InX3,SiX2,SiX4,GeX2,GeX4,SnX2,SnX4,PbX2,PbX4,#15X3,#15X5,AsX3,AsX5,SbX3,SbX5,BiX3,TeX4]')
        ca_idx = self.mol.GetSubstructMatches(ca_smarts)
        if len(ca_idx) == 0:
            raise ValueError('No cantral atom found. Invalid molecule.')
        if len(ca_idx) > 1:
            raise ValueError('More than one cantral atom found. Please specify index of central atom.')
        ca_idx = ca_idx[0][0]
        return ca_idx

    def get_global_features(self):
        period = self.periodic_table[self.ca_symbol]['Period']
        group = self.periodic_table[self.ca_symbol]['Group']
        lig_count = self.get_lig_count()
        ca_ox = self.get_ca_ox()
        ca_ring_count = self.get_ca_ring_count()
        return np.array([period, group, lig_count, ca_ox, ca_ring_count])

    def get_distance_features(self):
        return  [i if i < 18 else 18.0 for i in Chem.rdmolops.GetDistanceMatrix(self.mol)[self.ca_idx]]

    def get_lig_count(self):
        mol = Chem.RWMol(self.mol)
        mol.RemoveAtom(self.ca_idx)
        fragments = Chem.GetMolFrags(mol)
        return len(fragments)

    def get_ca_ring_count(self):
        ri = self.mol.GetRingInfo()
        atomRings = list(ri.AtomRings())
        ring_count = 0
        for ring in atomRings:
            if self.ca_idx in ring:
                ring_count += 1
        return ring_count

    def get_ca_ox(self):
        atom = self.mol.GetAtomWithIdx(self.ca_idx)
        atom_el_neg = self.periodic_table[atom.GetSymbol()]["Electronegativity (Pauling)"]
        contributions = []
        for n in list(atom.GetNeighbors()):
            neighbor_el_neg = self.periodic_table[n.GetSymbol()]["Electronegativity (Pauling)"]
            neighbor_bond_order = self.mol.GetBondBetweenAtoms(self.ca_idx, n.GetIdx()).GetBondTypeAsDouble()
            if neighbor_el_neg > atom_el_neg:
                contributions.append(neighbor_bond_order)
            elif neighbor_el_neg < atom_el_neg:
                contributions.append(-neighbor_bond_order)
            else:
                pass
        return int(sum(contributions) + atom.GetFormalCharge())
