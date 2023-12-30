import os
import hashlib
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem


# Creat output directories
if not os.path.isdir("manual_pams_out"):
    os.mkdir("manual_pams_out")
    os.mkdir(r"manual_pams_out\xyz_out")
    print("[manual_pams] Main output directory was created.")
    print("[manual_pams] XYZ output directory was created.")
else:
    print("[manual_pams] Main output directory already exists.")

if not os.path.isdir(r"manual_pams_out\xyz_out"):
    os.mkdir(r"manual_pams_out\xyz_out")
    print("[manual_pams] XYZ output directory was created.")


class ManualPAMS:
    """
    A class for the manual integration of molecules to the autoPAMS routines
    starting from SMILES strings only (e.g., extracted from a data base).
    A 3D starting structure of the Lewis acid is determined and saved if requested.
    ...
    Parameters
    ----------
    smiles: str
        Representation of the molecule (Lewis acid) in form of its smiles string.
    central_atom_symbol: str
        Atomic symbol of the central atom to which the fluoride atom binds.
        If not provided, it is found automatically.
    smiles__F: str
        Representation of the fluoride adduct in form of its smiles string.
        Can be provided if available.
    """

    def __init__(
        self,
        smiles=None,
        central_atom_symbol=None,
        smiles__F="Was not generated.",
    ):
        self.smiles = smiles
        self.central_atom_symbol = central_atom_symbol
        self.smiles__F = smiles__F

        self.error = None

        self.mol = None
        self.rw_ligands_mol = None
        self.central_atom_idx = None

        self.xyz_coords = None
        self.atom_connectivities = {}

        self.ligand_smiles = []
        self.central_atom_valency = None
        self.ligand_class = None
        self.denticity_class = None
        self.sub_denticity_class = None
        self.donor_atom_types = {}
        self.fingerprint = None

        self.name = None
        self.json_data = {}

    def __call__(
        self,
        mol_idx_number=1,
        set_identifier="",
        custom_name=None,
        embed_3D=True,
        ignore_mini_ligands=False,
    ):
        """
        Call ManualPAMS. It returns the generated name (identifier) of the molecule,
        the collected json data (central atom class, denticity class, etc.) and a
        dictionary with the atom connectivities of the Lewis acid and the fluoride adduct.
        ...
        Arguments
        ---------
        mol_idx_number: int
            Molecule index number, will be used to construct the name (identifier) of the molecule.
        set_identifier: str
            Identifier to assign the molecule to a certain subset of the data set, will be used to
            construct the name (identifier) of the molecule.
        custom_name: str
            If the name (identifier) of the molecule is already know, it can be used via this
            parameter.
        embed_3D: bool
            Whether or not the 3-dimensional xyz structure of the molecule should be calculated.
        ignore_mini_ligands: bool
            Whether or not the so-called mini ligands should be ignored when analyzing the atoms
            directly binding to the central atom.
        """

        print("---------------------------------------------------------")
        if custom_name is not None:
            print(f"[manual_pams] Working on {custom_name}")
        else:
            print(f"[manual_pams] Working on {self.smiles}")
        print("---------------------------------------------------------")

        self.get_mol()
        if self.mol is not None:
            # Get central atom information
            self.get_central_atom_symbol()
            if self.central_atom_symbol is not None:
                self.get_central_atom_idx()
                self.reorder_atoms()

                # Get data for json file with meta data.
                self.get_ligand_smiles()
                self.get_central_atom_valency()
                self.get_ligand_class()
                self.get_ligand_info()
                self.get_donor_atoms_types(ignore_mini_ligands)
                self.gen_fingerprint()
                self.get_name(mol_idx_number, set_identifier, custom_name)
                self.get_json_data()

                # Get xyz structure and atom connectivities if requested (default).
                if embed_3D:
                    self.embed_structure()
                    if self.xyz_coords is not None:
                        self.save_xyz_coords()
                        self.get_atom_connectivities()

        print(f"    [manual_pams] Error: {self.error}")
        print()
        return self.name, self.json_data, self.atom_connectivities

    def get_mol(self):
        """Read in the provided SMILES string and get RDKit mol object."""
        try:
            mol = Chem.MolFromSmiles(self.smiles)
        except:
            self.error = "Provided SMILES cannot be converted to RDKit mol object."

        if mol is not None:
            try:
                mol = Chem.AddHs(mol)
            except:
                self.error = "Provided SMILES cannot be converted to RDKit mol object."
            else:
                self.mol = mol
                self.smiles = Chem.MolToSmiles(self.mol)
        else:
            self.error = "Provided SMILES cannot be converted to RDKit mol object."

    def get_xyz_from_mol_file(self, file_path, uff_optimize=True):
        """Read in a mol file."""
        self.name = os.path.basename(file_path).split(".")[0]

        print("---------------------------------------------------------")
        print(f"[manual_pams] Working on {self.name}")
        print("---------------------------------------------------------")

        try:
            self.mol = Chem.MolFromMolFile(file_path, removeHs=False)
        except:
            self.error = "Provided mol file cannot be converted to RDKit mol object."

        if self.mol is not None:
            self.get_central_atom_symbol()
            if self.central_atom_symbol is not None:
                self.get_central_atom_idx()
                self.reorder_atoms()

                if uff_optimize is True:
                    try:
                        opt = AllChem.UFFOptimizeMolecule(self.mol)
                    except:
                        opt = -1
                        self.error = "UFF optimization failed."
                    else:
                        if opt == 0:
                            self.get_atom_connectivities()
                            self.xyz_coords = Chem.rdmolfiles.MolToXYZBlock(self.mol)
                            self.save_xyz_coords()
                            print(
                                f"    [manual_pams] UFF optimization was successful and structure was saved."
                            )
                        else:
                            self.error = "UFF optimization failed."
                else:
                    self.get_atom_connectivities()
                    self.xyz_coords = Chem.rdmolfiles.MolToXYZBlock(self.mol)
                    self.save_xyz_coords()
                    print(f"    [manual_pams] Structure was saved.")

        else:
            self.error = "Provided mol file cannot be converted to RDKit mol object."

        print(f"    [manual_pams] Error: {self.error}")
        return self.name, self.json_data, self.atom_connectivities

    def get_central_atom_symbol(self):
        """Get the central atom symbol."""
        all_central_atom_symbols = [
            "B",
            "Al",
            "Ga",
            "In",
            "Si",
            "Ge",
            "Sn",
            "Pb",
            "P",
            "As",
            "Sb",
            "Bi",
            "Te",
        ]

        if self.central_atom_symbol is None:
            for atom in self.mol.GetAtoms():
                if atom.GetSymbol() in all_central_atom_symbols:
                    self.central_atom_symbol = atom.GetSymbol()
                    break

        if self.central_atom_symbol is None:
            self.error = "Provided central atom symbol was not found."

    def get_central_atom_idx(self):
        """Get the central atom index in the mol object."""
        if self.central_atom_symbol is not None:
            for atom in self.mol.GetAtoms():
                if atom.GetSymbol() == self.central_atom_symbol:
                    self.central_atom_idx = atom.GetIdx()
                    break
        else:
            self.error = "Central atom index cannot be determined without knowing the central atom symbol."

    def reorder_atoms(self):
        """Put the central atom first in the RDKit mol object."""
        new_order = [self.central_atom_idx]
        new_order.extend(
            [x for x in range(self.mol.GetNumAtoms()) if x != self.central_atom_idx]
        )
        self.mol = rdmolops.RenumberAtoms(self.mol, new_order)
        self.central_atom_idx = 0

    def embed_structure(self):
        """Get 3D structure."""
        try:
            embed = AllChem.EmbedMolecule(self.mol, params=AllChem.ETKDG())
        except:
            embed = -1
            self.error = "Embedding failed."

        if embed == 0:
            try:
                opt = AllChem.UFFOptimizeMolecule(self.mol)
            except:
                opt = -1
                self.error = "UFF optimization failed."

            if opt == 0:
                self.xyz_coords = Chem.rdmolfiles.MolToXYZBlock(self.mol)

        # Double check that embedding was successful.
        if self.xyz_coords is None:
            self.error = "Embedding failed."
        else:
            splitted = self.xyz_coords.split("\n")
            try:
                first = int(splitted[0])
                if not len(splitted) > 2 and not isinstance(first, int):
                    self.error = "Embedding failed."
                    self.xyz_coords = None
            except:
                self.error = "Embedding failed."
                self.xyz_coords = None

    def save_xyz_coords(self):
        """Save xyz file."""
        path = os.path.join("manual_pams_out", "xyz_out", f"{self.name}.xyz")
        if not os.path.isfile(path):
            with open(path, "w") as f:
                f.write(self.xyz_coords)

    def get_ligand_smiles(self):
        """Get the SMILES strings of the ligands attached to the central atom."""
        mol = Chem.Mol(self.mol)
        mol = Chem.RWMol(mol)

        # Get central atom object
        central_atom = mol.GetAtomWithIdx(self.central_atom_idx)
        central_atom_bond_indices = [
            (bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx())
            for bond in central_atom.GetBonds()
        ]

        # Remove aromatic information from the molecule
        Chem.Kekulize(mol, clearAromaticFlags=True)

        # Set radicals
        for neighbor in central_atom.GetNeighbors():
            neighbor.SetNoImplicit(True)
            neighbor.SetNumRadicalElectrons(1)

        # Remove bonds to central atom
        _ = [mol.RemoveBond(bond[0], bond[1]) for bond in central_atom_bond_indices]

        # Remove central atom
        mol.RemoveAtom(self.central_atom_idx)

        # Get SMILES of the ligands
        mol = Chem.RemoveHs(mol)
        self.ligand_smiles = Chem.MolToSmiles(mol).split(".")

        # Sanitize mol. This is required for the get_ligand_class method, which runs GetRingInfo on the here generated mol object.
        Chem.SanitizeMol(mol)
        self.rw_ligands_mol = Chem.Mol(mol)

    def get_central_atom_valency(self):
        """Get the number of atoms around the central atom."""
        central_atom = self.mol.GetAtomWithIdx(self.central_atom_idx)
        self.central_atom_valency = str(len(central_atom.GetBonds()))

    def get_ligand_class(self):
        """Get the ligand class to which the molecule belongs. This gives information on the rings the central atom is in."""
        ligand_class = []
        ring_info = self.mol.GetRingInfo()
        for atom_ring in ring_info.AtomRings():
            if self.central_atom_idx in atom_ring:
                ligand_class.append(len(atom_ring))

        ligand_class.sort()
        self.ligand_class = "".join([str(x) for x in ligand_class])

        # Check if there is a macrocyclic ligand in the molecule
        # 1) Get number of rings in which the central atom is not present
        # 2) Remove the central atom and get the number of rings (This is done in the get_ligand_smiles method)
        # 3) If these two ring counts are different there is a macrocyclic ligand
        non_central_atom_rings = 0
        for atom_ring in ring_info.AtomRings():
            if self.central_atom_idx not in atom_ring:
                non_central_atom_rings += 1

        if non_central_atom_rings < self.rw_ligands_mol.GetRingInfo().NumRings():
            self.ligand_class = "macro_" + self.ligand_class

        # If there is no ligand class, set it to "None".
        if self.ligand_class == "":
            self.ligand_class = "None"

    def get_ligand_info(self):
        """
        Get ...
            1) the denticity class to which the molecule belongs, either "mono", "bi", or "tri".
            2) the sub-denticity class of the molecule, either "single", "double", or "None".
        """

        denticities = {1: "mono", 2: "bi", 3: "tri", 4: "tetra"}

        sub_denticities = {0: "None", 1: "single", 2: "double"}

        max_num_radical_electrons = 0
        bidentate_ligand_count = 0
        tridentate_ligand_count = 0

        for smiles in self.ligand_smiles:
            # Get information to determine the denticity class
            total_num_radical_electrons = 0
            mol = Chem.MolFromSmiles(smiles)
            for atom in mol.GetAtoms():
                if atom.GetNumRadicalElectrons() == 1:
                    total_num_radical_electrons += 1

            if total_num_radical_electrons > max_num_radical_electrons:
                max_num_radical_electrons = total_num_radical_electrons

            if total_num_radical_electrons == 2:
                bidentate_ligand_count += 1

            if total_num_radical_electrons == 3:
                tridentate_ligand_count += 1

        # Determine denticity class
        if max_num_radical_electrons in denticities:
            self.denticity_class = denticities[max_num_radical_electrons]
        else:
            self.error = "Denticity class could not be determined correctly."

        # Determine sub-denticity class
        if tridentate_ligand_count != 0:
            self.sub_denticity_class = "None"
        else:
            if bidentate_ligand_count in sub_denticities:
                self.sub_denticity_class = sub_denticities[bidentate_ligand_count]
            else:
                self.error = "Sub-denticity class could not be determined correctly."

    def get_donor_atoms_types(self, ignore_mini_ligands):
        """
        Get the hybridization of the donor atoms (atoms binding to the central atom).
        This is done after getting the denticity and sub-denticity class as mini-ligands
        should only be ignored for denticity classes "bi" and "tri".
        """
        mini_ligands = [
            "[H]",
            "[F]",
            "[Cl]",
            "[Br]",
            "[I]",
            "C[O]",
            "C[S]",
            "[CH]=C",
            "C[N]C",
            "C=[N]",
            "[CH3]",
        ]

        for smiles in self.ligand_smiles:
            mol = Chem.MolFromSmiles(smiles)
            smiles = Chem.MolToSmiles(mol)

            if (
                self.denticity_class != "mono"
                and ignore_mini_ligands is True
                and smiles in mini_ligands
            ):
                continue

            for atom in mol.GetAtoms():
                if atom.GetNumRadicalElectrons() == 1:
                    atom_type = str(atom.GetSymbol()) + str(atom.GetHybridization())
                    if (
                        atom.GetSymbol() in ["H", "F", "Cl", "Br", "I"]
                        and ignore_mini_ligands is True
                    ):
                        atom_type = "monoatomic"

                    if atom_type not in self.donor_atom_types:
                        self.donor_atom_types[atom_type] = 1
                    else:
                        self.donor_atom_types[atom_type] += 1

    def get_name(self, mol_idx_number, set_identifier, custom_name, idx_number_width=6):
        """Get the name (identifier) of the molecule."""
        if custom_name is not None:
            self.name = custom_name
        else:
            denticities = {"mono": "M", "bi": "B", "tri": "T", "tetra": "TE"}
            self.name = "_".join(
                [
                    self.central_atom_symbol,
                    self.central_atom_valency,
                    f"{set_identifier}{denticities[self.denticity_class]}{mol_idx_number:0{idx_number_width}d}",
                ]
            )

    def gen_fingerprint(self):
        """Get the fingerprint of a molecule, defined by its central atom and its list of ligands."""
        string_list = self.ligand_smiles.copy()
        string_list.append(self.central_atom_symbol)
        string_list.sort()
        string = "_".join(string_list)
        self.fingerprint = hashlib.md5(string.encode()).hexdigest()

    def get_json_data(self):
        """Get final data dictionary."""
        self.json_data = {
            "central_atom": f"{self.central_atom_symbol}_{self.central_atom_valency}",
            "denticity_class": self.denticity_class,
            "sub_denticity_class": self.sub_denticity_class,
            "ligand_class": self.ligand_class,
            "ligands": self.ligand_smiles,
            "donors": self.donor_atom_types,
            "smiles": self.smiles,
            "smiles__F": self.smiles__F,
            "fingerprint": self.fingerprint,
        }

    def get_atom_connectivities(self):
        """Get the atom connectivities of the Lewis acid and the fluoride adduct."""
        atom_connectivities_lewis_acid = np.empty(shape=(0, 2), dtype=np.int32)
        for bonds in self.mol.GetBonds():
            bond = np.array(
                [[bonds.GetBeginAtom().GetIdx(), bonds.GetEndAtom().GetIdx()]],
                dtype=np.dtype(int),
            )
            atom_connectivities_lewis_acid = np.append(
                atom_connectivities_lewis_acid, bond, axis=0
            )
        atom_connectivities_lewis_acid.sort(axis=1)
        atom_connectivities_lewis_acid = atom_connectivities_lewis_acid[
            np.lexsort(
                (
                    atom_connectivities_lewis_acid[:, 1],
                    atom_connectivities_lewis_acid[:, 0],
                )
            )
        ]

        # Store data in dictionary
        self.atom_connectivities = {self.name: atom_connectivities_lewis_acid}
