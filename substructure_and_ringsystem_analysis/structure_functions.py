import re
from rdkit import Chem


def getRingSystemBonds(mol):
    """
    Returns list of sets, each set containing the bond indices of a ring system.
    """
    ri = mol.GetRingInfo()
    bondRings = list(ri.BondRings())
    if len(bondRings) == 0:
        return list()
    ringSystems = []
    ringSystems.append(set(bondRings.pop(0)))
    while len(bondRings) > 0:
        for i, ringSystem in enumerate(ringSystems):
            for j, ring in enumerate(bondRings):
                if len(set(ring) & ringSystem) > 0:
                    ringSystems[i] = set(ring) | ringSystem
                    bondRings.pop(j)
                    break
            else:
                continue
            break
        else:
            ringSystems.append(set(bondRings.pop(0)))
    return ringSystems


def getRingSystemSmiles(mol, includeTerminalAtoms=True):
    """
    Returns list of smiles of ring systems.
    """
    # some lists for later:
    ringSmiles = []
    envs = getRingSystemBonds(mol)

    for env in envs:
        atomEnv = set()
        for bond_idx in env:
            atomEnv.add(mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx())
            atomEnv.add(mol.GetBondWithIdx(bond_idx).GetEndAtomIdx())

        # enlarge the environment by one further bond
        enlargedEnv = set()
        for atom_idx in atomEnv:
            a = mol.GetAtomWithIdx(atom_idx)
            for bond in a.GetBonds():
                bond_idx = bond.GetIdx()
                if bond_idx not in env:
                    enlargedEnv.add(bond_idx)
        enlargedEnv = list(enlargedEnv)
        enlargedEnv += env

        # find all relevant wildcard atoms (i.e. all atoms in shell "radius+1")
        wildcardAtoms = set()
        for atom_idx in atomEnv:
            neighbors = mol.GetAtomWithIdx(atom_idx).GetNeighbors()
            for neighbor in neighbors:
                n_idx = neighbor.GetIdx()
                is_H = neighbor.GetAtomicNum() == 1  # mandatory check
                if includeTerminalAtoms:
                    is_terminal = (
                        len(neighbor.GetNeighbors()) + neighbor.GetTotalNumHs() == 1
                    )
                    is_donor = (
                        neighbor.GetNumRadicalElectrons() == 1
                    )  # optional check, explicitly for this use case: donor atoms are never terminal
                    is_terminal = (
                        is_terminal and not is_donor
                    )  # optional check, explicitly for this use case: donor atoms are never terminal
                else:
                    is_terminal = False

                if n_idx not in atomEnv and not is_H and not is_terminal:
                    wildcardAtoms.add(n_idx)

        # generate submol
        atom_map = {}
        submol = Chem.PathToSubmol(mol, enlargedEnv, atomMap=atom_map)

        # set AtomicNum for wildcard atoms
        for atom_idx in wildcardAtoms:
            atom = submol.GetAtomWithIdx(atom_map[atom_idx])
            atom.SetAtomicNum(0)

        ### RDKit shenanigans part 1:
        ### set all Hs of submol as explicit and remove explicit Hs from wildcard atoms
        ### this enables the generated smarts to be read in as smiles
        ### but why?!?
        ###   Reason 1: rdkit.Chem.rdchem.Mol objects that were generated from smiles work better with the rdkit.Chem.Draw functions
        ###   Reason 2: see "RDKit shenanigans part 2"
        for mol_atom_idx, submol_atom_idx in atom_map.items():
            mol_atom = mol.GetAtomWithIdx(mol_atom_idx)
            submol_atom = submol.GetAtomWithIdx(submol_atom_idx)
            if not mol_atom_idx in wildcardAtoms:
                submol_atom.SetNumExplicitHs(mol_atom.GetTotalNumHs())
            else:
                submol_atom.SetNumExplicitHs(0)

        ### RDKit shenanigans part 2:
        ### 1. generate smarts
        ### 2. read in smarts as smiles
        ### 3. generate smiles
        ### but why?!?
        ###   Reason: smarts are not canonical, e.g. different smarts for equal ring systems can be generated
        ###   but why don't generate smiles in the first place?!?
        ###       Reason: directly generated smiles are broken
        smarts = Chem.MolToSmarts(submol)
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smarts))
        ### Fun fact: very view non-canonical smiles still exist, so let's do another round of "read in smiles" and "generate smiles":
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        ringSmiles.append(smiles)

    return ringSmiles


### https://iwatobipen.wordpress.com/2020/08/12/get-and-draw-molecular-fragment-with-user-defined-path-rdkit-memo/
def getSubSmilesRadN(
    mol, radius, skipRings=True, avoidSubsets=True, includeTerminalAtoms=True
):
    """
    Returns list of smiles of all substructures of given radius.
    If molecule is smaller than given radius, the molecule smiles is returned.

    If centrum of substructure is a ring atom, substructure is skipped by default.
    If a substructure is a subset of another substructure, substructure is skipped by default.
    Terminal atoms (-F, -Cl, =O, ...) are not converted to wildcard atoms by default.

    Be careful with molecules with polycyclic aromatic ring systems:
    Substructures of polycyclic aromatic ring systems can't be represented
    as rdkit smiles consistently (aromaticity is not represented consistently).
    Keep radius small to avoid substructures that contain closed ring of aromatic ring system.
    """
    # some lists for later:
    subSmiles = []
    envs = []
    atomEnvs = []
    wildcards = []
    ring_atom_count = 0

    # replace radical electrons with dummy neighbours since no radicals are present in the database
    # and rdkit needs neighbours for aromaticity calculation, otherwise radical atoms that would otherwise be aromatic
    # are not, which leads to errors in substructure search
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() == 1:  
            mol.GetAtomWithIdx(atom.GetIdx()).SetNumRadicalElectrons(0)
            edit = Chem.EditableMol(mol)
            new_atom = Chem.Atom(0)
            ca_index_sub = edit.AddAtom(new_atom)
            edit.AddBond(atom.GetIdx(), ca_index_sub, Chem.rdchem.BondType.SINGLE)
            mol = edit.GetMol()
    # reparse mol
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))

    
    # STEP 1: generate all envs, atomEnvs and wildcards of all substructure:
    # loop all atoms of mol:
    for atom in mol.GetAtoms():
        # skip central atoms in rings:
        if skipRings and atom.IsInRing():
            ring_atom_count += 1
            continue

        # skip dummy atoms
        if atom.GetAtomicNum() == 0:
            continue

        ### RDKit shenanigans part 0:
        ### skip H-Atoms:
        ### in order to set stereo-information at double bonds, H-Atom objects do exist
        ### these H-Atoms can not be removed by mol.RemoveHs()
        if atom.GetAtomicNum() == 1:
            continue

        # get environment at atomIdx with radius
        atomIdx = atom.GetIdx()
        if radius > 0:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atomIdx)
            # check if env of radius exists at atomIdx:
            if not set(env):
                continue
            atomEnv = set()
            for bond_idx in env:
                atomEnv.add(mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx())
                atomEnv.add(mol.GetBondWithIdx(bond_idx).GetEndAtomIdx())
        else:
            env = []
            atomEnv = set((atomIdx,))

        # enlarge the environment by one further bond
        enlargedEnv = set()
        for atom_idx in atomEnv:
            a = mol.GetAtomWithIdx(atom_idx)
            for bond in a.GetBonds():
                bond_idx = bond.GetIdx()
                if bond_idx not in env:
                    enlargedEnv.add(bond_idx)
        enlargedEnv = list(enlargedEnv)
        enlargedEnv += env

        # find all relevant wildcard atoms (i.e. all atoms in shell "radius+1")
        wildcardAtoms = set()
        for atom_idx in atomEnv:
            neighbors = mol.GetAtomWithIdx(atom_idx).GetNeighbors()
            for neighbor in neighbors:
                n_idx = neighbor.GetIdx()
                is_H = neighbor.GetAtomicNum() == 1
                if includeTerminalAtoms:
                    is_terminal = (
                        len(neighbor.GetNeighbors()) + neighbor.GetTotalNumHs() == 1
                    )
                    is_donor = (
                        neighbor.GetNumRadicalElectrons() == 1
                    )  # optional check, explicitly for this use case: donor atoms are never terminal
                    is_terminal = (
                        is_terminal and not is_donor
                    )  # optional check, explicitly for this use case: donor atoms are never terminal
                else:
                    is_terminal = False

                if n_idx not in atomEnv and not is_H and not is_terminal:
                    wildcardAtoms.add(n_idx)

        # generate enlargedAtomEnv and check if current enlargedAtomEnv is a duplicate:
        enlargedAtomEnv = atomEnv.copy()
        for bond_idx in enlargedEnv:
            enlargedAtomEnv.add(mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx())
            enlargedAtomEnv.add(mol.GetBondWithIdx(bond_idx).GetEndAtomIdx())
        for atom_idx in wildcardAtoms:
            enlargedAtomEnv.remove(atom_idx)

        if enlargedAtomEnv in atomEnvs:
            continue
        else:
            atomEnvs.append(enlargedAtomEnv)
            envs.append(enlargedEnv)
            wildcards.append(wildcardAtoms)

    # STEP 2: avoid subsets
    # check for subsets in atomEnvs
    if avoidSubsets:
        subSets = []
        for i, atomEnvA in enumerate(atomEnvs):
            for j, atomEnvB in enumerate(atomEnvs):
                if i != j:
                    if atomEnvA.issubset(atomEnvB):
                        subSets.append(i)
                    if atomEnvB.issubset(atomEnvA):
                        subSets.append(j)

        envs = [x for i, x in enumerate(envs) if i not in subSets]
        atomEnvs = [x for i, x in enumerate(atomEnvs) if i not in subSets]
        wildcards = [x for i, x in enumerate(wildcards) if i not in subSets]

    # STEP 3: generate submols and convert them to smiles:
    
    for enlargedEnv, wildcardAtoms in zip(envs, wildcards):
        # generate submol
        atom_map = {}
        submol = Chem.PathToSubmol(mol, enlargedEnv, atomMap=atom_map)

        # set AtomicNum or AtomMapNum for wildcard atoms
        for atom_idx in wildcardAtoms:
            atom = submol.GetAtomWithIdx(atom_map[atom_idx])
            # if wildcard atom is in aromatic ring, it is marked by AtomicMapNum 1000,
            # because information about element type is still necessary to kekulize the submol smarts
            if atom.GetIsAromatic() and atom.IsInRing():
                atom.SetAtomMapNum(1000)
            # else the wildcard atom is directly marked as wildcard atom, as information about element type is not needed:
            else:
                atom.SetAtomicNum(0)

        ### RDKit shenanigans part 1:
        ### set all Hs of submol as explicit and remove explicit Hs from wildcard atoms
        ### this enables the generated smarts to be read in as smiles
        ### but why?!?
        ###   Reason 1: rdkit.Chem.rdchem.Mol objects that were generated from smiles work better with the rdkit.Chem.Draw functions
        ###   Reason 2: see "RDKit shenanigans part 2"
        for mol_atom_idx, submol_atom_idx in atom_map.items():
            mol_atom = mol.GetAtomWithIdx(mol_atom_idx)
            submol_atom = submol.GetAtomWithIdx(submol_atom_idx)
            if (
                not mol_atom_idx in wildcardAtoms 
            ):  # or submol_atom.GetAtomMapNum() == 1000:
                submol_atom.SetNumExplicitHs(mol_atom.GetTotalNumHs())
            else:
                submol_atom.SetNumExplicitHs(0)

        ### RDKit shenanigans part 2:
        ### 1. generate smarts
        ### 2. read in smarts as smiles
        ### 3. generate smiles
        ### but why?!?
        ###   Reason: smarts are not canonical, e.g. different smarts for equal submols can be generated
        ###   but why don't generate smiles in the first place?!?
        ###       Reason: directly generated smiles are broken
        smarts = Chem.MolToSmarts(submol)
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smarts))
        ### at this point smarts was converted to smiles, so the remaining wildcard atoms can be set:
        smiles = re.sub("(\[)([a-zA-Z]{1,2})(H|H[2-9])?(\:1000\])", "*", smiles)
        ### Fun fact: very view non-canonical smiles still exist, so let's do another round of "read in smiles" and "generate smiles":
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        subSmiles.append(smiles)

    # check if mol object consist of only ring atoms, if skipRings flag is set to True:
    if skipRings:
        mol_has_only_ring_atoms = mol.GetNumAtoms() == ring_atom_count
    else:
        mol_has_only_ring_atoms = False

    # check if at least one env of radius exists in mol object, else smiles of mol object is returned
    if not subSmiles and not mol_has_only_ring_atoms:
        return [
            Chem.MolToSmiles(mol),
        ]

    return subSmiles
