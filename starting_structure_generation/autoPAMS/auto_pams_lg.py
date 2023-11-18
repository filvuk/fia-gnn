import json
import os
import random
from rdkit import Chem  # 2023.03.3

caps = None
cores = None
substituents_1000 = None
substituents_2 = None
substituents_1234 = None


def load_building_blocks(type_):
    """
    Load the building block scope which should be used to build the ligands.
    type_ can either be "full", "partially_reduced", or "fully_reduced".
    """

    def count(dictionary):
        """
        Function to count the building blocks.
        """
        total_length = 0
        for value in dictionary.values():
            if isinstance(value, dict):
                total_length += count(value)
            elif isinstance(value, list):
                total_length += len(value)
        return total_length

    # Open general building block (independent of provided type_)
    with open(os.path.join("smiles_files", "caps.json"), "r") as f:
        global caps
        caps = json.load(f)

    with open(os.path.join("smiles_files", "cores.json"), "r") as f:
        global cores
        cores = json.load(f)

    with open(os.path.join("smiles_files", "substituents_1000.json"), "r") as f:
        global substituents_1000
        substituents_1000 = json.load(f)

    # Open type_-specific building blocks
    if type_ == "fully_reduced":
        sub_2 = "substituents_2_fully_reduced.json"
        sub_1234 = "substituents_1234_fully_reduced.json"
        print("[auto_pams_lg] Loading fully reduced building ...")

    elif type_ == "partially_reduced":
        sub_2 = "substituents_2_partially_reduced.json"
        sub_1234 = "substituents_1234_partially_reduced.json"
        print("[auto_pams_lg] Loading partially reduced building blocks ...")

    elif type_ == "full":
        sub_2 = "substituents_2_full.json"
        sub_1234 = "substituents_1234_full.json"
        print("[auto_pams_lg] Loading all building blocks ...")

    else:
        sub_2 = "substituents_2_full.json"
        sub_1234 = "substituents_1234_full.json"
        print(
            "[auto_pams_lg] Invalide building block type specified. Defaulting to the full building block space ..."
        )

    with open(os.path.join("smiles_files", sub_2), "r") as f:
        global substituents_2
        substituents_2 = json.load(f)

    with open(os.path.join("smiles_files", sub_1234), "r") as f:
        global substituents_1234
        substituents_1234 = json.load(f)

    # Count building blocks
    if all(
        [
            caps is not None,
            cores is not None,
            substituents_1000 is not None,
            substituents_2 is not None,
            substituents_1234 is not None,
        ]
    ):
        print(f"    [auto_pams_lg] {count(cores)} ligand cores were loaded.")
        print(f"    [auto_pams_lg] {count(caps)} ligand caps were loaded.")
        print(
            f"    [auto_pams_lg] {count(substituents_1234)} general substituents were loaded."
        )
        print(
            f"    [auto_pams_lg] {count(substituents_1000)} aromatic ligand substituents were loaded."
        )
        print(
            f"    [auto_pams_lg] {count(substituents_2)} aromatic ring substituents were loaded."
        )
    print(
        "[auto_pams_lg] Building block were successfully loaded. Structures can now be generated."
    )


def analyze_core(core, spare_wildcards):
    """
    Analyze the given core and set up the building blocks dict.
    spare_wildcards allows that for wildcards with a specified length no substituent is set up
    """
    if isinstance(core, str):
        core = Chem.MolFromSmiles(core)

    building_blocks = {"core": core, "subs": {}}
    for atom in core.GetAtoms():
        if atom.GetSymbol() == "*":
            map_num = str(atom.GetAtomMapNum())

            if len(map_num) not in spare_wildcards:
                binding_atom_idx = [n.GetIdx() for n in atom.GetNeighbors()][0]
                binding_atom_symbol = core.GetAtomWithIdx(binding_atom_idx).GetSymbol()

                if map_num not in building_blocks["subs"]:
                    building_blocks["subs"][map_num] = {
                        "binding_atom_indices": [binding_atom_idx],
                        "binding_atom_symbols": [binding_atom_symbol],
                    }
                else:
                    building_blocks["subs"][map_num]["binding_atom_indices"].append(
                        binding_atom_idx
                    )
                    building_blocks["subs"][map_num]["binding_atom_symbols"].append(
                        binding_atom_symbol
                    )

    return building_blocks


def check_bond_rejection(forbidden_bonds, sub, map_num, building_blocks):
    """
    Check if an unwanted (hetero atom - hetero atom) bond is present,
    """
    reject = False
    for atom in sub.GetAtoms():
        if str(atom.GetAtomMapNum()) == map_num:
            pot_binding_atom_idx = [n.GetIdx() for n in atom.GetNeighbors()][0]
            pot_binding_atom_symbol = sub.GetAtomWithIdx(
                pot_binding_atom_idx
            ).GetSymbol()
            pot_bonds = [
                sorted([i, pot_binding_atom_symbol])
                for i in building_blocks["subs"][map_num]["binding_atom_symbols"]
            ]
            for b in pot_bonds:
                if b in forbidden_bonds:
                    reject = True
    return reject


def get_substituents(building_blocks):
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

    for map_num in building_blocks["subs"]:
        if len(map_num) == 2:
            reject = True
            while reject:
                subsubclass = random.choice(list(substituents_1234[map_num].keys()))
                sub_smi = random.choice(substituents_1234[str(map_num)][subsubclass])
                sub = Chem.MolFromSmiles(sub_smi)
                reject = check_bond_rejection(
                    forbidden_bonds, sub, map_num, building_blocks
                )
            if "*:1000" in sub_smi:
                subclass_A = random.choice(list(substituents_1000.keys()))
                subsub_A = Chem.MolFromSmiles(
                    random.choice(substituents_1000[subclass_A])
                )

                subclass_B = random.choice(list(substituents_2.keys()))
                subsub_B = Chem.MolFromSmiles(random.choice(substituents_2[subclass_B]))

                building_blocks["subs"][map_num]["sub"] = [sub, subsub_A, subsub_B]
            else:
                building_blocks["subs"][map_num]["sub"] = sub

            if map_num.endswith("2"):
                building_blocks["subs"][map_num][
                    "bond_order"
                ] = Chem.rdchem.BondType.DOUBLE
            else:
                building_blocks["subs"][map_num][
                    "bond_order"
                ] = Chem.rdchem.BondType.SINGLE

        elif len(map_num) == 1:
            subclass = random.choice(list(substituents_2.keys()))
            sub = Chem.MolFromSmiles(random.choice(substituents_2[subclass]))

            building_blocks["subs"][map_num]["sub"] = sub
            building_blocks["subs"][map_num]["bond_order"] = Chem.rdchem.BondType.SINGLE

        elif len(map_num) == 3:
            reject = True
            while reject:
                subclass = random.choice(list(caps[map_num].keys()))
                subsubclass = random.choice(list(caps[map_num][subclass].keys()))
                sub = Chem.MolFromSmiles(
                    random.choice(caps[map_num][subclass][subsubclass])
                )
                reject = check_bond_rejection(
                    forbidden_bonds, sub, map_num, building_blocks
                )

            building_blocks["subs"][map_num]["sub"] = sub
            building_blocks["subs"][map_num]["bond_order"] = Chem.rdchem.BondType.SINGLE

        elif len(map_num) == 4:
            subclass = random.choice(list(substituents_1000.keys()))
            sub = Chem.MolFromSmiles(random.choice(substituents_1000[subclass]))

            subclass_A = random.choice(list(substituents_2.keys()))
            subsub_A = Chem.MolFromSmiles(random.choice(substituents_2[subclass_A]))

            building_blocks["subs"][map_num]["sub"] = [sub, subsub_A]
            building_blocks["subs"][map_num]["bond_order"] = Chem.rdchem.BondType.SINGLE

    return building_blocks


def remove_wildcards(final, map_num):
    """
    Remove wildcard atoms.
    """
    to_be_removed = []
    for atom in final.GetAtoms():
        if str(atom.GetAtomMapNum()) == map_num:
            to_be_removed.append(atom.GetIdx())

    to_be_removed.sort(reverse=True)
    while len(to_be_removed) != 0:
        final.RemoveAtom(to_be_removed[0])
        to_be_removed.pop(0)

    return final


def assemble(building_blocks, single_sub_only):
    """
    Assemble the ligand from its building blocks in a recursive manner.
    """
    final = Chem.Mol(building_blocks["core"])

    # Preassemble substituents which consist of multiple building blocks
    for map_num in building_blocks["subs"]:
        while isinstance(building_blocks["subs"][map_num]["sub"], list):
            for atom in building_blocks["subs"][map_num]["sub"][-1].GetAtoms():
                if atom.GetSymbol() == "*":
                    num = str(atom.GetAtomMapNum())

            if num.endswith("2") and len(num) > 2:
                bond_order = Chem.rdchem.BondType.DOUBLE
            else:
                bond_order = Chem.rdchem.BondType.SINGLE

            pre_building_blocks = {
                "core": building_blocks["subs"][map_num]["sub"][-2],
                "subs": {
                    num: {
                        "sub": building_blocks["subs"][map_num]["sub"][-1],
                        "bond_order": bond_order,
                    }
                },
            }

            if len(building_blocks["subs"][map_num]["sub"]) == 2:
                building_blocks["subs"][map_num]["sub"] = assemble(
                    pre_building_blocks, single_sub_only=False
                )["final"]
            else:
                building_blocks["subs"][map_num]["sub"] = [
                    building_blocks["subs"][map_num]["sub"][-3],
                    assemble(pre_building_blocks, single_sub_only=False)["final"],
                ]

    for map_num in building_blocks["subs"]:
        sub = building_blocks["subs"][map_num]["sub"]
        bond_order = building_blocks["subs"][map_num]["bond_order"]

        core_binding_indices = []
        core_wc_indices = []
        core_stereo_adjustments = []

        # Get information for the core
        for atom in final.GetAtoms():
            if str(atom.GetAtomMapNum()) == map_num:
                core_binding_indices.append(
                    [n.GetIdx() for n in atom.GetNeighbors()][0]
                )
                core_wc_indices.append(atom.GetIdx())

        for wc_idx in core_wc_indices:
            entry = None
            for bond in final.GetBonds():
                if wc_idx in bond.GetStereoAtoms():
                    entry = (bond.GetIdx(), list(bond.GetStereoAtoms()).index(wc_idx))
            core_stereo_adjustments.append(entry)

        # In case only one position should be substituted
        if single_sub_only:
            core_binding_indices = [core_binding_indices[0]]
            core_wc_indices = [core_wc_indices[0]]
            core_stereo_adjustments = [core_stereo_adjustments[0]]

        # Get information for the substituents
        for atom in sub.GetAtoms():
            if str(atom.GetAtomMapNum()) == map_num:
                sub_binding_index = [n.GetIdx() for n in atom.GetNeighbors()][0]
                sub_wc_index = atom.GetIdx()

        for bond in sub.GetBonds():
            if sub_wc_index in bond.GetStereoAtoms():
                sub_stereo_adjustment = (
                    bond.GetIdx(),
                    list(bond.GetStereoAtoms()).index(sub_wc_index),
                )
                break
            else:
                sub_stereo_adjustment = None

        # Change the atom map numbers of the used wildcards (start atom(s)) to mark them for removal
        for wc_idx in core_wc_indices:
            final.GetAtomWithIdx(wc_idx).SetAtomMapNum(55555)

        # Assemble and adjust the stereochemistry
        for idx, start_idx in enumerate(core_binding_indices):
            end_idx = sub_binding_index + final.GetNumAtoms()
            end_wc_idx = sub_wc_index + final.GetNumAtoms()
            init_bond_num = final.GetNumBonds()

            final = Chem.CombineMols(final, sub)
            final = Chem.RWMol(final)
            final.AddBond(start_idx, end_idx, order=bond_order)

            # Change the atom map numbers of the used wildcards (end atom) to mark them for removal
            final.GetAtomWithIdx(end_wc_idx).SetAtomMapNum(55555)

            # Adjust the stereochemistry information for double bonds
            if bond_order == Chem.rdchem.BondType.DOUBLE:
                stereo_bond = final.GetBondBetweenAtoms(start_idx, end_idx)
                # get begin_atom_neighbor:
                begin_atom_neighbor_list = [
                    n.GetIdx()
                    for n in final.GetAtomWithIdx(start_idx).GetNeighbors()
                    if n.GetSymbol() != "*" and n.GetIdx() != end_idx
                ]
                if begin_atom_neighbor_list:
                    begin_atom_neighbor = begin_atom_neighbor_list[0]
                else:
                    begin_atom_neighbor_list = [
                        n.GetIdx()
                        for n in final.GetAtomWithIdx(start_idx).GetNeighbors()
                        if n.GetIdx() != end_idx
                    ]
                    begin_atom_neighbor = begin_atom_neighbor_list[0]
                # getend_atom_neighbor:
                end_atom_neighbor_list = [
                    n.GetIdx()
                    for n in final.GetAtomWithIdx(end_idx).GetNeighbors()
                    if n.GetSymbol() != "*" and n.GetIdx() != start_idx
                ]
                if end_atom_neighbor_list:
                    end_atom_neighbor = end_atom_neighbor_list[0]
                else:
                    end_atom_neighbor_list = [
                        n.GetIdx()
                        for n in final.GetAtomWithIdx(end_idx).GetNeighbors()
                        if n.GetIdx() != start_idx
                    ]
                    end_atom_neighbor = end_atom_neighbor_list[0]

                stereo_bond.SetStereo(Chem.rdchem.BondStereo.STEREOE)
                stereo_bond.SetStereoAtoms(begin_atom_neighbor, end_atom_neighbor)

            else:
                if core_stereo_adjustments[idx]:
                    stereo_bond = final.GetBondWithIdx(core_stereo_adjustments[idx][0])
                    stereo_atoms = list(stereo_bond.GetStereoAtoms())
                    stereo_atoms[core_stereo_adjustments[idx][1]] = end_idx
                    stereo_bond.SetStereoAtoms(stereo_atoms[0], stereo_atoms[1])

                if sub_stereo_adjustment:
                    stereo_bond = final.GetBondWithIdx(
                        sub_stereo_adjustment[0] + init_bond_num
                    )
                    stereo_atoms = list(stereo_bond.GetStereoAtoms())
                    stereo_atoms[sub_stereo_adjustment[1]] = start_idx
                    stereo_bond.SetStereoAtoms(stereo_atoms[0], stereo_atoms[1])

        final = remove_wildcards(final, map_num="55555")

    Chem.AssignStereochemistry(final, force=True)
    building_blocks["final"] = final
    return building_blocks


def build_final_ligand(ligand_class, ligand_subclass, max_num_atoms):
    """
    Build the final ligand by calling the individual functions.
    """
    if all(
        [
            caps is not None,
            cores is not None,
            substituents_1000 is not None,
            substituents_2 is not None,
            substituents_1234 is not None,
        ]
    ):
        # try building a ligand until max_num_atoms limit is met:
        while True:
            # Get a core
            if ligand_class == "mono":
                ligand_subclass = random.choice(list(cores[ligand_class].keys()))
                ligand_subsubclass = random.choice(
                    list(cores[ligand_class][ligand_subclass].keys())
                )
                ligand_subsubsubclass = random.choice(
                    list(
                        cores[ligand_class][ligand_subclass][ligand_subsubclass].keys()
                    )
                )
                core_smiles = random.choice(
                    cores[ligand_class][ligand_subclass][ligand_subsubclass][
                        ligand_subsubsubclass
                    ]
                )
            else:
                ligand_subsubclass = random.choice(
                    list(cores[ligand_class][ligand_subclass].keys())
                )
                ligand_subsubsubclass = random.choice(
                    list(
                        cores[ligand_class][ligand_subclass][ligand_subsubclass].keys()
                    )
                )
                core_smiles = random.choice(
                    cores[ligand_class][ligand_subclass][ligand_subsubclass][
                        ligand_subsubsubclass
                    ]
                )

            # Attach substituents to the core
            building_blocks = analyze_core(core_smiles, spare_wildcards=[3])
            building_blocks = get_substituents(building_blocks)
            final_ligand = assemble(building_blocks, single_sub_only=False)["final"]

            # Attach caps to the core and add their substituents
            done = False
            while not done:
                # Caps
                building_blocks = analyze_core(final_ligand, spare_wildcards=[])
                building_blocks = get_substituents(building_blocks)
                final_ligand = assemble(
                    building_blocks, single_sub_only=random.choice([True, False])
                )["final"]

                # Substituents of the caps
                building_blocks = analyze_core(final_ligand, spare_wildcards=[3])
                building_blocks = get_substituents(building_blocks)
                final_ligand = assemble(building_blocks, single_sub_only=False)["final"]

                building_blocks = analyze_core(final_ligand, spare_wildcards=[])
                if not building_blocks["subs"]:
                    done = True

            num_atoms = Chem.AddHs(final_ligand).GetNumAtoms()

            if max_num_atoms > num_atoms:
                break

        # This is done to avoid duplications.
        final_ligand = Chem.RemoveHs(
            final_ligand
        )  # a single H-atom can be a building block, leading to H-atom objects in mol object
        smiles = Chem.MolToSmiles(final_ligand)
        smiles = Chem.MolToSmiles(
            Chem.MolFromSmiles(smiles)
        )  # avoid non-canonical smiles

        return smiles

    else:

        class BuildingBlocksError(Exception):
            """Dummy class for BuildingBlocksError"""

            pass

        raise BuildingBlocksError("No building blocks were loaded.")
