import os
from ccdc import io
from ccdc import conformer
from ccdc.molecule import Molecule


# Creat output directories
if not os.path.isdir("manual_pams_out"):
    os.mkdir("manual_pams_out")
    os.mkdir(r"manual_pams_out\mol_out")
    print("[ccdc_embedder] Main output directory was created.")
    print("[ccdc_embedder] Mol file output directory was created.")
else:
    print("[ccdc_embedder] Main output directory already exists.")

if not os.path.isdir(r"manual_pams_out\mol_out"):
    os.mkdir(r"manual_pams_out\mol_out")
    print("[ccdc_embedder] Mol file output directory was created.")


def ccdc_embedder(name, smiles):
    """A function to calculate 3-dimensional coordinates of a molecule from its SMILES string."""

    print("---------------------------------------------------------")
    print(f"[ccdc_embedder] Working on {name}")
    print("---------------------------------------------------------")

    mol = None

    # Read in smiles
    try:
        mol = Molecule.from_string(smiles)
        mol.add_hydrogens()  # Take care: I think this is adding H atoms to Si(II) !!!
    except:
        print("    [ccdc_embedder] Provided SMILES string could not be converted to CCDC mol object.")
    
    if mol is not None:
        # Embed molecule
        conformer_generator = conformer.ConformerGenerator()
        conformer_generator.settings.max_conformers = 1
        conformers = conformer_generator.generate(mol)

        if conformers[0].molecule.all_atoms_have_sites:
            with io.EntryWriter(os.path.join("manual_pams_out", "mol_out", f"{name}.mol")) as f:
                f.write(conformers[0].molecule)
            print("    [ccdc_embedder] 3D embedding was successful.")
        else:
            print("    [ccdc_embedder] 3D embedding failed.")
    
    print()
