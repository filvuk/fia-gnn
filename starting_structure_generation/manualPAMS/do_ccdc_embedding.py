import os
import argparse
from ccdc import io
from ccdc import conformer
from ccdc.molecule import Molecule


def embed(name, smiles):
    """A function to calculate 3-dimensional coordinates of a molecule from its SMILES string."""
    mol = None
    returncode = -1

    # Read in smiles
    try:
        mol = Molecule.from_string(smiles)
    except:
        returncode = -2

    if mol is not None:
        # Embed molecule
        conformer_generator = conformer.ConformerGenerator()
        conformer_generator.settings.max_conformers = 1
        conformers = conformer_generator.generate(mol)

        if conformers[0].molecule.all_atoms_have_sites:
            with io.EntryWriter(
                os.path.join("manual_pams_out", "mol_out", f"{name}.mol")
            ) as f:
                f.write(conformers[0].molecule)
            returncode = 0

    return returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("smiles")
    args = parser.parse_args()

    embed(args.name, args.smiles)
