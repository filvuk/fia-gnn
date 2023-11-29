import os
import pandas as pd

from functools import partial
from pathlib import Path
from rdkit import Chem
from fia_gnn_preprocessing import FiaGnnPreprocessor, get_nfp_preprocessor
from tqdm import tqdm

tqdm.pandas()


def get_input_dict(df, prop, fia_gas_preprocessor, fia_solv_preprocessor, train=True):
    """
    A function that gets the actual data from the FiaGnnPreprocessor class and
    returns it as a dictionary to the run_preprocessing function.
    """
    smiles = df.la_smiles.iloc[0]
    mol = Chem.MolFromSmiles(smiles)

    try:
        # Get the inputs for the fia_gas model from the preprocessing class
        if prop == "fia_gas-DSDBLYP":
            input_dict = FiaGnnPreprocessor(
                mol=mol,
                fia_gas_preprocessor=fia_gas_preprocessor,
                fia_solv_preprocessor=fia_solv_preprocessor,
                train=train,
            ).fia_gas_input

            # Update the distance to central atom info in the fia_gas preprocessor
            max_ = float(max(input_dict["dist_to_central_atom"]))
            if max_ > fia_gas_preprocessor.max_atom_distance:
                fia_gas_preprocessor.max_atom_distance = max_

        # Get the inputs for the fia_solv model from the preprocessing class
        if prop == "fia_solv-DSDBLYP":
            input_dict = FiaGnnPreprocessor(
                mol=mol,
                fia_gas_preprocessor=fia_gas_preprocessor,
                fia_solv_preprocessor=fia_solv_preprocessor,
                train=train,
            ).fia_solv_input

            # Update the distance to central atom info in the fia_solv preprocessor
            max_ = float(max(input_dict["dist_to_central_atom"]))
            if max_ > fia_solv_preprocessor.max_atom_distance:
                fia_solv_preprocessor.max_atom_distance = max_

    except:
        input_dict = {}
        print(f"PREPROCESSING ERROR FOR '{smiles}'.")

    targets = df[prop].values[0]
    input_dict["output"] = [targets]

    return input_dict


def run_preprocessing(
    data_file_path,
):
    """
    A function to run the preprocessing prior to model training.
    The provided data file must contain a column called "la_smiles" (SMILES string of the Lewis acid)
    as well as columns called fia_gas-DSDBLYP and fia_solv_DSDBLYP with the FIA values
    to be learned.
    """
    property_names = ["fia_gas-DSDBLYP", "fia_solv-DSDBLYP"]
    preprocessor_names = ["preprocessor_fia_gas", "preprocessor_fia_solv"]

    preprocessor_fia_gas = get_nfp_preprocessor()
    preprocessor_fia_solv = get_nfp_preprocessor()

    print(f"Loading data file at {data_file_path} ...")
    data = pd.read_csv(data_file_path, low_memory=False)
    print("Done.")
    print()

    print("Doing preprocessing ...")
    for property_, preprocessor in zip(property_names, preprocessor_names):
        print(f"    Working on '{property_}' ...")
        mol_df = pd.DataFrame(data.groupby("la_smiles").set_assignment.first())

        # Do the preprocessing for the train set and train the preprocessor
        train = (
            data[data.set_assignment == "train"]
            .groupby("la_smiles")
            .progress_apply(
                partial(
                    get_input_dict,
                    prop=property_,
                    fia_gas_preprocessor=preprocessor_fia_gas,
                    fia_solv_preprocessor=preprocessor_fia_solv,
                    train=True,
                )
            )
        )

        # Do the preprocessing for everything but the train set. Preprocessor is not trained.
        not_train = (
            data[data.set_assignment != "train"]
            .groupby("la_smiles")
            .progress_apply(
                partial(
                    get_input_dict,
                    prop=property_,
                    fia_gas_preprocessor=preprocessor_fia_gas,
                    fia_solv_preprocessor=preprocessor_fia_solv,
                    train=False,
                )
            )
        )

        # Formatting and saving the data
        inputs = pd.concat([train, not_train])
        inputs.name = "model_inputs"
        mol_df = mol_df.join(inputs)

        if not os.path.exists(Path(os.getcwd(), f"preprocessed_data_{property_}")):
            os.makedirs(Path(os.getcwd(), f"preprocessed_data_{property_}"))

        mol_df.to_pickle(
            Path(
                os.getcwd(),
                f"preprocessed_data_{property_}/{preprocessor}_model_inputs.pkl",
            )
        )

        if property_ == "fia_gas-DSDBLYP":
            preprocessor_fia_gas.to_json(
                Path(os.getcwd(), f"preprocessed_data_{property_}/{preprocessor}.json")
            )

        if property_ == "fia_solv-DSDBLYP":
            preprocessor_fia_solv.to_json(
                Path(os.getcwd(), f"preprocessed_data_{property_}/{preprocessor}.json")
            )

        print("    Done.")
        print()


if __name__ == "__main__":
    print("#########################")
    print("# FIA-GNN Preprocessing #")
    print("#########################")
    print()

    path = os.path.split(os.getcwd())[0]
    path = os.path.join(path, "data", "FIA49k.csv.gz")

    run_preprocessing(data_file_path=path)

    print("Preprocessing finished.")
    print()
