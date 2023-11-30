import os
from subprocess import run, TimeoutExpired


# Creat output directories
if not os.path.isdir("manual_pams_out"):
    os.mkdir("manual_pams_out")
    os.mkdir(os.path.join("manual_pams_out", "mol_out"))
    print("[ccdc_embedder] Main output directory was created.")
    print("[ccdc_embedder] Mol file output directory was created.")
else:
    print("[ccdc_embedder] Main output directory already exists.")

if not os.path.isdir(os.path.join("manual_pams_out", "mol_out")):
    os.mkdir(os.path.join("manual_pams_out", "mol_out"))
    print("[ccdc_embedder] Mol file output directory was created.")


def ccdc_embedder(name, smiles):
    """
    Wrapper function around the embedding process. The process was found to
    potentially crash when being directly called. Doing it within a subprocess
    prevented this issue. Also, the CCDC embedder can end up hanging (forever).
    Therefore, a 30 second time out was added.
    """
    print("---------------------------------------------------------")
    print(f"[ccdc_embedder] Working on {name}")
    print("---------------------------------------------------------")

    try:
        result = run(
            f"python do_ccdc_embedding.py {name} {smiles}", shell=True, timeout=30
        )
        result = result.returncode
    except TimeoutExpired:
        result = -3
    finally:
        if result == 0:
            print("    [ccdc_embedder] 3D embedding was successful.")
        if result == -1:
            print("    [ccdc_embedder] 3D embedding failed.")
        if result == -2:
            print(
                "    [ccdc_embedder] Provided SMILES string could not be converted to CCDC mol object."
            )
        if result == -3:
            print(
                "    [ccdc_embedder] 3D embedding failed due to 30 second time limit."
            )
        print()
