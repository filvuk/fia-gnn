import os
import numpy as np  # 1.23.5

from scipy.spatial import distance  # 1.9.1
from scipy.optimize import minimize


# Create output directory
if not os.path.isdir("auto_pams_fa_xyz_out"):
    os.mkdir("auto_pams_fa_xyz_out")
    print("[auto_pams_fa] XYZ output directory was created.")
else:
    print("[auto_pams_fa] XYZ output directory already exists.")


class AutoPAMS_FA:
    """
    A class to geometrically calculate a xyz starting structure of a fluoride adduct
    from the optimized Lewis acid structure.

    Parameters
    ----------
    la_struc_file: str
        Path the the .xyz file of the Lewis acid, must be in XMOL format.
    atom_connectivities: dict
        Dictionary of atom connectivities which must contain the connectivities of the currently
        treated data point.
    """

    def __init__(self, la_struc_file, atom_connectivities):
        self.la_struc_file = la_struc_file
        self.atom_connectivities = atom_connectivities

        self.name = os.path.basename(self.la_struc_file).split(".")[0]

        self.error = None

        self._la_num_atoms = None
        self._la_atom_list = None
        self._la_coords = None
        self._shift_vec = None

        self._opt_solution = None

        self.fa_atom_connectivities = None

    def __call__(self):
        """Run the fluoride adduct generation."""

        # Element specific distances between central atom and added fluoride
        el_f_dist = {
            "Al": 1.68,
            "As": 1.82,
            "B": 1.41,
            "Bi": 2.11,
            "Ga": 1.78,
            "Ge": 1.8,
            "In": 1.98,
            "P": 1.69,
            "Pb": 2.09,
            "Sb": 1.98,
            "Si": 1.68,
            "Sn": 1.98,
            "Te": 2.01,
        }

        print("---------------------------------------------------------")
        print(f"[auto_pams_fa] Working on {self.name}")
        print("---------------------------------------------------------")

        # Get central atom and the element-specific distance
        central_atom = self.name.split("_")[0]
        if central_atom in el_f_dist:
            dist_constraint = el_f_dist[central_atom]
        else:
            dist_constraint = 1.85
            print(
                f"    [auto_pams_fa] Central atom '{central_atom}' was not found in the total scope of the central atoms.\n    Defaulting to 1.85 A as the distance constraint for optimization."
            )

        # Read the xyz file of the Lewis acid
        self.read_la_struc_file()
        if self.error is None:
            # Calculate the fluoride adduct structure
            self.run_optimization(dist_constraint=dist_constraint)
            if self._opt_solution.success == True:
                # Get the atom connectivities of the fluoride adduct from those of the Lewis acid
                self.get_fluoride_adduct_connectivities()
                if self.error is None:
                    # Save the fluoride adduct structure as xyz file
                    self.save_fluoride_adduct_structure()
                    print(
                        "    [auto_pams_fa] Fluoride adduct was successfully generated and saved."
                    )
                else:
                    print(f"    [auto_pams_fa] {self.error}")
            else:
                print(
                    "    [auto_pams_fa] Structure generation of the fluoride adduct failed."
                )

        else:
            print(f"    [auto_pams_fa] {self.error}")

        print()
        return self.atom_connectivities

    def read_la_struc_file(self):
        """Read the xyz file of the Lewis acid and format the data."""
        if os.path.isfile(self.la_struc_file):
            with open(self.la_struc_file, "r") as f:
                la = f.readlines()
        else:
            self.error = (
                f"XYZ file of the Lewis acid was not found at {self.la_struc_file}."
            )

        if self.error is None:
            self._la_num_atoms = int(la[0])
            self._la_atom_list = [line.split()[0] for line in la[2:]]

            # Get the coordinates of the Lewis acid and shift the 0th atom (central atom) to (0,0,0)
            la_coords = np.array(
                [line.split()[1:] for line in la[2:]], dtype=np.float32
            )
            self._shift_vec = la_coords[0]
            self._la_coords = la_coords - self._shift_vec

    def run_optimization(self, dist_constraint):
        """
        The distance of the added fluoride atom to its nearest neighbor
        (excluding the central atom (0th atom) and obviously itself (-1st atom))
        is maximized under the constraint of an element-specific central-atom-fluoride distance.
        """

        def cost_function(f_coords):
            new_coords = np.append(self._la_coords, f_coords.reshape(1, 3), axis=0)
            f_distances = distance.cdist(new_coords, new_coords, "euclidean")[-1][1:-1]
            return -np.min(f_distances)

        def constraint_function(f_coords):
            return np.linalg.norm(f_coords) - dist_constraint

        self._opt_solution = minimize(
            cost_function,
            np.array([10, 10, 10]),
            constraints=[{"type": "eq", "fun": constraint_function}],
            options={"maxiter": 999999},
        )

    def save_fluoride_adduct_structure(self):
        """Save the calculated structure of the fluoride adduct as a .xyz file."""

        def get_final_coords(la_num_atoms, la_atom_list, la_coords, f_coords):
            """Formatting the final coords to XMOL format."""
            la_atom_list.append("F")
            coords = np.append(la_coords, f_coords, axis=0)

            # Formatting
            final_coords = f"{la_num_atoms+1}\n\n"
            for idx, atom in enumerate(la_atom_list):
                x_y_z = coords[idx].astype(str)
                x_y_z = "    ".join(x_y_z)
                line = f"  {atom}    {x_y_z}\n"
                final_coords += line

            return final_coords

        def write_fa_xyz_file(name, final_coords):
            """Write the final file."""
            path = os.path.join("auto_pams_fa_xyz_out", f"{name}__F.xyz")
            with open(path, "w") as f:
                f.write(final_coords)

        # Save atom connectivities
        self.atom_connectivities[f"{self.name}__F"] = self.fa_atom_connectivities

        # Save xyz coordinates
        f_coords = self._opt_solution.x + self._shift_vec
        f_coords = f_coords.reshape(1, 3)
        self._la_coords = self._la_coords + self._shift_vec

        final_coords = get_final_coords(
            self._la_num_atoms, self._la_atom_list, self._la_coords, f_coords
        )
        write_fa_xyz_file(self.name, final_coords)

    def get_fluoride_adduct_connectivities(self):
        """
        Get the atom connectivities of the fluoride adduct from
        those of the Lewis acid (must be provided).
        """
        if self.name in self.atom_connectivities:
            la_atom_connectivities = self.atom_connectivities[self.name]

            fa_atom_connectivities = np.copy(la_atom_connectivities)
            F_atom_idx = np.max(fa_atom_connectivities) + 1
            fa_atom_connectivities = np.insert(
                fa_atom_connectivities,
                len(fa_atom_connectivities),
                np.array([0, F_atom_idx]),
                axis=0,
            )

            # Sorting
            fa_atom_connectivities.sort(axis=1)
            self.fa_atom_connectivities = fa_atom_connectivities[
                np.lexsort(
                    (
                        fa_atom_connectivities[:, 1],
                        fa_atom_connectivities[:, 0],
                    )
                )
            ]

        else:
            self.error = f"Atom connectivities of {self.name} were not found in the provided connectivities.\n    Fluoride atom connectivities cannot be calculated. No data was saved."
