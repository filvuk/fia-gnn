import numpy as np
import xyz2graph


class ValenceAngle:
    """
    A class that can be used to geometrically analyze bond angles around a given atom.
    ...
    Parameters
    ----------
    xyz_file_content: list
        File content of a .xyz file in XMOL format as obtained from file.readlines()
    """

    def __init__(self, xyz_file_content):
        self.xyz_file_content = xyz_file_content

        self.atom_list = None
        self.neighbors_indices = None
        self.neighbors_atoms = None
        self.xyz_coords = None
        self.central_atom_vector = None
        self.neighbors_vectors = None
        self.angle_pairs = None
        self.all_valence_angles = None
        self.valence_angle_sum = None
        self.all_fluoride_valence_angles = None
        self.max_fluoride_valence_angle = None

        self.get_mol_graph()
        self.get_atom_angle_pairs()
        self.get_all_valence_angles()

    def get_mol_graph(self):
        """
        Generate a graph of the molecule based on atom distances.
        """
        graph = xyz2graph.MolGraph()
        graph.xyz_file_content = self.xyz_file_content
        graph.format_xyz()
        graph.generate_adjacency_list()

        self.atom_list = np.array(graph.elements)

        self.neighbors_indices = []
        for pair in graph.adj_list:
            # The central atom has atom idx 0 and it always comes first
            if pair[0] == 0:
                self.neighbors_indices.append(pair[1])
            else:
                break

        self.neighbors_atoms = self.atom_list[self.neighbors_indices]
        self.xyz_coords = np.array([graph.x, graph.y, graph.z]).T
        self.central_atom_vector = self.xyz_coords[0]
        self.neighbors_vectors = self.xyz_coords[self.neighbors_indices]

    def get_atom_angle_pairs(self):
        """
        Get the pair of atoms which are attached to the central atom which define the angle.
        """
        self.angle_pairs = []
        for atom_idx_1, atom_symbol_1 in zip(
            self.neighbors_indices, self.neighbors_atoms
        ):
            for atom_idx_2, atom_symbol_2 in zip(
                self.neighbors_indices, self.neighbors_atoms
            ):
                if atom_idx_1 < atom_idx_2:
                    pair = [
                        f"{atom_symbol_1}-{atom_idx_1}",
                        f"{atom_symbol_2}-{atom_idx_2}",
                    ]
                    self.angle_pairs.append(pair)

    def get_all_valence_angles(self):
        """
        Get all valence angles of the central atom.
        """
        self.all_valence_angles = {}
        for p in self.angle_pairs:
            angle_name = (p[0], f"{self.atom_list[0]}-0", p[1])
            self.all_valence_angles[angle_name] = self.get_angle(
                self.central_atom_vector,
                self.xyz_coords[int(p[0].split("-")[-1])],
                self.xyz_coords[int(p[1].split("-")[-1])],
            )
        self.all_valence_angles = dict(
            sorted(self.all_valence_angles.items(), key=lambda x: x[1], reverse=True)
        )
        self.valence_angle_sum = sum(self.all_valence_angles.values())

    def get_element_F_angles(self):
        """
        Identify the valence angles of the central atom that contain a F atom.
        """
        self.all_fluoride_valence_angles = {}
        fluoride_name = f"F-{len(self.atom_list)-1}"
        for angle_name, angle in self.all_valence_angles.items():
            for atom in angle_name:
                if atom == fluoride_name:
                    self.all_fluoride_valence_angles[angle_name] = angle

        self.max_fluoride_valence_angle = max(self.all_fluoride_valence_angles.values())

    @staticmethod
    def get_angle(s, a, b):
        """
        Calculate the angle between two lines defined by three points, that cross in s.
        """
        u = np.array(a) - np.array(s)
        v = np.array(b) - np.array(s)
        return np.degrees(
            np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        )
