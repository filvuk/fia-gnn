# This is a modified version of xyz2graph. The original can be found here:
# https://github.com/zotko/xyz2graph

from itertools import combinations
from math import sqrt
import numpy as np

atomic_radii = dict(
    Al=1.21,
    As=1.19,
    B=0.84,
    Bi=1.48,
    Br=1.20,
    C=0.76,
    Cl=1.02,
    F=0.57,
    Ga=1.22,
    Ge=1.20,
    H=0.31,
    I=1.39,
    In=1.42,
    N=0.71,
    O=0.66,
    P=1.07,
    Pb=1.46,
    S=1.05,
    Sb=1.39,
    Se=1.20,
    Si=1.11,
    Sn=1.39,
    Te=1.38,
)


class MolGraph:
    """
    Represents a molecular graph.
    """

    def __init__(self):
        self.xyz_file_content = None
        self.elements = []
        self.x = []
        self.y = []
        self.z = []
        self.adj_list = np.empty(shape=(0, 2), dtype=np.int32)  # {}
        self.atomic_radii = []
        self.bond_lengths = {}

        self.molecular_charge = ""

        self.bond_length_scaler = 1.2

    def read_xyz(self, file_path):
        """
        Reads an XYZ file, searches for elements and their cartesian coordinates
        and add them to corresponding arrays.
        """
        with open(file_path) as f:
            self.xyz_file_content = f.readlines()

    def format_xyz(self):
        """
        Format the xyz data into individual lists
        """
        for line in self.xyz_file_content[2:]:
            if len(line) != 0:
                splitted = line.split()
                self.elements.append(splitted[0])
                self.x.append(float(splitted[1]))
                self.y.append(float(splitted[2]))
                self.z.append(float(splitted[3]))

        self.atomic_radii = [atomic_radii[element] for element in self.elements]

    def generate_adjacency_list(self):
        """
        Generates an adjacency list from atomic cartesian coordinates.
        """
        node_ids = range(len(self.elements))
        for i, j in combinations(node_ids, 2):
            x_i, y_i, z_i = self.__getitem__(i)[1]
            x_j, y_j, z_j = self.__getitem__(j)[1]
            distance = sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2 + (z_i - z_j) ** 2)
            if (
                0.1
                < distance
                < (self.atomic_radii[i] + self.atomic_radii[j])
                * self.bond_length_scaler
            ):
                bond = np.array([[i, j]], dtype=np.int32)
                self.adj_list = np.append(self.adj_list, bond, axis=0)
                self.adj_list = np.unique(self.adj_list, axis=0)

    def __getitem__(self, position):
        return self.elements[position], (
            self.x[position],
            self.y[position],
            self.z[position],
        )
