from abc import ABC, abstractmethod
import numpy as np
from lammps import lammps, PyLammps
from mpi4py import MPI


PROJECT_PATH = "/home/nagibator69/science/CollisionProject/"


def simple_box_creator(sizes, fcc_period, index=0):
    """
    Build a simple FCC box, dump XYZ, and return atom positions.

    Args:
        sizes (tuple of float): Number of unit cells in (X,Y,Z).
        fcc_period (float): FCC lattice constant.
        index (int): Scene index to differentiate filenames.

    Returns:
        ndarray of shape (n_atoms, 4): Columns [type, x, y, z].
    """
    path = PROJECT_PATH + f"tools/create_scenes/data/test_scene_{index}.txt"
    X, Y, Z = sizes

    lmp = lammps()  # comm = MPI.COMM_WORLD
    L = PyLammps(ptr=lmp)

    # L.command("neigh_modify delay 1 every 1 check yes")
    L.command("boundary p p p")
    L.command("comm_style tiled")
    L.units("metal")
    L.atom_style("molecular")
    L.pair_style("hybrid lj/cut 4")  # + "".join([" lj/cut 4"] * types_number))
    L.bond_style("harmonic")

    L.command(f"lattice fcc {fcc_period}")
    L.region(f"box block 0 {X} 0 {Y} 0 {Z} units lattice")
    L.create_box(f"1 box")

    L.region(f"box0 block 0 {X} 0 {Y} 0 {Z} units lattice")
    L.create_atoms(f"{1} region box0")

    #     L.region(f"box block 0 {X} 0 {Y} 0 {Z} units lattice")
    #     L.create_box("1 box")
    #     L.command(f"lattice fcc {fcc_period}")
    #     L.region(f"box0 block 0 {X} 0 {Y} 0 {Z} units lattice")
    #     L.create_atoms(f"{1} region box0")

    L.command(f"write_dump all xyz {path}")
    comm = MPI.COMM_WORLD
    comm.Barrier()
    f = open(path, "r")
    res = []
    for line in f.readlines()[2:]:
        res.append(list(map(float, line.split())))

    f.close()
    res = np.array(res)
    return res


class SimpleData(ABC):
    """
    Abstract base for writing LAMMPS data files.

    Subclasses must implement calculate_parameters() and main().
    """

    def __init__(self, file_name="new_file", kwargs=dict()):
        self.data_file = "LAMMPS data file via write_data, version 29 Sep 2021\n\n"
        self.file_name = file_name
        self.kwargs = kwargs
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

    @abstractmethod
    def calculate_parameters(self):
        """Compute self.atoms, self.velocities, box bounds, masses, etc."""
        pass

    def create_file(self):
        """
        Assemble header, Masses, Atoms, Velocities sections and write to disk.
        """

        mult = len(set(self.atoms[:, 0])) // self.atom_types

        self.data_file += f"{self.atoms_num} atoms\n"
        self.data_file += f"{self.atom_types * mult} atom types\n"
        self.data_file += f"{(2*self.atom_types-1)*mult} bond types\n\n"

        self.data_file += f"{self.xlo} {self.xhi} xlo xhi\n"
        self.data_file += f"{self.ylo} {self.yhi} ylo yhi\n"
        self.data_file += f"{self.zlo} {self.zhi} zlo zhi\n\n"

        self.data_file += "Masses\n\n"
        for i in range(len(self.masses)):
            self.data_file += f"{i+1} {self.masses[i]}\n"
        self.data_file += "\n"

        self.data_file += "Atoms # molecular\n\n"
        for i in range(len(self.atoms)):
            atom_type = int(self.atoms[i][0])
            self.data_file += (
                f"{i+1} "
                + "0 "
                + f"{atom_type} "
                + " ".join(list(self.atoms[i][1:].astype(np.str_)))
                + " 0 0 0"
                + "\n"
            )

        self.data_file += "\nVelocities # molecular\n\n"
        for i in range(len(self.atoms)):
            atom_type = int(self.velocities[i][0])
            self.data_file += (
                f"{i+1} " + " ".join(list(self.velocities[i].astype(np.str_))) + "\n"
            )
        if self.rank == 0:
            text_file = open(self.file_name, "w")
            text_file.write(self.data_file)
            text_file.close()

    @abstractmethod
    def main(self):
        """Top-level: call calculate_parameters() then create_file()."""
        pass


class MultilayerData(SimpleData):
    """
    Build multilayer structures by stacking scaled FCC boxes.

    Extends SimpleData to compute multiple atom types, positions, and bounds.
    """

    def get_inter_layer_dist(self, a, k=2):
        b = k * a
        h2 = (2 * a * b + b**2 - a**2) / 8
        return h2**0.5

    def calculate_parameters(self):
        self.width_x, self.width_y, self.width_z = self.kwargs["width"]
        self.atom_types = self.kwargs["types_number"]
        self.fcc_period = self.kwargs["fcc_period"]
        self.masses = self.kwargs["masses"]

        z = 0

        self.atoms = simple_box_creator(
            sizes=[self.width_x, self.width_y, self.width_z], fcc_period=self.fcc_period
        )
        max_x, max_y, max_z = np.max(self.atoms[:, 1:], axis=0)
        step_atoms = self.atoms.copy()
        self.all_masses = np.ones(shape=step_atoms.shape[0]) * self.masses[0]
        for atom_type in range(1, self.atom_types):
            new_atoms = step_atoms.copy()
            new_atoms[:, 0] += atom_type
            new_atoms[:, 1:] *= 2**atom_type
            new_atoms[:, -1] += sum(
                [
                    max_z * 2**i + self.get_inter_layer_dist(self.fcc_period * 2**i)
                    for i in range(atom_type)
                ]
            )
            new_atoms = new_atoms[new_atoms[:, 1] <= max_x]
            new_atoms = new_atoms[new_atoms[:, 2] <= max_y]
            self.all_masses = np.append(
                self.all_masses,
                np.ones(shape=new_atoms.shape[0]) * self.masses[atom_type],
                axis=0,
            )
            self.atoms = np.append(self.atoms, new_atoms, axis=0)
            # new_fcc *= 2

        self.atoms_num = self.atoms.shape[0]

        x, y, z = self.atoms[:, 1:].T

        self.velocities = np.array([[0, 0, 0] for i in range(self.atoms_num)])

        assert (
            self.velocities.shape[0] == self.atoms.shape[0]
        ), "Number of atoms doesn't correspond to velocities"

        self.xlo, self.xhi = 0, self.width_x * self.fcc_period  # + self.fcc_period / 2
        self.ylo, self.yhi = 0, self.width_y * self.fcc_period  # + self.fcc_period / 2
        # self.zlo, self.zhi = z.min(), z.max()  # 2**(self.atom_types-1)/2**0.5
        # self.zlo, self.zhi = z.min() - self.get_inter_layer_dist(self.fcc_period, k=2**(self.atom_types-1)), z.max()
        self.zlo, self.zhi = z.min() - 0.9 * self.fcc_period / np.sqrt(
            2
        ), z.max() + 0.9 * self.fcc_period / np.sqrt(2) * 2 ** (self.atom_types - 1)

    def main(self):
        self.calculate_parameters()
        self.create_file()
