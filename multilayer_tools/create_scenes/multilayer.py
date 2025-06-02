from multilayer_tools.create_scenes.datafile_creator import MultilayerData
import numpy as np
from lammps import lammps, PyLammps


def get_inter_layer_dist(a, k=2):
    """
    Compute vertical separation between successive FCC layers.

    Args:
        a (float): FCC lattice parameter.
        k (int): Relative scaling factor for layer spacing.

    Returns:
        float: Inter-layer distance h = sqrt[(2a·(k a) + (k a)^2 − a^2) / 8].
    """
    b = k * a
    h2 = (2 * a * b + b**2 - a**2) / 8
    return h2**0.5


def upper_layer_coord(atom_type, h, fcc_period):
    """
    Compute the z-coordinate of the upper boundary for a given atom layer.

    Args:
        atom_type (int): 1-based layer index.
        h (float): Base height increment.
        fcc_period (float): Lattice constant of the FCC lattice.

    Returns:
        float: Sum of vertical shifts for stacking and inter-layer spacing.
    """
    z = [fcc_period * (h - 0.5) * 2**i for i in range(atom_type)]
    cross_z = [get_inter_layer_dist(fcc_period) * 2**i for i in range(atom_type - 1)]

    return sum(z + cross_z)


def create_multilayer(index=0, **particle_kwargs):
    """
    Generate a multilayer atomic system, write LAMMPS data, and initialize PyLammps.

    Args:
        index (int): Scene index (unused here).
        **particle_kwargs: Must include:
            - file_name (str): Output data filename.
            - types_number (int): Number of distinct atomic layers/types.
            - width (tuple of 3 ints): Number of unit cells in x,y,z for base layer.
            - fcc_period (float): FCC lattice constant.
            - masses (list of floats): Atomic masses per layer.
    Returns:
        PyLammps: LAMMPS interface object with region and groups defined.
    """
    assert "file_name" in particle_kwargs, "file_name param has to be specified!"
    file_name = particle_kwargs["file_name"]
    mlt = MultilayerData(
        file_name=file_name,
        kwargs=particle_kwargs,
    )
    mlt.main()
    lmp = lammps()  # comm = MPI.COMM_WORLD
    L = PyLammps(ptr=lmp)
    L.atom_style("molecular")
    L.units("metal")
    L.pair_style("hybrid/overlay lj/cut 25")
    L.bond_style("harmonic/restrain")
    L.region("box block 0 1 0 1 0 1 units lattice")

    types_number = particle_kwargs["types_number"]
    L.create_box(
        f"{types_number} box bond/types {2*types_number-1} extra/bond/per/atom 30"
    )
    L.read_data(file_name + " add merge")
    for i in range(1, particle_kwargs["types_number"] + 1):
        L.group(f"{i} type {i}")

    h = particle_kwargs["width"][2]
    fp = particle_kwargs["fcc_period"]
    for atom_type in range(1, particle_kwargs["types_number"] + 1):

        u_layer = upper_layer_coord(
            atom_type,
            h=h,
            fcc_period=fp,
        )
        print(
            f"New upper layer on {u_layer-fp**(2*(atom_type-1))/10} {u_layer+fp**(2*(atom_type-1))/10}"
        )
        L.region(
            f"upper_layer_{atom_type} block {u_layer-fp**(2*(atom_type-1))/10} {u_layer+fp**(2*(atom_type-1))/10} INF INF INF INF"
        )
        L.group(f"upper_layer_{atom_type} region upper_layer_{atom_type}")

    return L
