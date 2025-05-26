from multilayer_tools.create_scenes.datafile_creator import MultilayerData
import numpy as np
from lammps import lammps, PyLammps


def get_inter_layer_dist(a, k=2):
    b = k * a
    h2 = (2 * a * b + b**2 - a**2) / 8
    return h2**0.5


def upper_layer_coord(atom_type, h, fcc_period):
    z = [fcc_period * (h - 0.5) * 2**i for i in range(atom_type)]
    cross_z = [get_inter_layer_dist(fcc_period) * 2**i for i in range(atom_type - 1)]

    return sum(z + cross_z)


def create_multilayer(index=0, **particle_kwargs):
    if "file_name" not in particle_kwargs:
        file_name: str = (
            f"/trinity/home/artem.chuprov/projects/Multiscale_BlackBox/data/Cu/Multilayer_Step_{index}"
        )
    else:
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
