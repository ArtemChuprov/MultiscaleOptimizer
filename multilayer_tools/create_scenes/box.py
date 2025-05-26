from lammps import lammps, PyLammps
import math

def create_box(
    X: tuple,
    Y: tuple,
    Z: tuple,
    types_number: int,
    fcc_period: float = 3.615,
    mass: float = 63.546,
    index: int = 0,
):
    xlo, xhi = X
    ylo, yhi = Y
    zlo, zhi = Z

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

    L.region(f"box block {xlo} {xhi} {ylo} {yhi} {zlo} {zhi} units lattice")
    L.create_box(
        f"{types_number} box bond/types {2*types_number-1} extra/bond/per/atom 15"
    )
    L.create_atoms(f"{types_number} box")
    L.command(f"mass * {mass}")

    delta_x = (xhi - xlo) / types_number
    delta_y = (yhi - ylo) / types_number
    delta_z = (zhi - zlo) / types_number

    for i in range(0, types_number):
        # L.region(f'box{i} block {xlo+i*delta_x} {xlo+(i+1)*delta_x} {ylo+i*delta_y} {ylo+(i+1)*delta_y} {zlo+i*delta_z} {zlo+(i+1)*delta_z} units lattice')
        L.region(
            f"box{i+1} block {xlo} {xhi} {ylo} {yhi} {zlo+i*delta_z} {zlo+(i+1)*delta_z} units lattice"
        )
        L.group(f"{i+1} region box{i+1}")
        L.command(f"set group {i+1} type {i+1}")

    return L


# if dump:
#     L.command(f"write_data create_scenes/data/{name}")
# else:
#     return L


if __name__ == "__main__":
    create_box(X=(0, 10), Y=(10, 20), Z=(20, 30), types_number=3)
