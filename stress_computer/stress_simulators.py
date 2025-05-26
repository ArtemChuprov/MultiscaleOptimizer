from lammps import lammps, PyLammps
import numpy as np
from IPython.display import clear_output
import os
from shutil import rmtree
from sklearn.linear_model import LinearRegression
from abc import ABC, abstractmethod
from time import sleep
from collections.abc import Iterable


class BasicStressSimulator:
    def __init__(
        self,
        types_number: int,
        dump_flg: bool = False,
        index: int = 0,
        seed: int = 0,
    ):
        self.types_number = types_number
        self.dump_flg = dump_flg
        self.index = index
        self.seed = seed

    def compute_young_uniaxial(self, L: PyLammps, direction="zz", delta=0.01):
        """
        Zero-temperature uniaxial test:
        - Apply strain delta along 'direction'
        - Relax other two directions to zero stress via fix box/relax
        - Return E = Δσ / Δε
        """
        # 1. Equilibrate at zero external pressure
        L.fix(
            "relax0 all box/relax aniso 0.0 x 0.0 y 0.0 z 0.0"
        )  # hold all stresses at zero
        L.command("minimize 1e-3 1e-3 10000 10000")  # minimize energy & box
        L.run(0)
        L.unfix("relax0")
        # Compute baseline stress
        L.compute("peratom 1 stress/atom NULL")
        L.compute("sig0 all reduce ave c_peratom[3]")  # zz component index=3
        L.run(0)
        sigma0 = L.eval("c_sig0")

        # 2. Deform box along 'direction'
        axis = direction[0]  # 'x','y','z'
        L.change_box(f"all boundary p p p {axis} scale {1+delta} remap")  # apply δε
        # 3. Relax lateral stresses only
        # target zero in other two axes:
        lat = {"x", "y", "z"} - {axis}
        targ = " ".join(f"{d} 0.0" for d in sorted(lat))
        print(targ)
        L.fix("relax1 all box/relax " + targ)
        L.command("minimize 1e-3 1e-3 10000 10000")
        L.run(0)
        L.unfix("relax1")

        # 4. Measure stressed state
        L.compute("sig1 1 reduce ave c_peratom[3]")
        L.run(0)
        sigma1 = L.eval("c_sig1")

        # 5. Compute Young's modulus
        E = (sigma1 - sigma0) / delta
        L.write_data("final_shape")
        return E
