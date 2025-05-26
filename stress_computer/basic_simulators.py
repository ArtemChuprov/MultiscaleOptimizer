from lammps import lammps, PyLammps
import numpy as np
from IPython.display import clear_output
import os
from shutil import rmtree
from sklearn.linear_model import LinearRegression
from abc import ABC, abstractmethod
from time import sleep
from collections.abc import Iterable
import os


class SimpleBasicSimulator(ABC):
    def __init__(
        self,
        potential_params: dict,
        scene_creator=None,
        scene_params=dict(),
        minimize: int = 5000,
        file_path=None,
        path_dump: str = "Dump/coords",
        index: int = 0,
        seed: int = 0,
    ):
        """
        params:
        - potential_params (dict): a dict of dicts with params for potentials used in simulation, potential-name -> coefficient-name -> values.
        - scene_creator: a class used to create a lammps scene. If None, the method has to be clarified in a simulation method.
        - timestep (float): the size of one timestep.
        - minimize (int): number of maximum minimization steps.
        - file_path (str): a path to a scene file to read instead of scene_creator.
        - path_dump (str): path to a folder where to save dumps.
        - index (int): index of a simulator, used for parallel computing.
        - seed (int): random seed used in all random places.
        """
        self.potential_params = potential_params
        self.minimize = minimize
        self.scene_creator = scene_creator
        self.scene_params = scene_params.copy()
        self.types_number = scene_params["types_number"]
        self.file_path = file_path
        self.path_dump = path_dump
        self.index = index
        self.seed = seed
        files = [
            f
            for f in os.listdir(path_dump)
            if os.path.isfile(os.path.join(path_dump, f))
        ]
        for file in files:
            file_path = os.path.join(path_dump, file)
            try:
                os.remove(file_path)
            except:
                pass

        if isinstance(scene_params["fcc_period"], float):
            self.min_fcc_dist = [scene_params["fcc_period"] / np.sqrt(2)]
        else:
            self.min_fcc_dist = np.array(scene_params["fcc_period"].copy())
            self.scene_params["fcc_period"] = self.scene_params["fcc_period"][0]

    @abstractmethod
    def create_scene(self) -> PyLammps:
        """
        A method used to create a starting simulation scene. If file_path is set, that it will be read.
        Otherwise, scene_creator will be used.
        Returns PyLammps object, which can be used further.
        """
        pass

    @abstractmethod
    def set_potentials(self, L: PyLammps) -> PyLammps:
        """
        A method used to set potentials specific for the task using passed in the init paramters.
        Returns PyLammps object, which can be used further.
        """
        pass

    def basic_simulation(self) -> PyLammps:
        """
        A method used to perform basic lammps simulation.
        Returns PyLammps object, which can be used further.
        """
        L = self.scene_creator(index=self.index, **self.scene_params)
        L.dump(f"1 all custom 10 ./coords/dmp.*.txt id type x y z mass vx")
        L = self.set_potentials(L=L)
        L.fix("box_relax all box/relax iso 0 vmax 0.01")
        L.command(f"minimize 1e-3 1e-3 {self.minimize} 10000")
        # L.run(0)
        L.unfix("box_relax")
        L.write_data("start_shape")

        return L


class LJBasicSimulator(SimpleBasicSimulator):
    def create_scene(self):
        if self.file_path is None:
            return self.scene_creator(index=self.index, **self.scene_params)

    def set_potentials(self, L: PyLammps) -> PyLammps:
        eps, sigma, lj_cutoff = (
            self.potential_params["LJ"]["eps"],
            self.potential_params["LJ"]["sigma"],
            self.potential_params["LJ"]["lj_cutoff"],
        )
        add_LJ_walls = self.potential_params["LJ"]["walls"]
        k, cross_springs = (
            self.potential_params["harmonic"]["k"],
            self.potential_params["harmonic"]["cross_springs"],
        )
        N = self.types_number
        # L.bond_style("harmonic/restrain")

        bonds_counter = 1
        bonds = []
        for i in range(N):
            for j in range(N):
                if (j >= i) and abs(
                    i - j
                ) <= 1:  # or abs(i-j)==self.scene_params['types_number']):
                    if i == j:
                        # ONE-TYPE INTERACTION CASE

                        # Simple LJ interaction
                        new_eps, new_sigma = eps[i], sigma[i]

                        # Bond interaction
                        new_k, new_r0 = k[i], self.min_fcc_dist[i]

                        # Cutoff ditances
                        if isinstance(lj_cutoff, Iterable):
                            cutoff = lj_cutoff[i]
                        elif isinstance(lj_cutoff, int) or isinstance(lj_cutoff, float):
                            cutoff = lj_cutoff

                    else:
                        # CROSS INTERACTION CASE

                        # Lorentz-Berthelot LJ rules
                        new_eps, new_sigma = (
                            np.sqrt(eps[i] * eps[j]),
                            (sigma[i] + sigma[j]) / 2,
                        )

                        if not cross_springs:
                            # Sequent bonds
                            if k[i] > 0 or k[j] > 0:
                                new_k = (k[i] * k[j]) / (k[i] + k[j])
                            else:
                                new_k = 0
                        elif cross_springs:
                            new_k = max(k[self.types_number + i], 0)
                        # (self.min_fcc_dist[i] + self.min_fcc_dist[j]) / 2

                        # Cutoff ditances
                        if isinstance(lj_cutoff, list):
                            cutoff = (lj_cutoff[i] + lj_cutoff[j]) / 2 * 0.8
                        elif isinstance(lj_cutoff, int) or isinstance(lj_cutoff, float):
                            cutoff = lj_cutoff
                    new_r0 = new_sigma * (2 ** (1 / 6))
                    if lj_cutoff is not None:
                        L.pair_coeff(
                            f"{i+1} {j+1} lj/cut {new_eps} {new_sigma} {cutoff}"
                        )
                    else:
                        L.pair_coeff(f"{i+1} {j+1} lj/cut {new_eps} {new_sigma}")

                    if abs(i - j) <= 1:
                        print("new bond: ", i, j, new_k, new_r0)
                        L.bond_coeff(f"{bonds_counter} {new_k}")
                        bonds.append((i + 1, j + 1, bonds_counter, new_r0))
                        bonds_counter += 1
                        # L.bond_coeff(f"{bonds_counter} {new_k} {new_r0/np.sqrt(2)}")
                        # bonds.append((i + 1, j + 1, bonds_counter, new_r0))
                        # bonds_counter += 1

                elif j >= i and abs(i - j) > 1:
                    # EXCLUDING LONG-DISTANCE INTERACTIONS
                    L.pair_coeff(f"{i+1} {j+1} lj/cut 0 0 0")

        for i, j, bc, r0 in bonds:
            if i == j:
                # new_bond = (
                #     f"many upper_layer_{i} upper_layer_{j} {bc} {0.9*r0} {1.1*r0}"
                # )
                new_bond = f"many {i} {j} {bc} {0.9*r0} {1.3*r0}"
            else:
                new_bond = f"many {i} {j} {bc} {0.9*r0} {1.3*r0}"

            # print(f"new bond: {new_bond}")
            L.create_bonds(new_bond)
            # new_bond = f"many {i} {j} {bc} {0.97*r0/np.sqrt(2)} {1.03*r0/np.sqrt(2)}"
            # L.create_bonds(new_bond)
        L.neigh_modify("once yes")
        L.command("special_bonds lj 1.0 1.0 1.0")

        # Adding LJ-walls
        if add_LJ_walls:
            L.command("variable zlo equal 'zlo'")
            L.command("variable zhi equal 'zhi'")

            L.fix(
                f"wall_low 1 wall/lj126 zlo v_zlo {eps[0]} {sigma[0]} {sigma[0]*1.5} pbc yes"
            )
            L.fix(
                f"wall_high {N} wall/lj126 zhi v_zhi {eps[-1]} {sigma[-1]} {sigma[-1]*1.5} pbc yes"
            )
        return L
