from stress_computer.stress_simulators import BasicStressSimulator
from stress_computer.basic_simulators import LJBasicSimulator
from multilayer_tools.create_scenes.multilayer import create_multilayer
import numpy as np

eps = 0.9
sigma = 2.5
atom_types = 3
fcc_period = 3.615
masses = [63.5 * 2 ** (3 * i) for i in range(atom_types)]

layers_kwargs = {
    "width": (8, 8, 8),
    "types_number": atom_types,
    "fcc_period": [fcc_period * 2**i for i in range(atom_types)],
    "masses": masses,
    "file_name": "./Multilayer",
}


def calculate_young(EPS, dir):
    potential = {
        "LJ": {
            "eps": np.array([eps * 2 ** (3 * i) for i in range(atom_types)]),
            "sigma": np.array([sigma * 2**i for i in range(atom_types)]),
            "lj_cutoff": np.array([(2.5 * sigma) * 2**i for i in range(atom_types)]),
            "walls": True,
        },
        "harmonic": {
            "k": np.array(
                [0 * 4**i for i in range(atom_types)]
                + [0 * 4**i for i in range(atom_types - 1)]
            ),
            "cross_springs": True,
        },
    }

    # # create_box(**box1_params)
    simulator = LJBasicSimulator(
        potential_params=potential,
        scene_creator=create_multilayer,
        scene_params=layers_kwargs,
        path_dump="./coords",
        minimize=10000,
    )

    L = simulator.basic_simulation()

    stress_computer = BasicStressSimulator(
        types_number=atom_types,
        dump_flg=False,
    )

    E = stress_computer.compute_young_uniaxial(L, direction=dir, delta=0.03)
    return E


