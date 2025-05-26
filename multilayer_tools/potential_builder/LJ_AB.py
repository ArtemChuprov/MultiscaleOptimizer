import numpy as np
from .to_file import write_ponetial_file


def create_LJ_AB(
    epsilon,
    sigma,
    A: float = 1.0,
    B: float = 1.0,
    name: str = "modified_lj",
    r_min=0.3,
    r_max=4.0,
    points_number: int = 100,
):
    # Define the modified Lennard-Jones potential
    pot_func = lambda r: 4 * epsilon * (A * (sigma / r) ** 12 - B * (sigma / r) ** 6)
    force_func = (
        lambda r: 4
        * epsilon
        * (12 * A * (sigma**12) / r**13 - 6 * B * (sigma**6) / r**7)
    )
    # Generate tabulated values
    r_values = np.linspace(r_min, r_max, points_number)  # Distance values (avoid r=0)
    potential_values = pot_func(r_values)
    force_values = force_func(r_values)
    write_ponetial_file(
        r_values=r_values,
        potential_values=potential_values,
        force_values=force_values,
        name=name,
        points_number=points_number,
    )
