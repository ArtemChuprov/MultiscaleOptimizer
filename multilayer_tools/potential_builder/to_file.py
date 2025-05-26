def write_ponetial_file(
    r_values, potential_values, force_values, name: str, points_number: int = 100
):
    # Save to file (for LAMMPS table)
    with open(
        f"/home/nagibator69/science/CollisionProject/notebooks/potentials/{name}.table",
        "w",
    ) as f:
        f.write("# Custom Artem potential\n")
        f.write("LJ_MOD\n")
        f.write(
            f"N {points_number} R {min(r_values)} {max(r_values)}\n"
        )  # Number of points
        f.write("# r U(r) F(r)\n")
        i = 1
        for r, U, F in zip(r_values, potential_values, force_values):
            f.write(f"{i} {r:.6f} {U:.6f} {F:.6f}\n")
            i += 1
