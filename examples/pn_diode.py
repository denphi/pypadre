#!/usr/bin/env python3
"""
Example: Simple PN Diode simulation using PyPADRE.

This example demonstrates how to set up a basic PN junction diode
simulation with PADRE.
"""

from nanohubpadre import (
    Simulation, Mesh, Region, Electrode, Doping, Contact,
    Models, System, Method, Solve, Log
)


def create_pn_diode():
    """Create a PN diode simulation."""

    # Create simulation
    sim = Simulation(title="PN Junction Diode Example")

    # ========== MESH ==========
    # Create a rectangular mesh with 40x17 nodes
    sim.mesh = Mesh(nx=40, ny=17, outfile="pn_mesh.pg")

    # X mesh: 0 to 2 microns
    sim.mesh.add_x_mesh(node=1, location=0.0)
    sim.mesh.add_x_mesh(node=40, location=2.0)

    # Y mesh: 0 to 2 microns with grading toward junction
    sim.mesh.add_y_mesh(node=1, location=0.0)
    sim.mesh.add_y_mesh(node=10, location=0.5, ratio=0.8)  # Fine near junction
    sim.mesh.add_y_mesh(node=17, location=2.0, ratio=1.3)

    # ========== REGION ==========
    # Silicon substrate
    sim.add_region(Region(
        number=1,
        material="silicon",
        semiconductor=True,
        ix_low=1, ix_high=40,
        iy_low=1, iy_high=17
    ))

    # ========== DOPING ==========
    # P-type substrate (uniform)
    sim.add_doping(Doping(
        uniform=True,
        p_type=True,
        concentration=1e16
    ))

    # N+ top region (Gaussian profile)
    sim.add_doping(Doping(
        gaussian=True,
        n_type=True,
        concentration=1e19,
        junction=0.5,  # Junction at 0.5 microns
        peak=0.0       # Peak at surface
    ))

    # ========== ELECTRODES ==========
    # Top contact (N+ region)
    sim.add_electrode(Electrode(
        number=1,
        ix_low=1, ix_high=40,
        iy_low=1, iy_high=1
    ))

    # Bottom contact (P substrate)
    sim.add_electrode(Electrode(
        number=2,
        ix_low=1, ix_high=40,
        iy_low=17, iy_high=17
    ))

    # ========== CONTACTS ==========
    # Ohmic contacts for all electrodes
    sim.add_contact(Contact(all_contacts=True, neutral=True))

    # ========== MODELS ==========
    # Drift-diffusion with SRH recombination
    sim.models = Models(
        temperature=300,
        srh=True,
        conmob=True,
        fldmob=True
    )

    # ========== SYSTEM ==========
    # Two-carrier, fully coupled Newton
    sim.system = System(carriers=2, newton=True)

    # ========== METHOD ==========
    # Standard method with trap for convergence
    sim.method = Method(
        trap=True,
        a_trap=0.5,
        itlimit=30
    )

    # ========== LOG ==========
    # Log I-V data
    sim.log = Log(ivfile="pn_iv.log")

    # ========== SOLVE ==========
    # Initial equilibrium solution
    sim.add_solve(Solve(
        initial=True,
        outfile="pn_eq"
    ))

    # Forward bias sweep: 0 to 0.8V.
    # The first bias solve after SOLVE INIT must use PREVIOUS: PROJECT
    # extrapolates from two prior solutions and only one exists here.
    sim.add_solve(Solve(
        previous=True,
        v1=0.0, v2=0.0,  # Start at 0V
        vstep=0.05,
        nsteps=16,
        electrode=2,
        no_append=True,
        outfile="pn_fwd_a"
    ))

    # Ramp back down to 0 V before reversing: jumping straight from +0.8 V
    # to -0.5 V is a 1.3 V step across a forward-biased junction and will
    # not converge.
    sim.add_solve(Solve(
        project=True,
        v2=0.8,
        vstep=-0.1,
        nsteps=8,
        electrode=2,
        no_append=True,
        outfile="pn_ramp_a"
    ))

    # Reverse bias: 0 to -5V
    sim.add_solve(Solve(
        project=True,
        v2=0.0,
        vstep=-0.5,
        nsteps=10,
        electrode=2,
        no_append=True,
        outfile="pn_rev_a"
    ))

    return sim


def main():
    # Create the simulation
    sim = create_pn_diode()

    # Generate and print the input deck
    deck = sim.generate_deck()
    print(deck)

    # Optionally write to file
    # sim.write_deck("pn_diode.inp")

    # Optionally run PADRE (requires PADRE to be installed)
    # result = sim.run(padre_executable="padre", output_file="pn_diode.out")
    # print(f"PADRE returned: {result.returncode}")


if __name__ == "__main__":
    main()
