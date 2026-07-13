#!/usr/bin/env python3
"""
Python equivalent of mesfet.in for PADRE simulation.

This script generates the same PADRE input deck as mesfet.in
MESFET structure

NOTE: this is a legacy deck-translation example, not a device
recommendation. It reproduces an old hand-written deck verbatim: the
61x51 mesh (3111 nodes) exceeds the ~2500-node limit of PADRE on
nanoHUB, and the junction-isolated p-substrate differs from the
semi-insulating substrate of the reference nanoHUB MESFET tool.
For a physically representative MESFET use nanohubpadre.create_mesfet().
"""

from nanohubpadre import (
    Simulation, Mesh, Region, Electrode, Doping, Contact,
    Models, System, Solve, Log, Plot3D
)


def create_mesfet_simulation():
    """Create the MESFET simulation equivalent to mesfet.in"""

    sim = Simulation()

    # Mesh specification
    sim.mesh = Mesh(nx=61, ny=51)
    sim.mesh.add_x_mesh(1, 0, ratio=1.1)
    sim.mesh.add_x_mesh(11, 0.1, ratio=0.8)
    sim.mesh.add_x_mesh(21, 0.2, ratio=0.8)
    sim.mesh.add_x_mesh(41, 0.4, ratio=0.8)
    sim.mesh.add_x_mesh(51, 0.5, ratio=0.8)
    sim.mesh.add_x_mesh(61, 0.6, ratio=1.1)
    sim.mesh.add_y_mesh(1, 0.0, ratio=1.1)
    sim.mesh.add_y_mesh(20, 0.8, ratio=0.9)
    sim.mesh.add_y_mesh(51, 1, ratio=0.8)

    # Regions
    sim.add_region(Region(1, ix_low=1, ix_high=61, iy_low=1, iy_high=20, silicon=True))
    sim.add_region(Region(2, ix_low=1, ix_high=11, iy_low=20, iy_high=51, silicon=True))
    sim.add_region(Region(3, ix_low=11, ix_high=51, iy_low=20, iy_high=51, silicon=True))
    sim.add_region(Region(4, ix_low=51, ix_high=61, iy_low=20, iy_high=51, silicon=True))

    # Electrodes
    sim.add_electrode(Electrode(2, ix_low=51, ix_high=61, iy_low=51, iy_high=51))  # Drain
    sim.add_electrode(Electrode(1, ix_low=1, ix_high=11, iy_low=51, iy_high=51))   # Source
    sim.add_electrode(Electrode(3, ix_low=21, ix_high=41, iy_low=51, iy_high=51))  # Gate

    # Doping
    sim.add_doping(Doping(region=1, p_type=True, uniform=True, concentration=1e17))
    sim.add_doping(Doping(region=2, n_type=True, uniform=True, concentration=1e20))
    sim.add_doping(Doping(region=3, n_type=True, uniform=True, concentration=1e17))
    sim.add_doping(Doping(region=4, n_type=True, uniform=True, concentration=1e20))

    # Plot doping
    sim.add_command(Plot3D(doping=True, outfile="doping"))

    # Contacts
    sim.add_contact(Contact(all_contacts=True, neutral=True))
    sim.add_contact(Contact(number=3, workfunction=4.87))

    # Models and system
    sim.models = Models(temperature=300, bgn=True, conmob=True, fldmob=True)
    sim.system = System(newton=True, carriers=1, electrons=True)

    # Solve initial
    sim.add_solve(Solve(initial=True, outfile="initsol"))
    sim.add_command(Plot3D(potential=True, electrons=True, outfile="equ"))

    # Gate bias sweep
    sim.add_solve(Solve(v3=0, vstep=-0.1, nsteps=4, electrode=3))
    sim.add_command(Plot3D(potential=True, electrons=True, outfile="non_equ1"))

    # Drain sweep
    sim.add_log(Log(ivfile="idvd"))
    sim.add_solve(Solve(v2=0, vstep=0.1, nsteps=20, electrode=2))
    sim.add_command(Plot3D(potential=True, electrons=True, outfile="non_equ2"))

    return sim


if __name__ == "__main__":
    sim = create_mesfet_simulation()
    deck = sim.generate_deck()
    print(deck)
