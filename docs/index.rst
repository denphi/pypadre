.. nanohub-padre documentation master file

Welcome to nanohub-padre's documentation!
==========================================

**nanohub-padre** is a Python library for creating and running PADRE semiconductor
device simulations on nanoHUB or locally. It provides a Pythonic interface to
generate PADRE input decks and visualize simulation results — no manual file
parsing required.

PADRE (Physics-based Accurate Device Resolution and Evaluation) is a 2D device
simulator that solves the drift-diffusion equations for semiconductor devices.

Features
--------

* **One-line device creation** — factory functions for PN diode, MOS capacitor,
  MOSFET, MESFET, BJT, Schottky diode, and solar cell
* **Built-in visualization** — ``plot_band_diagram()``, ``plot_iv()``,
  ``plot_cv()``, ``plot_electrostatics()``, ``plot_carriers()``,
  ``plot_transfer()``, ``plot_output()``, ``plot_gummel()``, ``plot_contour()``
* **Automatic output management** — outputs are registered and parsed
  automatically after ``sim.run()``; no manual file handling
* **Rappture-compatible decks** — generated input decks match the nanoHUB
  Rappture tool reference decks (keyword order, interface cards, permi=, etc.)
* **Jupyter notebooks** — six ready-to-run example notebooks covering all
  major device types

Quick Start
-----------

**MOS Capacitor C-V Analysis**

.. code-block:: python

   from nanohubpadre import create_mos_capacitor

   sim = create_mos_capacitor(
       oxide_thickness=0.005,       # 5 nm SiO2
       substrate_doping=1e17,       # silicon_thickness defaults to 5 um
       substrate_type="p",
       gate_type="n_poly",
       log_cv=True,
       log_bands_eq=True,
       vg_sweep=(-2.0, 2.0, 0.05),
       ac_frequency=1e6,
   )

   result = sim.run()
   if result.returncode != 0:
       raise RuntimeError(f"Simulation failed:\n{result.stderr}")

   sim.plot_band_diagram(suffix="eq", title="Band Diagram at Equilibrium")
   sim.plot_cv(title="MOS Capacitor C-V")

**MOSFET Transfer Characteristic**

.. code-block:: python

   from nanohubpadre import create_mosfet

   sim = create_mosfet(channel_length=0.025, device_type="nmos")

   result = sim.run()
   if result.returncode != 0:
       raise RuntimeError(f"Simulation failed:\n{result.stderr}")

   sim.plot_transfer(gate_electrode=3, drain_electrode=2)

**Building from Scratch**

.. code-block:: python

   from nanohubpadre import (
       Simulation, Mesh, Region, Electrode, Doping,
       Contact, Material, Interface, Models, System, Solve, Log
   )

   sim = Simulation(title="PN Diode")

   sim.mesh = Mesh(nx=100, ny=3)
   sim.mesh.add_x_mesh(1, 0.0)
   sim.mesh.add_x_mesh(50, 0.5, ratio=0.8)
   sim.mesh.add_x_mesh(100, 1.0)
   sim.mesh.add_y_mesh(1, 0.0)
   sim.mesh.add_y_mesh(3, 1.0)

   sim.add_region(Region(1, ix_low=1, ix_high=100, iy_low=1, iy_high=3,
                         material="silicon", semiconductor=True))
   sim.add_electrode(Electrode(1, ix_low=1,   ix_high=1,   iy_low=1, iy_high=3))
   sim.add_electrode(Electrode(2, ix_low=100, ix_high=100, iy_low=1, iy_high=3))

   sim.add_doping(Doping(p_type=True, concentration=1e17, uniform=True, x_right=0.5))
   sim.add_doping(Doping(n_type=True, concentration=1e17, uniform=True, x_left=0.5))

   sim.add_contact(Contact(all_contacts=True, neutral=True))
   sim.add_material(Material(name="silicon", taun0=1e-6, taup0=1e-6))
   sim.models = Models(temperature=300, srh=True, conmob=True, fldmob=True,
                       print_models=True)
   sim.system = System(electrons=True, holes=True, newton=True)

   sim.add_solve(Solve(initial=True))
   sim.add_log(Log(ivfile="iv"))
   sim.add_solve(Solve(project=True, v1=0.0, vstep=0.05, nsteps=20, electrode=1))

   result = sim.run()
   if result.returncode != 0:
       raise RuntimeError(f"Simulation failed:\n{result.stderr}")

   sim.plot_iv(title="PN Diode Forward I-V")

Installation
------------

.. code-block:: bash

   pip install nanohub-padre

Or from source:

.. code-block:: bash

   git clone https://github.com/nanohub/nanohub-padre.git
   cd nanohub-padre
   pip install -e .

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   user_guide
   devices
   examples
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
