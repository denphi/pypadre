# nanohub-padre

A Python library for creating and running PADRE semiconductor device simulations.

## Overview

nanohub-padre provides a Pythonic interface to generate PADRE input decks, making it easier to set up complex device simulations programmatically. PADRE is a 2D/3D device simulator that solves the drift-diffusion equations for semiconductor devices.

## Features

- **Pythonic Interface**: Define meshes, regions, doping profiles, and solver settings using Python objects
- **Device Factory Functions**: Pre-built functions to create common devices (PN/PIN diode, NIN/PIP isotype diode, MOSFET, BJT, solar cell, etc.)
- **Complete PADRE Support**: Covers mesh generation, material properties, physical models, and solve commands
- **Validation**: Built-in parameter validation and helpful error messages
- **Examples**: Ready-to-run examples for common device structures

## Installation

```bash
pip install nanohub-padre
```

Or install from source:

```bash
git clone https://github.com/nanohub/nanohub-padre.git
cd nanohub-padre
pip install -e .
```

## Quick Start

### Using Device Factory Functions (Recommended)

```python
from nanohubpadre import create_mosfet, Solve, Log

# Create an NMOS transistor with one line
sim = create_mosfet(
    channel_length=0.05,
    device_type="nmos",
    temperature=300
)

# Add solve commands
sim.add_solve(Solve(initial=True))
sim.add_log(Log(ivfile="idvg"))
sim.add_solve(Solve(v3=0, vstep=0.1, nsteps=15, electrode=3))

# Generate the input deck
print(sim.generate_deck())
```

### Building from Scratch

```python
from nanohubpadre import (
    Simulation, Mesh, Region, Electrode, Doping,
    Contact, Models, System, Solve
)

# Create simulation
sim = Simulation(title="Simple PN Diode")

# Define mesh
sim.mesh = Mesh(nx=100, ny=3)
sim.mesh.add_x_mesh(1, 0)
sim.mesh.add_x_mesh(100, 1.0)
sim.mesh.add_y_mesh(1, 0)
sim.mesh.add_y_mesh(3, 1)

# Define silicon region
sim.add_region(Region(1, ix_low=1, ix_high=100, iy_low=1, iy_high=3, silicon=True))

# Define electrodes
sim.add_electrode(Electrode(1, ix_low=1, ix_high=1, iy_low=1, iy_high=3))
sim.add_electrode(Electrode(2, ix_low=100, ix_high=100, iy_low=1, iy_high=3))

# Define doping
sim.add_doping(Doping(p_type=True, concentration=1e17, uniform=True, x_right=0.5))
sim.add_doping(Doping(n_type=True, concentration=1e17, uniform=True, x_left=0.5))

# Set contacts
sim.add_contact(Contact(all_contacts=True, neutral=True))

# Configure models
sim.models = Models(temperature=300, srh=True, conmob=True, fldmob=True)
sim.system = System(electrons=True, holes=True, newton=True)

# Solve
sim.add_solve(Solve(initial=True))

# Generate and print the input deck
print(sim.generate_deck())
```

## Device Factory Functions

The library includes factory functions for common devices:

| Function | Description |
|----------|-------------|
| `create_pn_diode` | PN junction diode (PIN with `intrinsic_width`) |
| `create_nin_diode` | NIN/PIP isotype diode — unipolar, non-rectifying |
| `create_mos_capacitor` | MOS capacitor for C-V analysis |
| `create_mosfet` | NMOS/PMOS transistor |
| `create_mesfet` | Metal-semiconductor FET |
| `create_bjt` | NPN/PNP bipolar transistor |
| `create_schottky_diode` | Schottky barrier diode |
| `create_solar_cell` | PN junction solar cell |

## Examples

The `examples/` directory contains Python equivalents of common PADRE simulations:

- **pndiode.py**: PN junction diode I-V characterization
- **moscap.py**: MOS capacitor C-V analysis
- **mosfet_equivalent.py**: NMOS transistor transfer and output characteristics
- **mesfet.py**: Metal-Semiconductor FET simulation
- **single_mosgap.py**: Simple oxide-silicon structure

Device factory examples in `examples/devices/`:

- **pn_diode_example.py**: PN diode using factory function
- **mosfet_example.py**: NMOS using factory function
- **bjt_example.py**: NPN BJT using factory function
- **solar_cell_example.py**: Solar cell using factory function

Run an example:

```bash
PYTHONPATH=. python3 examples/pndiode.py > pndiode.inp
padre < pndiode.inp > pndiode.out
```

## Supported Commands

nanohub-padre supports all major PADRE commands:

| Category | Commands |
|----------|----------|
| Mesh | MESH, X.MESH, Y.MESH, Z.MESH |
| Structure | REGION, ELECTRODE |
| Doping | DOPING (uniform, Gaussian, ERFC, file) |
| Boundaries | CONTACT, INTERFACE, SURFACE |
| Materials | MATERIAL, ALLOY |
| Models | MODELS |
| Solver | SYSTEM, METHOD, LINALG, SOLVE |
| Output | LOG, PLOT.1D, PLOT.2D, PLOT.3D, CONTOUR, VECTOR |
| Control | OPTIONS, LOAD, REGRID, ADAPT |

## Documentation

Full documentation is available at [https://nanohub-padre.readthedocs.io/](https://nanohub-padre.readthedocs.io/)

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
