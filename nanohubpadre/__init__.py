"""
nanohub-padre - Python library for PADRE semiconductor device simulator

This library provides a Pythonic interface to create and run PADRE simulations
for semiconductor device modeling.

Import as: import nanohubpadre
"""

from .simulation import Simulation
from .mesh import Mesh, XMesh, YMesh, ZMesh
from .region import Region
from .electrode import Electrode
from .doping import Doping
from .contact import Contact
from .material import Material, Alloy
from .models import Models
from .solver import Solve, Method, System, LinAlg
from .log import Log
from .interface import Interface
from .regrid import Regrid, Adapt
from .plotting import Plot1D, Plot2D, Contour, Vector
from .options import Options, Load
from .plot3d import Plot3D, Plot3DData, parse_plot3d_file
from .base import Comment, Title
from .parser import (
    parse_padre_output,
    PadreOutputParser,
    SimulationResult,
    BiasPoint,
    MeshStatistics,
    ConvergenceStatus,
    parse_iv_file,
    parse_iv_content,
    IVFileParser,
    IVData,
)

# Device factory functions
from .devices import (
    create_pn_diode,
    create_mos_capacitor,
    create_mosfet,
    create_mesfet,
    create_bjt,
    create_schottky_diode,
    create_solar_cell,
    # Aliases
    pn_diode,
    mos_capacitor,
    mosfet,
    mesfet,
    bjt,
    schottky_diode,
    solar_cell,
    # Description and schematics
    describe,
    list_devices,
    device_schematic,
    DeviceSchematic,
)

# Environment loading (use command)
from .use import use, load_padre, list_available_modules

# Analytic first-order estimates (sweep-range / geometry sanity checks)
from .estimates import estimate_mosfet_vt, max_depletion_width_um

# Visualization functions
from .visualization import (
    plot_iv,
    plot_transfer_characteristic,
    plot_output_characteristic,
    plot_diode_iv,
    plot_2d_map,
    plot_contour,
    get_available_backends,
)

# Solution file parser and visualization
from .solution import (
    parse_solution_file,
    load_solution_series,
    SolutionData,
    MeshData,
)

# Output management
from .outputs import (
    OutputManager,
    OutputType,
    PlotData,
    OutputEntry,
)

from ._version import __version__  # noqa: F401
__all__ = [
    "Simulation",
    "Mesh", "XMesh", "YMesh", "ZMesh",
    "Region",
    "Electrode",
    "Doping",
    "Contact",
    "Material", "Alloy",
    "Models",
    "Solve", "Method", "System", "LinAlg",
    "Log",
    "Interface",
    "Regrid", "Adapt",
    "Plot1D", "Plot2D", "Contour", "Vector",
    "Options", "Load",
    "Plot3D",
    "Plot3DData",
    "parse_plot3d_file",
    "Comment", "Title",
    # Parser
    "parse_padre_output",
    "PadreOutputParser",
    "SimulationResult",
    "BiasPoint",
    "MeshStatistics",
    "ConvergenceStatus",
    # I-V file parser
    "parse_iv_file",
    "parse_iv_content",
    "IVFileParser",
    "IVData",
    # Device factory functions
    "create_pn_diode",
    "create_mos_capacitor",
    "create_mosfet",
    "create_mesfet",
    "create_bjt",
    "create_schottky_diode",
    "create_solar_cell",
    # Aliases
    "pn_diode",
    "mos_capacitor",
    "mosfet",
    "mesfet",
    "bjt",
    "schottky_diode",
    "solar_cell",
    # Description and schematics
    "describe",
    "list_devices",
    "device_schematic",
    "DeviceSchematic",
    # Environment loading
    "use",
    "load_padre",
    "list_available_modules",
    # Analytic estimates
    "estimate_mosfet_vt",
    "max_depletion_width_um",
    # Visualization
    "plot_iv",
    "plot_transfer_characteristic",
    "plot_output_characteristic",
    "plot_diode_iv",
    "plot_2d_map",
    "plot_contour",
    "get_available_backends",
    # Solution file parser
    "parse_solution_file",
    "load_solution_series",
    "SolutionData",
    "MeshData",
    # Output management
    "OutputManager",
    "OutputType",
    "PlotData",
    "OutputEntry",
]
