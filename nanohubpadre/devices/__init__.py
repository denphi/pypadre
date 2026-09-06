"""
Device factory functions for common semiconductor structures.

Provides convenient functions to create pre-configured simulations
for common device types like diodes, MOSFETs, MESFETs, and bipolar transistors.
"""

from .pn_diode import create_pn_diode, pn_diode
from .nin_diode import create_nin_diode, nin_diode, pip_diode
from .mos_capacitor import create_mos_capacitor, mos_capacitor
from .mosfet import create_mosfet, mosfet
from .mesfet import create_mesfet, mesfet
from .bjt import create_bjt, bjt
from .schottky_diode import create_schottky_diode, schottky_diode
from .solar_cell import create_solar_cell, solar_cell
from .describe import describe, list_devices
from .schematics import device_schematic, DeviceSchematic

__all__ = [
    # Factory functions
    "create_pn_diode",
    "create_nin_diode",
    "create_mos_capacitor",
    "create_mosfet",
    "create_mesfet",
    "create_bjt",
    "create_schottky_diode",
    "create_solar_cell",
    # Aliases
    "pn_diode",
    "nin_diode",
    "pip_diode",
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
]
