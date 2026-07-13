"""
Solar Cell factory function.

.. warning::
   This device is **experimental**.  PADRE is a transistor solver optimised
   for µA–mA current ranges.  For a typical silicon solar cell (200 µm base,
   ~10¹⁶ cm⁻³ doping) the dark saturation current I₀ is ~10⁻²⁰ A for a
   1 µm × 1 µm cross-section — well below PADRE's floating-point resolution.
   Terminal currents from the IV log are therefore numerical noise.
   Band diagrams, carrier profiles, and electrostatic potential are still
   computed correctly and can be used for qualitative analysis.
"""

import warnings
from typing import Optional, Tuple
from ..simulation import Simulation
from ..mesh import Mesh
from ..region import Region
from ..electrode import Electrode
from ..doping import Doping
from ..contact import Contact
from ..material import Material
from ..models import Models
from ..solver import System, Solve
from ..log import Log
from ._common import check_mesh_size


def create_solar_cell(
    # Geometry parameters
    emitter_depth: float = 0.5,
    base_thickness: float = 200.0,
    device_width: float = 1.0,
    device_z_width: float = 1.0,
    # Mesh parameters
    nx: int = 3,
    ny: int = 100,
    # Doping parameters
    emitter_doping: float = 1e19,
    base_doping: float = 1e16,
    device_type: str = "n_on_p",
    # Physical models
    temperature: float = 300,
    srh: bool = True,
    auger: bool = True,
    conmob: bool = True,
    fldmob: bool = True,
    # Material parameters
    taun0: float = 1e-5,
    taup0: float = 1e-5,
    # Surface recombination
    front_surface_velocity: float = 1e4,
    back_surface_velocity: float = 1e7,
    # Simulation options
    title: Optional[str] = None,
    # Output logging options
    log_iv: bool = False,
    iv_file: str = "iv_dark",
    log_bands_eq: bool = False,
    # Voltage sweep options
    forward_sweep: Optional[Tuple[float, float, float]] = None,
    sweep_electrode: int = 1,
) -> Simulation:
    """
    Create a solar cell simulation.

    Creates a PN junction solar cell structure with configurable surface
    recombination velocities for studying photovoltaic performance.

    Parameters
    ----------
    emitter_depth : float
        Emitter junction depth in microns (default: 0.5)
    base_thickness : float
        Base (substrate) thickness in microns (default: 200.0)
    device_width : float
        Device width in microns (default: 1.0)
    device_z_width : float
        Depth of the device in the third dimension in microns (default:
        1.0).  Terminal currents scale linearly with this value; increase
        it (e.g. 1e4 for a 1 cm-deep strip) to lift solar-cell dark
        currents above PADRE's floating-point noise floor.
    nx : int
        Mesh points in x direction (default: 3)
    ny : int
        Mesh points in y direction (default: 100)
    emitter_doping : float
        Emitter doping concentration in cm^-3 (default: 1e19)
    base_doping : float
        Base doping concentration in cm^-3 (default: 1e16)
    device_type : str
        "n_on_p" or "p_on_n" (default: "n_on_p")
    temperature : float
        Simulation temperature in Kelvin (default: 300)
    srh : bool
        Enable Shockley-Read-Hall recombination (default: True)
    auger : bool
        Enable Auger recombination (default: True)
    conmob : bool
        Enable concentration-dependent mobility (default: True)
    fldmob : bool
        Enable field-dependent mobility (default: True)
    taun0 : float
        Electron lifetime in seconds (default: 1e-5)
    taup0 : float
        Hole lifetime in seconds (default: 1e-5)
    front_surface_velocity : float
        Front surface recombination velocity in cm/s (default: 1e4)
    back_surface_velocity : float
        Back surface recombination velocity in cm/s (default: 1e7)
    title : str, optional
        Simulation title
    log_iv : bool
        If True, add I-V logging (default: False)
    iv_file : str
        Filename for I-V log (default: "iv_dark")
    log_bands_eq : bool
        If True, log band diagrams at equilibrium (default: False)
    forward_sweep : tuple (v_start, v_end, v_step), optional
        Forward voltage sweep for dark I-V characteristic.
        Example: (0.0, 0.8, 0.05) sweeps from 0V to 0.8V
    sweep_electrode : int
        Electrode number for voltage sweep (default: 1, front contact)

    Returns
    -------
    Simulation
        Configured solar cell simulation

    Example
    -------
    >>> # Basic solar cell - add your own solve commands
    >>> sim = create_solar_cell(base_thickness=300, base_doping=1e15)
    >>> sim.add_solve(Solve(initial=True))
    >>> print(sim.generate_deck())
    >>>
    >>> # Dark I-V characteristic
    >>> sim = create_solar_cell(
    ...     log_iv=True,
    ...     forward_sweep=(0.0, 0.75, 0.05)
    ... )
    >>> result = sim.run()
    """
    warnings.warn(
        "create_solar_cell() is experimental. "
        "PADRE cannot resolve solar-cell dark currents (I₀ ~ 10⁻²⁰ A) with the "
        "default 1 µm × 1 µm cross-section — IV log output will be numerical noise. "
        "Increase device_z_width (e.g. 1e4 µm) to scale terminal currents up. "
        "Band diagrams and carrier profiles are still valid.",
        UserWarning,
        stacklevel=2,
    )

    is_n_on_p = device_type.lower() == "n_on_p"
    sim = Simulation(title=title or f"Solar Cell ({'N-on-P' if is_n_on_p else 'P-on-N'})")
    sim._device_type = "solar_cell"
    sim._device_kwargs = dict(
        emitter_depth=emitter_depth, base_thickness=base_thickness,
        device_width=device_width, device_z_width=device_z_width,
        nx=nx, ny=ny, emitter_doping=emitter_doping,
        base_doping=base_doping, device_type=device_type,
        temperature=temperature, srh=srh, auger=auger, conmob=conmob,
        fldmob=fldmob, taun0=taun0, taup0=taup0,
        front_surface_velocity=front_surface_velocity,
        back_surface_velocity=back_surface_velocity,
        title=title, log_iv=log_iv, iv_file=iv_file,
        log_bands_eq=log_bands_eq, forward_sweep=forward_sweep,
        sweep_electrode=sweep_electrode,
    )

    total_depth = emitter_depth + base_thickness

    check_mesh_size(nx, ny, "create_solar_cell")

    # Mesh with refinement in emitter and near junction.
    # width= sets the z-depth: terminal currents scale with it.
    ny_emitter = int(ny * 0.3)  # 30% of mesh in emitter region
    sim.mesh = Mesh(nx=nx, ny=ny, width=device_z_width)
    sim.mesh.add_x_mesh(1, 0)
    sim.mesh.add_x_mesh(nx, device_width)
    sim.mesh.add_y_mesh(1, 0)
    sim.mesh.add_y_mesh(ny_emitter, emitter_depth, ratio=1.2)
    sim.mesh.add_y_mesh(ny, total_depth, ratio=1.1)

    # Region
    sim.add_region(Region(1, ix_low=1, ix_high=nx, iy_low=1, iy_high=ny, silicon=True))

    # Electrodes
    sim.add_electrode(Electrode(1, ix_low=1, ix_high=nx, iy_low=1, iy_high=1))  # Front (emitter)
    sim.add_electrode(Electrode(2, ix_low=1, ix_high=nx, iy_low=ny, iy_high=ny))  # Back (base)

    # Doping (Gaussian emitter, uniform base).  peak=0 pins the Gaussian
    # maximum at the front surface so junction= yields the intended depth.
    if is_n_on_p:
        sim.add_doping(Doping(region=1, p_type=True, uniform=True, concentration=base_doping))
        sim.add_doping(Doping(region=1, n_type=True, gaussian=True,
                              concentration=emitter_doping,
                              junction=emitter_depth, peak=0.0))
    else:
        sim.add_doping(Doping(region=1, n_type=True, uniform=True, concentration=base_doping))
        sim.add_doping(Doping(region=1, p_type=True, gaussian=True,
                              concentration=emitter_doping,
                              junction=emitter_depth, peak=0.0))

    # Contacts with surface recombination
    sim.add_contact(Contact(number=1, neutral=True, surf_rec=True,
                            vsurfn=front_surface_velocity, vsurfp=front_surface_velocity))
    sim.add_contact(Contact(number=2, neutral=True, surf_rec=True,
                            vsurfn=back_surface_velocity, vsurfp=back_surface_velocity))

    # Material with lifetimes
    sim.add_material(Material(name="silicon", taun0=taun0, taup0=taup0))

    # Models
    sim.models = Models(temperature=temperature, srh=srh, auger=auger,
                        conmob=conmob, fldmob=fldmob)
    sim.system = System(electrons=True, holes=True, newton=True)

    # I-V logging
    if log_iv:
        sim.add_log(Log(ivfile=iv_file))

    # Only add solve commands if sweeps are specified
    if forward_sweep is not None or log_bands_eq:
        # Always start with equilibrium solve
        sim.add_solve(Solve(initial=True, outfile="eq"))

        # Log band diagram at equilibrium (vertical cut along device depth)
        if log_bands_eq:
            x_mid = device_width / 2
            sim.log_band_diagram(
                outfile_prefix="eq",
                x_start=x_mid, x_end=x_mid,
                y_start=0.0, y_end=total_depth
            )

        # Forward bias sweep for dark I-V characteristic
        # (first solve after INIT must use PREV, not PROJ)
        if forward_sweep is not None:
            v_start, v_end, v_step = forward_sweep
            nsteps = int(abs(v_end - v_start) / abs(v_step))

            # Sweep voltage
            sim.add_solve(Solve(
                previous=True,
                v1=v_start if sweep_electrode == 1 else None,
                v2=v_start if sweep_electrode == 2 else None,
                vstep=v_step,
                nsteps=nsteps,
                electrode=sweep_electrode,
                outfile="iv_fwd"
            ))

    return sim


# Alias
solar_cell = create_solar_cell
