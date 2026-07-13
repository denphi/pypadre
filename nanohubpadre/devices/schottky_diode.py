"""
Schottky Diode factory function.
"""

from typing import Optional, Tuple
from ..simulation import Simulation
from ..mesh import Mesh
from ..region import Region
from ..electrode import Electrode
from ..doping import Doping
from ..contact import Contact
from ..models import Models
from ..solver import System, Solve
from ..log import Log
from ._common import check_mesh_size


def create_schottky_diode(
    # Geometry parameters
    length: float = 2.0,
    width: float = 1.0,
    device_z_width: float = 1.0,
    # Mesh parameters
    nx: int = 100,
    ny: int = 20,
    # Doping parameters
    doping: float = 1e16,
    doping_type: str = "n",
    # Contact parameters
    workfunction: float = 4.8,
    barrier_lowering: bool = False,
    surf_rec: bool = True,
    vsurfn: float = 2.2e6,
    vsurfp: float = 1.9e6,
    # Physical models
    temperature: float = 300,
    srh: bool = False,
    conmob: bool = True,
    fldmob: bool = True,
    # Simulation options
    title: Optional[str] = None,
    # Output logging options
    log_iv: bool = False,
    iv_file: str = "iv",
    log_bands_eq: bool = False,
    log_bands_bias: bool = False,
    # Voltage sweep options
    forward_sweep: Optional[Tuple[float, float, float]] = None,
    reverse_sweep: Optional[Tuple[float, float, float]] = None,
) -> Simulation:
    """
    Create a Schottky diode simulation.

    Creates a metal-semiconductor junction with configurable barrier height.

    Parameters
    ----------
    length : float
        Device length in microns (default: 2.0)
    width : float
        Device width (y extent) in microns (default: 1.0)
    device_z_width : float
        Depth of the device in the third dimension in microns (default:
        1.0). Terminal currents scale linearly with this value.
    nx : int
        Mesh points in x direction (default: 100)
    ny : int
        Mesh points in y direction (default: 20)
    doping : float
        Semiconductor doping concentration in cm^-3 (default: 1e16)
    doping_type : str
        Semiconductor doping type: "n" or "p" (default: "n")
    workfunction : float
        Metal workfunction in V (default: 4.8)
    barrier_lowering : bool
        Enable image-force barrier lowering (default: False)
    surf_rec : bool
        Enable the thermionic-emission boundary condition (finite surface
        recombination velocity) at the Schottky contact (default: True).
        Without it the contact is a plain workfunction Dirichlet condition
        and the forward I-V magnitude does not follow thermionic-emission
        theory.
    vsurfn : float
        Electron thermionic surface recombination velocity in cm/s
        (default: 2.2e6, silicon A*=112 A/cm²K²). Only used when
        surf_rec=True.
    vsurfp : float
        Hole thermionic surface recombination velocity in cm/s
        (default: 1.9e6). Only used when surf_rec=True.
    temperature : float
        Simulation temperature in Kelvin (default: 300)
    srh : bool
        Enable Shockley-Read-Hall recombination (default: False)
    conmob : bool
        Enable concentration-dependent mobility (default: True)
    fldmob : bool
        Enable field-dependent mobility (default: True)
    title : str, optional
        Simulation title
    log_iv : bool
        If True, add I-V logging (default: False)
    iv_file : str
        Filename for I-V log (default: "iv")
    log_bands_eq : bool
        If True, log band diagrams at equilibrium (default: False)
    log_bands_bias : bool
        If True, log band diagrams during voltage sweeps (default: False)
    forward_sweep : tuple (v_start, v_end, v_step), optional
        If provided, adds a forward bias voltage sweep.
        Example: (0.0, 0.5, 0.05) sweeps from 0V to 0.5V
    reverse_sweep : tuple (v_start, v_end, v_step), optional
        If provided, adds a reverse bias voltage sweep.
        Example: (0.0, -5.0, -0.5) sweeps from 0V to -5V

    Returns
    -------
    Simulation
        Configured Schottky diode simulation

    Example
    -------
    >>> # Basic simulation - add your own solve commands
    >>> sim = create_schottky_diode(doping=1e17, workfunction=4.7)
    >>> sim.add_solve(Solve(initial=True))
    >>> print(sim.generate_deck())
    >>>
    >>> # Complete simulation with sweeps and logging
    >>> sim = create_schottky_diode(
    ...     log_iv=True,
    ...     log_bands_eq=True,
    ...     forward_sweep=(0.0, 0.5, 0.05)
    ... )
    >>> result = sim.run()
    """
    is_n_type = doping_type.lower() == "n"
    sim = Simulation(title=title or "Schottky Diode")
    sim._device_type = "schottky_diode"
    sim._device_kwargs = dict(
        length=length, width=width, device_z_width=device_z_width,
        nx=nx, ny=ny, doping=doping,
        doping_type=doping_type, workfunction=workfunction,
        barrier_lowering=barrier_lowering, surf_rec=surf_rec,
        vsurfn=vsurfn, vsurfp=vsurfp,
        temperature=temperature, srh=srh, conmob=conmob, fldmob=fldmob,
        title=title, log_iv=log_iv, iv_file=iv_file,
        log_bands_eq=log_bands_eq, log_bands_bias=log_bands_bias,
        forward_sweep=forward_sweep, reverse_sweep=reverse_sweep,
    )

    check_mesh_size(nx, ny, "create_schottky_diode")

    # Mesh with refinement near the Schottky contact (y=0): spacing must
    # be finest AT the contact, where the barrier and depletion region
    # are, and expand away from it.
    sim.mesh = Mesh(nx=nx, ny=ny, outfile="mesh",
                    width=None if device_z_width == 1.0 else device_z_width)
    sim.mesh.add_x_mesh(1, 0)
    sim.mesh.add_x_mesh(nx, length)
    sim.mesh.add_y_mesh(1, 0)

    # Ensure intermediate point is valid (between 1 and ny)
    y_idx = int(ny * 0.3)
    if y_idx < 2:
        y_idx = max(2, ny - 1)

    # Only add if we have space (ny >= 3)
    if ny >= 3 and y_idx < ny:
        sim.mesh.add_y_mesh(y_idx, width * 0.1, ratio=1.25)  # expand away from contact

    sim.mesh.add_y_mesh(ny, width, ratio=1.2)

    # Region
    sim.add_region(Region(1, ix_low=1, ix_high=nx, iy_low=1, iy_high=ny, silicon=True))

    # Electrodes
    sim.add_electrode(Electrode(1, ix_low=1, ix_high=nx, iy_low=1, iy_high=1))  # Schottky contact
    sim.add_electrode(Electrode(2, ix_low=1, ix_high=nx, iy_low=ny, iy_high=ny))  # Ohmic back

    # Doping
    sim.add_doping(Doping(region=1, n_type=is_n_type, p_type=not is_n_type,
                          uniform=True, concentration=doping))

    # Contacts — the Schottky contact uses the thermionic-emission
    # boundary condition (finite surface recombination velocities)
    if surf_rec:
        sim.add_contact(Contact(number=1, workfunction=workfunction,
                                surf_rec=True, vsurfn=vsurfn, vsurfp=vsurfp,
                                barrierl=barrier_lowering))
    else:
        sim.add_contact(Contact(number=1, workfunction=workfunction,
                                barrierl=barrier_lowering))
    sim.add_contact(Contact(number=2, neutral=True))

    # Models
    sim.models = Models(temperature=temperature, srh=srh, conmob=conmob, fldmob=fldmob)
    sim.system = System(electrons=True, holes=True, newton=True)

    # I-V logging — must be declared before any solves
    if log_iv:
        sim.add_log(Log(ivfile=iv_file))

    # Add solve commands if sweeps are specified
    if forward_sweep is not None or reverse_sweep is not None or log_bands_eq:
        sim.add_solve(Solve(initial=True, outfile="eq"))

        if log_bands_eq:
            sim.log_band_diagram(
                outfile_prefix="eq",
                x_start=0.0, y_start=0.0,
                x_end=0.0, y_end=width
            )

        if forward_sweep is not None:
            v_start, v_end, v_step = forward_sweep
            nsteps = int(abs(v_end - v_start) / abs(v_step))
            # Store forward point count so callers can split the combined IV file
            sim._fwd_nsteps = nsteps + 1
            # Only 1 prior solution (initial): use previous=True for first sweep
            sim.add_solve(Solve(
                previous=True, v1=v_start, vstep=v_step,
                nsteps=nsteps, electrode=1, outfile="fwd"
            ))
            if log_bands_bias:
                sim.log_band_diagram(
                    outfile_prefix="fwd",
                    x_start=0.0, y_start=0.0,
                    x_end=0.0, y_end=width,
                    include_qf=True
                )

        if reverse_sweep is not None:
            v_start, v_end, v_step = reverse_sweep
            nsteps = int(abs(v_end - v_start) / abs(v_step))
            # If forward sweep ran first, 2+ solutions exist → project=True
            # Otherwise only 1 prior solution → previous=True
            use_project = forward_sweep is not None
            sim.add_solve(Solve(
                project=use_project, previous=not use_project,
                v1=v_start, vstep=v_step,
                nsteps=nsteps, electrode=1, outfile="rev"
            ))
            if log_bands_bias:
                sim.log_band_diagram(
                    outfile_prefix="rev",
                    x_start=0.0, y_start=0.0,
                    x_end=0.0, y_end=width,
                    include_qf=True
                )

    return sim


# Alias
schottky_diode = create_schottky_diode
