"""
PN Junction Diode factory function.
"""

from typing import List, Optional, Tuple
from ..simulation import Simulation
from ..mesh import Mesh
from ..region import Region
from ..electrode import Electrode
from ..doping import Doping
from ..contact import Contact
from ..material import Material
from ..models import Models
from ..solver import System, Solve
from ..options import Options
from ..log import Log
from ._common import check_mesh_size, solve_guess, add_bias_ramp, GATE_STEP


def create_pn_diode(
    # Geometry parameters
    length: float = 1.0,
    width: float = 1.0,
    device_z_width: float = 1.0,
    junction_position: float = 0.5,
    intrinsic_width: float = 0.0,
    # Mesh parameters
    nx: int = 200,
    ny: int = 3,
    # Doping parameters
    p_doping: float = 1e17,
    n_doping: float = 1e17,
    intrinsic_doping: float = 1.5e10,
    # Physical models
    temperature: float = 300,
    srh: bool = True,
    conmob: bool = True,
    fldmob: bool = True,
    impact: bool = False,
    # Material parameters
    taun0: float = 1e-6,
    taup0: float = 1e-6,
    # Simulation options
    title: Optional[str] = None,
    postscript: bool = False,
    # Output logging options
    log_iv: bool = False,
    iv_file: str = "iv",
    log_bands_eq: bool = False,
    log_bands_bias: bool = False,
    # Voltage sweep options
    forward_sweep: Optional[Tuple[float, float, float]] = None,
    reverse_sweep: Optional[Tuple[float, float, float]] = None,
    sweep_electrode: int = 2,
    # Physics-profile logging
    log_physics_at: Optional[List[float]] = None,
) -> Simulation:
    """
    Create a PN junction diode simulation.

    Creates a 1D-like PN diode structure with configurable doping profiles,
    mesh refinement, and physical models.

    Parameters
    ----------
    length : float
        Total device length in microns (default: 1.0)
    width : float
        Device width (y extent) in microns (default: 1.0)
    device_z_width : float
        Depth of the device in the third dimension in microns (default:
        1.0). Terminal currents scale linearly with this value. Note:
        earlier versions passed `width` here, silently scaling currents
        with the y extent squared.
    junction_position : float
        Position of the P-I (or P-N) junction as fraction of length (default: 0.5)
    intrinsic_width : float
        Width of the intrinsic (lightly-doped) region in microns (default: 0.0).
        When 0, structure is a standard PN diode.  When > 0, an intrinsic
        region is inserted between P and N, forming a PIN diode.
    nx : int
        Number of mesh points in x direction (default: 200)
    ny : int
        Number of mesh points in y direction (default: 3)
    p_doping : float
        P-type doping concentration in cm^-3 (default: 1e17)
    n_doping : float
        N-type doping concentration in cm^-3 (default: 1e17)
    intrinsic_doping : float
        Intrinsic region doping concentration in cm^-3 (default: 1.5e10).
        Only used when intrinsic_width > 0.
    temperature : float
        Simulation temperature in Kelvin (default: 300)
    srh : bool
        Enable Shockley-Read-Hall recombination (default: True)
    conmob : bool
        Enable concentration-dependent mobility (default: True)
    fldmob : bool
        Enable field-dependent mobility (default: True)
    impact : bool
        Enable impact ionization (default: False)
    taun0 : float
        Electron lifetime in seconds (default: 1e-6)
    taup0 : float
        Hole lifetime in seconds (default: 1e-6)
    title : str, optional
        Simulation title
    postscript : bool
        Enable PostScript output (default: False)
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
        Example: (0.0, 0.8, 0.05) sweeps from 0V to 0.8V in 0.05V steps
    reverse_sweep : tuple (v_start, v_end, v_step), optional
        If provided, adds a reverse bias voltage sweep.
        Example: (0.0, -5.0, -0.5) sweeps from 0V to -5V in 0.5V steps
    sweep_electrode : int
        Electrode number to apply voltage sweeps (default: 2)
    log_physics_at : list of float, optional
        Bias voltages (V) at which to capture physics profiles along the
        device: recombination rate, electric field, electron density, hole
        density, and net carrier density.  When provided the forward sweep
        is replaced by a step-by-step ramp that inserts Plot1D commands
        between each bias point.  The list must start at 0.0 (equilibrium
        is always logged first).
        Example: [0.0, 0.2, 0.4, 0.6]

    Returns
    -------
    Simulation
        Configured PN diode simulation ready to run

    Example
    -------
    >>> # Basic PN diode - no solve commands, add your own
    >>> sim = create_pn_diode(length=2.0, p_doping=1e16, n_doping=1e18)
    >>> sim.add_solve(Solve(initial=True))
    >>> print(sim.generate_deck())
    >>>
    >>> # PIN diode with 0.2 um intrinsic region
    >>> sim = create_pn_diode(
    ...     length=1.0, junction_position=0.3, intrinsic_width=0.4,
    ...     p_doping=1e17, n_doping=1e17,
    ...     log_iv=True, forward_sweep=(0.0, 1.0, 0.05)
    ... )
    >>> result = sim.run()
    >>>
    >>> # Complete PN simulation with sweeps and logging
    >>> sim = create_pn_diode(
    ...     log_iv=True,
    ...     log_bands_eq=True,
    ...     log_bands_bias=True,
    ...     forward_sweep=(0.0, 0.8, 0.05),
    ...     reverse_sweep=(0.0, -5.0, -0.5)
    ... )
    >>> result = sim.run()
    >>> sim.plot_band_diagram()  # Plots all logged band diagrams
    """
    is_pin = intrinsic_width > 0.0
    sim = Simulation(title=title or ("PIN Diode" if is_pin else "PN Junction Diode"))
    sim._device_type = "pn_diode"
    sim._device_kwargs = dict(
        length=length, width=width, device_z_width=device_z_width,
        junction_position=junction_position,
        intrinsic_width=intrinsic_width, nx=nx, ny=ny,
        p_doping=p_doping, n_doping=n_doping, intrinsic_doping=intrinsic_doping,
        temperature=temperature, srh=srh, conmob=conmob, fldmob=fldmob,
        impact=impact, taun0=taun0, taup0=taup0, title=title,
        postscript=postscript, log_iv=log_iv, iv_file=iv_file,
        log_bands_eq=log_bands_eq, log_bands_bias=log_bands_bias,
        forward_sweep=forward_sweep, reverse_sweep=reverse_sweep,
        sweep_electrode=sweep_electrode, log_physics_at=log_physics_at,
    )

    # Options
    if postscript:
        sim.options = Options(postscript=True)

    check_mesh_size(nx, ny, "create_pn_diode")

    # Compute junction boundaries
    junction_x = junction_position * length           # P-I (or P-N) boundary
    intrinsic_end = junction_x + intrinsic_width      # I-N boundary (== junction_x when PN)

    # Mesh with refinement near junction(s).
    # width= is the z-depth (current scaling), not the y extent.
    sim.mesh = Mesh(nx=nx, ny=ny, width=device_z_width, outfile="mesh")

    if is_pin:
        # Place roughly equal mesh density in each region; refine at both junctions
        nx_p = max(2, int(nx * junction_x / length))
        nx_i = max(2, int(nx * intrinsic_width / length))
        nx_n = nx - nx_p - nx_i
        nx_pi = nx_p                        # index of P-I interface
        nx_in = nx_p + nx_i                 # index of I-N interface

        sim.mesh.add_x_mesh(1, 0, ratio=1)
        sim.mesh.add_x_mesh(nx_pi, junction_x, ratio=0.8)
        sim.mesh.add_x_mesh(nx_in, intrinsic_end, ratio=1.05)
        sim.mesh.add_x_mesh(nx, length, ratio=1.05)
    else:
        mid_point = nx // 2
        sim.mesh.add_x_mesh(1, 0, ratio=1)
        sim.mesh.add_x_mesh(mid_point, junction_x, ratio=0.8)
        sim.mesh.add_x_mesh(nx, length, ratio=1.05)

    sim.mesh.add_y_mesh(1, 0, ratio=1)
    sim.mesh.add_y_mesh(ny, width, ratio=1)

    # Silicon regions
    if is_pin:
        sim.add_region(Region(1, ix_low=1, ix_high=nx_pi, iy_low=1, iy_high=ny, silicon=True))
        sim.add_region(Region(1, ix_low=nx_pi, ix_high=nx_in, iy_low=1, iy_high=ny, silicon=True))
        sim.add_region(Region(1, ix_low=nx_in, ix_high=nx, iy_low=1, iy_high=ny, silicon=True))
    else:
        mid_point = nx // 2
        sim.add_region(Region(1, ix_low=1, ix_high=mid_point, iy_low=1, iy_high=ny, silicon=True))
        sim.add_region(Region(1, ix_low=mid_point, ix_high=nx, iy_low=1, iy_high=ny, silicon=True))

    # Electrodes at device ends
    sim.add_electrode(Electrode(1, ix_low=1, ix_high=1, iy_low=1, iy_high=ny))
    sim.add_electrode(Electrode(2, ix_low=nx, ix_high=nx, iy_low=1, iy_high=ny))

    # Doping
    sim.add_doping(Doping(region=1, p_type=True, concentration=p_doping,
                          x_left=0, x_right=junction_x, y_top=0, y_bottom=width, uniform=True))
    if is_pin:
        sim.add_doping(Doping(region=1, n_type=True, concentration=intrinsic_doping,
                              x_left=junction_x, x_right=intrinsic_end,
                              y_top=0, y_bottom=width, uniform=True))
    sim.add_doping(Doping(region=1, n_type=True, concentration=n_doping,
                          x_left=intrinsic_end, x_right=length, y_top=0, y_bottom=width, uniform=True))

    # Contacts - ohmic contacts for all electrodes
    sim.add_contact(Contact(all_contacts=True, neutral=True))

    # Material with lifetimes
    sim.add_material(Material(name="silicon", taun0=taun0, taup0=taup0,
                              trap_type="0", etrap=0))

    # Physical models
    sim.models = Models(srh=srh, conmob=conmob, fldmob=fldmob, impact=impact,
                        temperature=temperature)
    sim.system = System(electrons=True, holes=True, newton=True)

    # I-V logging
    if log_iv:
        sim.add_log(Log(ivfile=iv_file))

    # Line cut position for band diagrams (horizontal through middle of device)
    y_cut = width / 2.0

    # Only add solve commands if sweeps or logging are specified
    if (forward_sweep is not None or reverse_sweep is not None
            or log_bands_eq or log_physics_at is not None):
        # Always start with equilibrium solve
        sim.add_solve(Solve(initial=True, outfile="eq"))
        n_prior = 1     # solutions computed so far
        v_sweep = 0.0   # last solved bias on the sweep electrode

        # Log equilibrium band diagram if requested
        if log_bands_eq:
            sim.log_band_diagram(
                outfile_prefix="eq",
                x_start=0.0, y_start=y_cut,
                x_end=length, y_end=y_cut
            )

        # Physics-profile stepping: step one bias at a time so that
        # Plot1D commands fire at each requested voltage.
        # When active this replaces forward_sweep.
        if log_physics_at is not None:
            _cut = dict(x_start=0.0, y_start=y_cut,
                        x_end=length, y_end=y_cut)

            # Equilibrium snapshot
            sim.log_physics("eq", **_cut)

            # Step through remaining bias points one at a time
            # (the first step after INIT must use PREV, not PROJ)
            v_prev = 0.0
            for v in log_physics_at[1:]:
                dv  = v - v_prev
                tag = f"v{v:.1f}".replace(".", "p")   # e.g. "v0p2"
                solve_kwargs = dict(**solve_guess(n_prior),
                                    vstep=dv, nsteps=1,
                                    electrode=sweep_electrode, outfile=f"sol_{tag}")
                if sweep_electrode == 1:
                    solve_kwargs["v1"] = v_prev
                else:
                    solve_kwargs["v2"] = v_prev
                sim.add_solve(Solve(**solve_kwargs))
                n_prior += 2
                sim.log_physics(tag, **_cut)
                v_prev = v
            v_sweep = v_prev

        # Forward bias sweep (skipped when log_physics_at is used)
        elif forward_sweep is not None:
            v_start, v_end, v_step = forward_sweep
            nsteps = int(abs(v_end - v_start) / abs(v_step))
            sim.add_solve(Solve(
                **solve_guess(n_prior),
                v1=v_start if sweep_electrode == 1 else 0.0,
                v2=v_start if sweep_electrode == 2 else 0.0,
                vstep=v_step,
                nsteps=nsteps,
                electrode=sweep_electrode,
                outfile="fwd"
            ))
            n_prior += nsteps + 1
            v_sweep = v_start + v_step * nsteps
            if log_bands_bias:
                sim.log_band_diagram(
                    outfile_prefix="fwd",
                    x_start=0.0, y_start=y_cut,
                    x_end=length, y_end=y_cut,
                    include_qf=True
                )

        # Reverse bias sweep
        if reverse_sweep is not None:
            v_start, v_end, v_step = reverse_sweep
            nsteps = int(abs(v_end - v_start) / abs(v_step))

            # A previous sweep left the diode at a different bias: ramp
            # back to the reverse-sweep start instead of jumping.  The
            # ramp retraces already-logged forward bias points, so the
            # I-V data is unaffected apart from duplicates.
            if abs(v_start - v_sweep) > 1e-10:
                n_prior = add_bias_ramp(sim, sweep_electrode, v_sweep,
                                        v_start, n_prior,
                                        max_step=GATE_STEP,
                                        outfile="rev_ramp")
                v_sweep = v_start

            sim.add_solve(Solve(
                **solve_guess(n_prior),
                v1=v_start if sweep_electrode == 1 else 0.0,
                v2=v_start if sweep_electrode == 2 else 0.0,
                vstep=v_step,
                nsteps=nsteps,
                electrode=sweep_electrode,
                outfile="rev"
            ))
            n_prior += nsteps + 1
            v_sweep = v_start + v_step * nsteps
            if log_bands_bias:
                sim.log_band_diagram(
                    outfile_prefix="rev",
                    x_start=0.0, y_start=y_cut,
                    x_end=length, y_end=y_cut,
                    include_qf=True
                )

    return sim


# Alias
pn_diode = create_pn_diode
