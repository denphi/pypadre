"""
Bipolar Junction Transistor (BJT) factory function.
"""

from typing import Optional, Tuple, List
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
from ..plot3d import Plot3D
from ._common import (
    check_mesh_size, solve_guess, add_bias_ramp, JUNCTION_STEP, DRAIN_STEP,
)


def create_bjt(
    # Geometry parameters
    emitter_width: float = 1.0,
    base_width: float = 0.5,
    collector_width: float = 2.0,
    device_depth: float = 1.0,
    device_z_width: float = 1.0,
    # Mesh parameters (keep nx*ny below the ~2500-node nanoHUB limit)
    nx: int = 60,
    ny: int = 30,
    # Doping parameters
    emitter_doping: float = 1e20,
    base_doping: float = 1e17,
    collector_doping: float = 1e16,
    device_type: str = "npn",
    # Minority-carrier lifetimes per region (reference tool values)
    emitter_tau: float = 1e-7,
    base_tau: float = 1e-6,
    collector_tau: float = 1e-7,
    # Physical models
    temperature: float = 300,
    srh: bool = True,
    auger: bool = True,
    bgn: bool = True,
    conmob: bool = True,
    fldmob: bool = True,
    surf_rec_contacts: bool = True,
    # Simulation options
    title: Optional[str] = None,
    # Output logging options
    log_iv: bool = False,
    iv_file: str = "ic_vce.log",
    log_bands_eq: bool = False,
    # Voltage sweep options - Common-emitter output characteristics
    vbe: float = 0.0,
    vce_sweep: Optional[Tuple[float, float, float]] = None,
    # Gummel plot sweep
    gummel_sweep: Optional[Tuple[float, float, float]] = None,
    gummel_vce: float = 2.0,
    # 2D contour map options
    contour_maps: bool = False,
    contour_vbe: float = 0.7,
    contour_vce: float = 2.0,
    contour_quantities: Optional[List[str]] = None,
) -> Simulation:
    """
    Create a Bipolar Junction Transistor (BJT) simulation.

    Creates a 1D-like NPN or PNP transistor structure.

    Parameters
    ----------
    emitter_width : float
        Emitter region width in microns (default: 1.0)
    base_width : float
        Base region width in microns (default: 0.5)
    collector_width : float
        Collector region width in microns (default: 2.0)
    device_depth : float
        Device depth in microns (default: 1.0)
    device_z_width : float
        Depth of the device in the third dimension in microns (default:
        1.0). Terminal currents scale linearly with this value.
    nx : int
        Mesh points in x direction (default: 60)
    ny : int
        Mesh points in y direction (default: 30)
    emitter_doping : float
        Emitter doping concentration in cm^-3 (default: 1e20)
    base_doping : float
        Base doping concentration in cm^-3 (default: 1e17)
    collector_doping : float
        Collector doping concentration in cm^-3 (default: 1e16)
    device_type : str
        "npn" or "pnp" (default: "npn")
    emitter_tau : float
        Minority-carrier lifetime in the emitter in seconds (default: 1e-7,
        matching the nanoHUB reference tool)
    base_tau : float
        Minority-carrier lifetime in the base in seconds (default: 1e-6)
    collector_tau : float
        Minority-carrier lifetime in the collector in seconds (default: 1e-7)
    temperature : float
        Simulation temperature in Kelvin (default: 300)
    surf_rec_contacts : bool
        Use surface-recombination contacts as in the reference tool: all
        contacts get finite surface recombination, and the base contact
        blocks minority-carrier (collector-side) recombination (VSURFN=0
        for NPN, VSURFP=0 for PNP) so the base current reflects real base
        injection (default: True)
    srh : bool
        Enable Shockley-Read-Hall recombination (default: True)
    auger : bool
        Enable Auger recombination (default: True)
    bgn : bool
        Enable band-gap narrowing (default: True)
    conmob : bool
        Enable concentration-dependent mobility (default: True)
    fldmob : bool
        Enable field-dependent mobility (default: True)
    title : str, optional
        Simulation title
    log_iv : bool
        If True, add I-V logging (default: False)
    iv_file : str
        Filename for I-V log (default: "ic_vce.log")
    log_bands_eq : bool
        If True, log band diagrams at equilibrium (default: False)
    vbe : float
        Base-emitter voltage for output characteristics (default: 0.0)
    vce_sweep : tuple (v_start, v_end, v_step), optional
        Collector-emitter voltage sweep for output characteristics (Ic vs Vce).
        Example: (0.0, 3.0, 0.1) sweeps Vce from 0V to 3V
    gummel_sweep : tuple (v_start, v_end, v_step), optional
        Base-emitter voltage sweep for Gummel plot (Ic, Ib vs Vbe).
        Example: (0.0, 0.8, 0.05) sweeps Vbe from 0V to 0.8V
    gummel_vce : float
        Fixed Vce for Gummel plot (default: 2.0)
    contour_maps : bool
        If True, add Plot3D scatter file dumps at equilibrium and under bias
        for 2D contour visualization (default: False)
    contour_vbe : float
        Base-emitter voltage for contour map bias point (default: 0.7)
    contour_vce : float
        Collector-emitter voltage for contour map bias point (default: 2.0)
    contour_quantities : list of str, optional
        Quantities to dump. Default: ["potential", "doping", "electrons",
        "holes", "e_field", "qfn", "qfp"]

    Returns
    -------
    Simulation
        Configured BJT simulation

    Example
    -------
    >>> # Basic BJT - add your own solve commands
    >>> sim = create_bjt(device_type="npn", base_width=0.3)
    >>> sim.add_solve(Solve(initial=True))
    >>> print(sim.generate_deck())
    >>>
    >>> # Common-emitter output characteristic
    >>> sim = create_bjt(
    ...     log_iv=True,
    ...     vbe=0.7,
    ...     vce_sweep=(0.0, 3.0, 0.1)
    ... )
    >>> result = sim.run()
    >>>
    >>> # Gummel plot
    >>> sim = create_bjt(
    ...     log_iv=True,
    ...     iv_file="gummel",
    ...     gummel_sweep=(0.0, 0.8, 0.05),
    ...     gummel_vce=2.0
    ... )
    >>>
    >>> # 2D contour maps
    >>> sim = create_bjt(contour_maps=True, contour_vbe=0.7, contour_vce=2.0)
    >>> result = sim.run()
    >>> sim.plot_contour("pot_eq", title="Potential — Equilibrium", cbar_title="V")
    """
    is_npn = device_type.lower() == "npn"
    sim = Simulation(title=title or f"{'NPN' if is_npn else 'PNP'} BJT")
    sim._device_type = "bjt"
    sim._device_kwargs = dict(
        emitter_width=emitter_width, base_width=base_width,
        collector_width=collector_width, device_depth=device_depth,
        device_z_width=device_z_width,
        nx=nx, ny=ny, emitter_doping=emitter_doping, base_doping=base_doping,
        collector_doping=collector_doping, device_type=device_type,
        emitter_tau=emitter_tau, base_tau=base_tau, collector_tau=collector_tau,
        temperature=temperature, srh=srh, auger=auger, bgn=bgn,
        conmob=conmob, fldmob=fldmob, surf_rec_contacts=surf_rec_contacts,
        title=title, log_iv=log_iv,
        iv_file=iv_file, log_bands_eq=log_bands_eq, vbe=vbe,
        vce_sweep=vce_sweep, gummel_sweep=gummel_sweep, gummel_vce=gummel_vce,
        contour_maps=contour_maps, contour_vbe=contour_vbe,
        contour_vce=contour_vce, contour_quantities=contour_quantities,
    )

    total_width = emitter_width + base_width + collector_width

    check_mesh_size(nx, ny, "create_bjt")

    # Mesh point distribution (guard against degenerate integer truncation)
    nx_e = max(2, int(nx * emitter_width / total_width))
    nx_b = max(2, int(nx * base_width / total_width))
    nx_c = nx - nx_e - nx_b
    if nx_c < 2:
        raise ValueError(
            f"nx={nx} leaves only {nx_c} collector mesh columns; increase nx "
            f"or rebalance the region widths."
        )

    # X mesh, refined into both junctions (E-B at x=emitter_width, B-C at
    # x=emitter_width+base_width) like the reference deck: contract toward
    # each junction, expand away from it.
    x_eb = emitter_width
    x_bc = emitter_width + base_width
    nx_bmid = nx_e + max(1, nx_b // 2)

    # width= is the z-depth; omit when 1.0 (the PADRE default)
    _zw = None if device_z_width == 1.0 else device_z_width
    sim.mesh = Mesh(nx=nx, ny=ny, width=_zw)
    sim.mesh.add_x_mesh(1, 0, ratio=1)
    sim.mesh.add_x_mesh(nx_e, x_eb, ratio=0.8)                      # fine at E-B
    sim.mesh.add_x_mesh(nx_bmid, x_eb + base_width / 2, ratio=1.2)  # expand from E-B
    sim.mesh.add_x_mesh(nx_e + nx_b, x_bc, ratio=0.8)               # fine at B-C
    sim.mesh.add_x_mesh(nx, total_width, ratio=1.2)                 # expand from B-C
    sim.mesh.add_y_mesh(1, 0, ratio=1)
    sim.mesh.add_y_mesh(ny, device_depth, ratio=1)

    # Regions — named materials so each gets its own minority lifetime
    sim.add_region(Region(1, ix_low=1, ix_high=nx_e, iy_low=1, iy_high=ny,
                          material="emat", semiconductor=True))  # Emitter
    sim.add_region(Region(2, ix_low=nx_e, ix_high=nx_e + nx_b, iy_low=1, iy_high=ny,
                          material="bmat", semiconductor=True))  # Base
    sim.add_region(Region(3, ix_low=nx_e + nx_b, ix_high=nx, iy_low=1, iy_high=ny,
                          material="cmat", semiconductor=True))  # Collector

    # Electrodes.  The base contact is inset from both junction columns:
    # an ohmic contact touching a junction depletion region acts as a
    # carrier sink there and corrupts the base current.
    b_pad = max(1, nx_b // 4)
    b_lo = nx_e + b_pad
    b_hi = nx_e + nx_b - b_pad
    if b_hi < b_lo:
        b_lo = b_hi = nx_e + nx_b // 2
    sim.add_electrode(Electrode(1, ix_low=1, ix_high=1, iy_low=1, iy_high=ny))  # Emitter contact
    sim.add_electrode(Electrode(2, ix_low=b_lo, ix_high=b_hi, iy_low=ny, iy_high=ny))  # Base contact
    sim.add_electrode(Electrode(3, ix_low=nx, ix_high=nx, iy_low=1, iy_high=ny))  # Collector contact

    # Doping (NPN: n-emitter, p-base, n-collector)
    if is_npn:
        sim.add_doping(Doping(region=1, n_type=True, uniform=True, concentration=emitter_doping))
        sim.add_doping(Doping(region=2, p_type=True, uniform=True, concentration=base_doping))
        sim.add_doping(Doping(region=3, n_type=True, uniform=True, concentration=collector_doping))
    else:
        sim.add_doping(Doping(region=1, p_type=True, uniform=True, concentration=emitter_doping))
        sim.add_doping(Doping(region=2, n_type=True, uniform=True, concentration=base_doping))
        sim.add_doping(Doping(region=3, p_type=True, uniform=True, concentration=collector_doping))

    # Contacts — reference tool uses surface-recombination contacts, with
    # the base contact blocking recombination of the carrier collected at
    # the collector (electrons for NPN) so it only supplies base current.
    sim.add_contact(Contact(all_contacts=True, neutral=True))
    if surf_rec_contacts:
        sim.add_contact(Contact(number=1, n_surf_rec=True, p_surf_rec=True))
        sim.add_contact(Contact(number=3, n_surf_rec=True, p_surf_rec=True))
        if is_npn:
            sim.add_contact(Contact(number=2, n_surf_rec=True, p_surf_rec=True,
                                    vsurfn=0))
        else:
            sim.add_contact(Contact(number=2, n_surf_rec=True, p_surf_rec=True,
                                    vsurfp=0))

    # Materials — per-region minority-carrier lifetimes (reference deck:
    # Emat/Cmat taup0=1e-7, Bmat taun0=1e-6 for NPN; mirrored for PNP)
    if is_npn:
        sim.add_material(Material(name="emat", default="silicon",
                                  taup0=emitter_tau, permittivity=11.8))
        sim.add_material(Material(name="bmat", default="silicon",
                                  taun0=base_tau, permittivity=11.8))
        sim.add_material(Material(name="cmat", default="silicon",
                                  taup0=collector_tau, permittivity=11.8))
    else:
        sim.add_material(Material(name="emat", default="silicon",
                                  taun0=emitter_tau, permittivity=11.8))
        sim.add_material(Material(name="bmat", default="silicon",
                                  taup0=base_tau, permittivity=11.8))
        sim.add_material(Material(name="cmat", default="silicon",
                                  taun0=collector_tau, permittivity=11.8))

    # Models
    sim.models = Models(temperature=temperature, srh=srh, auger=auger, bgn=bgn,
                        conmob=conmob, fldmob=fldmob)
    sim.system = System(electrons=True, holes=True, newton=True)

    # I-V logging.  When bias solves are generated below, the LOG statement
    # is inserted right before the characteristic sweep instead, so that
    # equilibrium/pre-bias ramp points do not pollute the I-V file.
    has_sweeps = vce_sweep is not None or gummel_sweep is not None
    if log_iv and not has_sweeps:
        sim.add_log(Log(ivfile=iv_file))

    # Only add solve commands if sweeps or contour maps are specified
    if vce_sweep is not None or gummel_sweep is not None or log_bands_eq or contour_maps:
        # Always start with equilibrium solve
        sim.add_solve(Solve(initial=True, outfile="eq_sol"))
        n_prior = 1                       # solutions computed so far
        bias = {1: 0.0, 2: 0.0, 3: 0.0}   # last solved electrode biases
        log_pending = log_iv and has_sweeps

        # Log band diagram at equilibrium (horizontal cut along device from emitter to collector)
        if log_bands_eq:
            y_mid = device_depth / 2
            sim.log_band_diagram(
                outfile_prefix="eq",
                x_start=0.0, x_end=total_width,
                y_start=y_mid, y_end=y_mid
            )

        # Common-emitter output characteristics (Ic vs Vce at fixed Vbe)
        if vce_sweep is not None:
            v_start, v_end, v_step = vce_sweep
            nsteps = int(abs(v_end - v_start) / abs(v_step))

            # Ramp base-emitter voltage up in small steps (forward-biased
            # junction: never applied as a single jump)
            if abs(vbe - bias[2]) > 1e-10:
                n_prior = add_bias_ramp(sim, 2, bias[2], vbe, n_prior,
                                        max_step=JUNCTION_STEP,
                                        outfile="vbe_set_sol")
                bias[2] = vbe

            # Enable I-V logging only now, so ramp points stay out of the file
            if log_pending:
                sim.add_log(Log(ivfile=iv_file))
                log_pending = False

            # Sweep collector-emitter voltage
            sim.add_solve(Solve(
                **solve_guess(n_prior),
                v3=v_start,
                vstep=v_step,
                nsteps=nsteps,
                electrode=3,
                outfile="ic_vce_sol"
            ))
            n_prior += nsteps + 1
            bias[3] = v_start + v_step * nsteps

        # Gummel plot (Ic, Ib vs Vbe at fixed Vce)
        if gummel_sweep is not None:
            v_start, v_end, v_step = gummel_sweep
            nsteps = int(abs(v_end - v_start) / abs(v_step))

            # Ramp collector-emitter voltage to the operating point
            if abs(gummel_vce - bias[3]) > 1e-10:
                n_prior = add_bias_ramp(sim, 3, bias[3], gummel_vce, n_prior,
                                        max_step=DRAIN_STEP,
                                        outfile="vce_set_sol")
                bias[3] = gummel_vce

            # Ramp base back to the sweep start if a previous sweep moved it
            if abs(v_start - bias[2]) > 1e-10:
                n_prior = add_bias_ramp(sim, 2, bias[2], v_start, n_prior,
                                        max_step=JUNCTION_STEP,
                                        outfile="vbe_start_sol")
                bias[2] = v_start

            if log_pending:
                sim.add_log(Log(ivfile=iv_file))
                log_pending = False

            # Sweep base-emitter voltage
            sim.add_solve(Solve(
                **solve_guess(n_prior),
                v2=v_start,
                vstep=v_step,
                nsteps=nsteps,
                electrode=2,
                outfile="gummel_sol"
            ))
            n_prior += nsteps + 1
            bias[2] = v_start + v_step * nsteps

        # 2D contour maps (Plot3D scatter files)
        if contour_maps:
            quantities = contour_quantities or [
                "potential", "doping", "electrons", "holes",
                "e_field", "qfn", "qfp",
            ]

            # Map quantity names to Plot3D kwargs and output file suffixes
            _qty_map = {
                "potential":  ("potential",  "pot"),
                "doping":     ("doping",     "dop"),
                "electrons":  ("electrons",  "el"),
                "holes":      ("holes",      "hh"),
                "e_field":    ("e_field",    "ef"),
                "qfn":        ("qfn",        "qfn"),
                "qfp":        ("qfp",        "qfp"),
                "band_val":   ("band_val",   "bv"),
                "band_cond":  ("band_cond",  "bc"),
                "net_charge": ("net_charge", "nch"),
                "recomb":     ("recomb",     "rec"),
            }

            # Equilibrium dumps
            for qty in quantities:
                if qty not in _qty_map:
                    continue
                kwarg, suffix = _qty_map[qty]
                extras = {"absolute": True} if qty == "doping" else {}
                sim.add_command(Plot3D(**{kwarg: True}, outfile=f"{suffix}_eq", **extras))

            # Bias solve for contour maps (only if no other sweep already applied bias)
            if vce_sweep is None and gummel_sweep is None:
                # Ramp Vbe in small steps (forward-biased junction)
                if abs(contour_vbe) > 1e-10:
                    n_prior = add_bias_ramp(sim, 2, bias[2], contour_vbe,
                                            n_prior, max_step=JUNCTION_STEP,
                                            outfile="contour_vbe_set")
                    bias[2] = contour_vbe
                # Ramp Vce
                if abs(contour_vce) > 1e-10:
                    n_prior = add_bias_ramp(sim, 3, bias[3], contour_vce,
                                            n_prior, max_step=DRAIN_STEP,
                                            outfile="contour_bias")
                    bias[3] = contour_vce

            # Bias dumps
            for qty in quantities:
                if qty == "doping":
                    continue  # doping doesn't change with bias
                if qty not in _qty_map:
                    continue
                kwarg, suffix = _qty_map[qty]
                sim.add_command(Plot3D(**{kwarg: True}, outfile=f"{suffix}_bias"))

    return sim


# Alias
bjt = create_bjt
