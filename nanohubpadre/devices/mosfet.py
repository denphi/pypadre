"""
MOSFET factory function.
"""

import warnings
from typing import Optional, Tuple, List
from ..estimates import estimate_mosfet_vt
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
    check_mesh_size, solve_guess, add_bias_ramp, GATE_STEP, DRAIN_STEP,
)


def create_mosfet(
    # Geometry parameters (defaults match the nanoHUB Rappture MOSFET tool)
    channel_length: float = 0.15,
    gate_oxide_thickness: float = 0.002,
    junction_depth: float = 0.02,
    device_width: float = 0.25,
    device_depth: float = 0.05,
    device_z_width: float = 1.0,
    # Mesh parameters
    nx: int = 53,
    ny: int = 46,
    # Doping parameters
    channel_doping: float = 1e18,
    substrate_doping: float = 5e16,
    source_drain_doping: float = 2e20,
    device_type: str = "nmos",
    # Physical models
    temperature: float = 300,
    bgn: bool = True,
    conmob: bool = True,
    fldmob: bool = True,
    gatmob: bool = True,
    carriers: int = 1,
    # Simulation options
    title: Optional[str] = None,
    # Output logging options
    log_iv: bool = False,
    iv_file: str = "idvg",
    log_bands_eq: bool = False,
    log_bands_bias: bool = False,
    # Voltage sweep options
    vgs_sweep: Optional[Tuple[float, float, float]] = None,
    vds: float = 0.0,
    vds_sweep: Optional[Tuple[float, float, float]] = None,
    vgs: float = 0.0,
    # 2D contour map options
    contour_maps: bool = False,
    contour_vgs: float = 1.0,
    contour_vds: float = 1.0,
    contour_quantities: Optional[List[str]] = None,
) -> Simulation:
    """
    Create a MOSFET simulation.

    Creates an NMOS or PMOS transistor structure with source, drain,
    gate, and substrate contacts.

    Parameters
    ----------
    channel_length : float
        Gate/channel length in microns (default: 0.15)
    gate_oxide_thickness : float
        Gate oxide thickness in microns (default: 0.002 = 2 nm)
    junction_depth : float
        Source/drain junction depth in microns (default: 0.02)
    device_width : float
        Total device width in microns (default: 0.25)
    device_depth : float
        Substrate depth in microns (default: 0.05)
    device_z_width : float
        Depth of the device in the third dimension in microns (default:
        1.0). Terminal currents scale linearly with this value.
    nx : int
        Mesh points in x direction (default: 53)
    ny : int
        Mesh points in y direction (default: 46)
    channel_doping : float
        Channel doping concentration in cm^-3 (default: 1e18)
    substrate_doping : float
        Substrate doping concentration in cm^-3 (default: 5e16)
    source_drain_doping : float
        Source/drain doping concentration in cm^-3 (default: 2e20)
    device_type : str
        "nmos" or "pmos" (default: "nmos")
    temperature : float
        Simulation temperature in Kelvin (default: 300)
    bgn : bool
        Enable band-gap narrowing (default: True)
    conmob : bool
        Enable concentration-dependent mobility (default: True)
    fldmob : bool
        Enable field-dependent mobility / velocity saturation (default: True).
        Required for correct drain-current saturation.
    gatmob : bool
        Enable transverse (gate) field mobility degradation (default: True)
    carriers : int
        Number of carriers to solve (1 or 2, default: 1)
    title : str, optional
        Simulation title
    log_iv : bool
        If True, add I-V logging (default: False)
    iv_file : str
        Filename for I-V log (default: "idvg")
    log_bands_eq : bool
        If True, log band diagrams at equilibrium (default: False)
    log_bands_bias : bool
        If True, log band diagrams during voltage sweeps (default: False)
    vgs_sweep : tuple (v_start, v_end, v_step), optional
        Gate voltage sweep for transfer characteristic (Id-Vg).
        Example: (0.0, 1.5, 0.1) sweeps Vg from 0V to 1.5V in 0.1V steps
    vds : float
        Drain voltage for transfer characteristic (default: 0.0)
    vds_sweep : tuple (v_start, v_end, v_step), optional
        Drain voltage sweep for output characteristic (Id-Vd).
        Example: (0.0, 1.0, 0.05) sweeps Vd from 0V to 1.0V in 0.05V steps
    vgs : float
        Gate voltage for output characteristic (default: 0.0)
    contour_maps : bool
        If True, add Plot3D scatter file dumps at equilibrium and under bias
        for 2D contour visualization (default: False)
    contour_vgs : float
        Gate voltage for contour map bias point (default: 1.0)
    contour_vds : float
        Drain voltage for contour map bias point (default: 1.0)
    contour_quantities : list of str, optional
        Quantities to dump. Default: ["potential", "doping", "electrons",
        "holes", "e_field", "qfn", "qfp"]

    Returns
    -------
    Simulation
        Configured MOSFET simulation

    Example
    -------
    >>> # Basic MOSFET - no solve commands, add your own
    >>> sim = create_mosfet(channel_length=0.05, device_type="nmos")
    >>> sim.add_solve(Solve(initial=True))
    >>> print(sim.generate_deck())
    >>>
    >>> # Complete transfer characteristic
    >>> sim = create_mosfet(
    ...     log_iv=True,
    ...     vgs_sweep=(0.0, 1.5, 0.1),
    ...     vds=0.1
    ... )
    >>> result = sim.run()
    >>> # Access I-V data
    >>> iv = sim.outputs.get("idvg")
    """
    is_nmos = device_type.lower() == "nmos"
    sim = Simulation(title=title or f"{'NMOS' if is_nmos else 'PMOS'} MOSFET")
    sim._device_type = "mosfet"
    sim._device_kwargs = dict(
        channel_length=channel_length, gate_oxide_thickness=gate_oxide_thickness,
        junction_depth=junction_depth, device_width=device_width,
        device_depth=device_depth, device_z_width=device_z_width,
        nx=nx, ny=ny, channel_doping=channel_doping,
        substrate_doping=substrate_doping, source_drain_doping=source_drain_doping,
        device_type=device_type, temperature=temperature, bgn=bgn,
        conmob=conmob, fldmob=fldmob, gatmob=gatmob, carriers=carriers,
        title=title, log_iv=log_iv, iv_file=iv_file, log_bands_eq=log_bands_eq,
        log_bands_bias=log_bands_bias, vgs_sweep=vgs_sweep, vds=vds,
        vds_sweep=vds_sweep, vgs=vgs,
        contour_maps=contour_maps, contour_vgs=contour_vgs,
        contour_vds=contour_vds, contour_quantities=contour_quantities,
    )

    # Validate geometry
    if channel_length >= device_width:
        raise ValueError(
            f"channel_length ({channel_length} µm) must be less than device_width "
            f"({device_width} µm). Use device_width > channel_length + 0.1 µm "
            f"to leave room for source/drain regions."
        )

    check_mesh_size(nx, ny, "create_mosfet")

    # Calculate dimensions
    sd_width = (device_width - channel_length) / 2
    total_height = device_depth + junction_depth + gate_oxide_thickness

    # Mesh points distribution.  The y nodes follow the reference deck's
    # allocation (~20% substrate, ~70% junction/channel zone, ~10% oxide)
    # rather than a geometric split: the junction zone is where carrier
    # gradients are steepest and needs most of the resolution.
    nx_sd = max(2, int(nx * sd_width / device_width))
    nx_ch = nx - 2 * nx_sd
    ny_sub = max(2, int(ny * 0.2))
    ny_ox = max(2, int(ny * 0.1))
    ny_junc = ny - ny_sub - ny_ox
    if nx_ch < 3:
        raise ValueError(
            f"nx={nx} leaves only {nx_ch} channel mesh columns; increase nx "
            f"or reduce the source/drain width fraction."
        )
    if ny_junc < 4:
        raise ValueError(
            f"ny={ny} leaves only {ny_junc} junction-zone mesh rows; "
            f"increase ny (need at least ~10)."
        )

    # width= is the z-depth; omit when 1.0 (the PADRE default)
    _zw = None if device_z_width == 1.0 else device_z_width
    sim.mesh = Mesh(nx=nx, ny=ny, width=_zw)

    # X mesh (source - channel - drain), refined at both junctions
    sim.mesh.add_x_mesh(1, 0)
    sim.mesh.add_x_mesh(nx_sd, sd_width, ratio=0.8)
    sim.mesh.add_x_mesh(nx_sd + nx_ch // 2, sd_width + channel_length / 2, ratio=1.25)
    sim.mesh.add_x_mesh(nx_sd + nx_ch, sd_width + channel_length, ratio=0.8)
    sim.mesh.add_x_mesh(nx, device_width, ratio=1.25)

    # Y mesh (substrate - junction - oxide).
    # The junction-depth zone is split so spacing contracts toward the
    # Si/SiO2 interface where the inversion layer forms (reference deck
    # uses r=0.87 then r=0.7); the oxide expands away from the interface.
    ny_jsplit = ny_sub + max(1, ny_junc // 2)
    y_jsplit = device_depth + 0.6 * junction_depth
    sim.mesh.add_y_mesh(1, 0)
    sim.mesh.add_y_mesh(ny_sub, device_depth, ratio=0.8)
    sim.mesh.add_y_mesh(ny_jsplit, y_jsplit, ratio=0.875)
    sim.mesh.add_y_mesh(ny_sub + ny_junc, device_depth + junction_depth, ratio=0.7)
    sim.mesh.add_y_mesh(ny, total_height, ratio=1.25)

    # Regions
    # Substrate
    sim.add_region(Region(1, ix_low=1, ix_high=nx, iy_low=1, iy_high=ny_sub, silicon=True))
    # Source region
    sim.add_region(Region(2, ix_low=1, ix_high=nx_sd, iy_low=ny_sub, iy_high=ny_sub + ny_junc, silicon=True))
    # Drain region
    sim.add_region(Region(3, ix_low=nx_sd + nx_ch, ix_high=nx, iy_low=ny_sub, iy_high=ny_sub + ny_junc, silicon=True))
    # Channel region
    sim.add_region(Region(4, ix_low=nx_sd, ix_high=nx_sd + nx_ch, iy_low=ny_sub, iy_high=ny_sub + ny_junc, silicon=True))
    # Gate oxide
    sim.add_region(Region(5, ix_low=nx_sd, ix_high=nx_sd + nx_ch, iy_low=ny_sub + ny_junc, iy_high=ny, oxide=True))
    # Filler oxides over source/drain
    sim.add_region(Region(6, ix_low=1, ix_high=nx_sd, iy_low=ny_sub + ny_junc, iy_high=ny, oxide=True))
    sim.add_region(Region(7, ix_low=nx_sd + nx_ch, ix_high=nx, iy_low=ny_sub + ny_junc, iy_high=ny, oxide=True))

    # Electrodes
    sim.add_electrode(Electrode(1, ix_low=1, ix_high=nx_sd, iy_low=ny_sub + ny_junc, iy_high=ny_sub + ny_junc))  # Source
    sim.add_electrode(Electrode(2, ix_low=nx_sd + nx_ch, ix_high=nx, iy_low=ny_sub + ny_junc, iy_high=ny_sub + ny_junc))  # Drain
    sim.add_electrode(Electrode(3, ix_low=nx_sd, ix_high=nx_sd + nx_ch, iy_low=ny, iy_high=ny))  # Gate
    sim.add_electrode(Electrode(4, ix_low=1, ix_high=nx, iy_low=1, iy_high=1))  # Substrate

    # Doping (NMOS: n+ S/D, p-channel, p-sub; PMOS: opposite)
    if is_nmos:
        sim.add_doping(Doping(region=[2, 3], uniform=True, concentration=source_drain_doping, n_type=True))
        sim.add_doping(Doping(region=4, uniform=True, concentration=channel_doping, p_type=True))
        sim.add_doping(Doping(region=1, uniform=True, concentration=substrate_doping, p_type=True))
    else:
        sim.add_doping(Doping(region=[2, 3], uniform=True, concentration=source_drain_doping, p_type=True))
        sim.add_doping(Doping(region=4, uniform=True, concentration=channel_doping, n_type=True))
        sim.add_doping(Doping(region=1, uniform=True, concentration=substrate_doping, n_type=True))

    # Contacts
    sim.add_contact(Contact(number=3, n_polysilicon=is_nmos, p_polysilicon=not is_nmos))

    # Materials — matches the reference Rappture deck:
    #   material name=silicon eg300=1.12 permi=11.8 In.Model=klaassen
    #   + MUn=1400 VSATn=1.03e+07 En.Mu=2 IIN.MU=55.6,9.9e16,0.66,0.84,3.47e20
    #   material name=sio2 permi=3.9
    sim.add_material(Material(
        name="silicon", eg300=1.12, permittivity=11.8,
        in_model="klaassen", mun=1400, vsatn=1.03e7,
        en_mu=[2], iin_mu=[55.6, 9.9e16, 0.66, 0.84, 3.47e20],
    ))
    sim.add_material(Material(name="sio2", permittivity=3.9))

    # Models — conmob/fldmob/gatmob are required for accurate mobility,
    # velocity saturation (Id saturation), and gate-field degradation.
    sim.models = Models(temperature=temperature, bgn=bgn,
                        conmob=conmob, fldmob=fldmob, gatmob=gatmob,
                        print_models=True)
    if carriers == 1:
        sim.system = System(newton=True, carriers=1, electrons=is_nmos, holes=not is_nmos)
    else:
        sim.system = System(newton=True, carriers=2, electrons=True, holes=True)

    # Sanity check: warn when the analytic threshold voltage lies beyond
    # the requested gate sweep (device would never turn on in the sweep).
    if vgs_sweep is not None:
        vt = estimate_mosfet_vt(channel_doping, gate_oxide_thickness,
                                temperature, is_nmos)
        v_hi = max(vgs_sweep[0], vgs_sweep[1])
        v_lo = min(vgs_sweep[0], vgs_sweep[1])
        beyond = (vt > v_hi) if is_nmos else (vt < v_lo)
        if beyond:
            warnings.warn(
                f"create_mosfet: estimated threshold voltage Vt ≈ {vt:.2f} V "
                f"lies outside the vgs_sweep range [{v_lo}, {v_hi}] V — the "
                f"device will not turn on during the sweep. Reduce "
                f"channel_doping or gate_oxide_thickness, or extend the sweep.",
                UserWarning,
                stacklevel=2,
            )

    # I-V logging.  When bias solves are generated below, the LOG statement
    # is inserted right before the characteristic sweep instead, so that
    # equilibrium/pre-bias ramp points do not pollute the I-V file.
    has_sweeps = vgs_sweep is not None or vds_sweep is not None
    if log_iv and not has_sweeps:
        sim.add_log(Log(ivfile=iv_file))

    # Only add solve commands if sweeps, contour maps, or band logging are specified
    if vgs_sweep is not None or vds_sweep is not None or log_bands_eq or contour_maps:
        # Always start with equilibrium solve
        sim.add_solve(Solve(initial=True, outfile="eq"))
        n_prior = 1                       # solutions computed so far
        bias = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}  # last solved electrode biases
        log_pending = log_iv and has_sweeps

        if log_bands_eq:
            # Vertical cut through middle of channel
            x_mid = device_width / 2.0
            sim.log_band_diagram(
                outfile_prefix="eq",
                x_start=x_mid, y_start=0.0,
                x_end=x_mid, y_end=total_height
            )

        # Transfer characteristic (Id-Vg at fixed Vd)
        if vgs_sweep is not None:
            v_start, v_end, v_step = vgs_sweep
            nsteps = int(abs(v_end - v_start) / abs(v_step))

            # Ramp drain to the operating bias (never a single jump)
            if abs(vds) > 1e-10:
                n_prior = add_bias_ramp(sim, 2, bias[2], vds, n_prior,
                                        max_step=DRAIN_STEP, outfile="vd_set")
                bias[2] = vds

            # Ramp gate to the sweep starting voltage if nonzero
            if abs(v_start - bias[3]) > 1e-10:
                n_prior = add_bias_ramp(sim, 3, bias[3], v_start, n_prior,
                                        max_step=GATE_STEP, outfile="vg_start")
                bias[3] = v_start

            # Enable I-V logging only now, so ramp points stay out of the file
            if log_pending:
                sim.add_log(Log(ivfile=iv_file))
                log_pending = False

            # Sweep gate voltage
            sim.add_solve(Solve(
                **solve_guess(n_prior),
                v3=v_start,
                vstep=v_step,
                nsteps=nsteps,
                electrode=3,
                outfile="idvg"
            ))
            n_prior += nsteps + 1
            bias[3] = v_start + v_step * nsteps

            if log_bands_bias:
                x_mid = device_width / 2.0
                sim.log_band_diagram(
                    outfile_prefix="idvg",
                    x_start=x_mid, y_start=0.0,
                    x_end=x_mid, y_end=total_height,
                    include_qf=True
                )

        # Output characteristic (Id-Vd at fixed Vg)
        if vds_sweep is not None:
            v_start, v_end, v_step = vds_sweep
            nsteps = int(abs(v_end - v_start) / abs(v_step))

            # Ramp gate to the operating bias if not already in transfer mode
            if vgs_sweep is None and abs(vgs) > 1e-10:
                n_prior = add_bias_ramp(sim, 3, bias[3], vgs, n_prior,
                                        max_step=GATE_STEP, outfile="vg_set")
                bias[3] = vgs

            # Ramp drain back to the sweep starting voltage if needed
            if abs(v_start - bias[2]) > 1e-10:
                n_prior = add_bias_ramp(sim, 2, bias[2], v_start, n_prior,
                                        max_step=DRAIN_STEP, outfile="vd_start")
                bias[2] = v_start

            if log_pending:
                sim.add_log(Log(ivfile=iv_file))
                log_pending = False

            # Sweep drain voltage
            sim.add_solve(Solve(
                **solve_guess(n_prior),
                v2=v_start,
                vstep=v_step,
                nsteps=nsteps,
                electrode=2,
                outfile="idvd"
            ))
            n_prior += nsteps + 1
            bias[2] = v_start + v_step * nsteps

            if log_bands_bias:
                x_mid = device_width / 2.0
                sim.log_band_diagram(
                    outfile_prefix="idvd",
                    x_start=x_mid, y_start=0.0,
                    x_end=x_mid, y_end=total_height,
                    include_qf=True
                )

        # 2D contour maps (Plot3D scatter files)
        if contour_maps:
            quantities = contour_quantities or [
                "potential", "doping", "electrons", "holes",
                "e_field", "qfn", "qfp",
            ]

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
            if vgs_sweep is None and vds_sweep is None:
                # Ramp Vgs (gate = electrode 3)
                if abs(contour_vgs) > 1e-10:
                    n_prior = add_bias_ramp(sim, 3, bias[3], contour_vgs,
                                            n_prior, max_step=GATE_STEP,
                                            outfile="contour_vgs_set")
                    bias[3] = contour_vgs
                # Ramp Vds (drain = electrode 2)
                if abs(contour_vds) > 1e-10:
                    n_prior = add_bias_ramp(sim, 2, bias[2], contour_vds,
                                            n_prior, max_step=0.1,
                                            outfile="contour_bias")
                    bias[2] = contour_vds

            # Bias dumps
            for qty in quantities:
                if qty == "doping":
                    continue
                if qty not in _qty_map:
                    continue
                kwarg, suffix = _qty_map[qty]
                sim.add_command(Plot3D(**{kwarg: True}, outfile=f"{suffix}_bias"))

    return sim


# Alias
mosfet = create_mosfet
