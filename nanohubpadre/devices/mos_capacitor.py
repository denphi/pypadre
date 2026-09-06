"""
MOS Capacitor factory function.
"""

import warnings
from typing import Optional, Tuple
from ..estimates import max_depletion_width_um
from ..simulation import Simulation
from ..mesh import Mesh
from ..region import Region
from ..electrode import Electrode
from ..doping import Doping
from ..contact import Contact
from ..interface import Interface, Surface
from ..material import Material
from ..models import Models
from ..solver import System, Solve
from ..log import Log
from ._common import (check_mesh_size, solve_guess, add_bias_ramp, GATE_STEP,
                      sweep_steps, expand_ratio)


def create_mos_capacitor(
    # Geometry parameters
    oxide_thickness: float = 0.002,
    silicon_thickness: Optional[float] = None,
    device_width: float = 1.0,
    device_z_width: float = 1.0,
    # Mesh parameters
    ny_oxide: int = 12,
    ny_silicon: int = 200,
    nx: int = 3,
    # Doping parameters
    substrate_doping: float = 1e16,
    substrate_type: str = "p",
    # Material parameters
    oxide_permittivity: float = 3.9,
    oxide_qf: float = 0,
    oxide_qftrap: float = 0,
    # Carrier lifetimes
    taun0: float = 1e-9,
    taup0: float = 1e-9,
    # Physical models
    temperature: float = 300,
    conmob: bool = True,
    fldmob: bool = True,
    # Gate contact
    gate_type: str = "n_poly",
    gate_workfunction: Optional[float] = None,
    # Gate configuration
    gate_config: str = "single",
    back_oxide_thickness: float = 0.002,
    back_gate_type: str = "n_poly",
    back_gate_workfunction: Optional[float] = None,
    # Simulation options
    title: Optional[str] = None,
    # Output logging options
    log_cv: bool = False,
    cv_file: str = "cv_data",
    log_cv_lf: bool = False,
    cv_lf_file: str = "cv_lf_data",
    log_bands_eq: bool = False,
    log_bands_bias: bool = False,
    log_qf_eq: bool = False,
    log_qf_bias: bool = False,
    log_profiles_eq: bool = False,
    log_profiles_bias: bool = False,
    # Voltage sweep options
    vg_sweep: Optional[Tuple[float, float, float]] = None,
    ac_frequency: float = 1e6,
    ac_frequency_lf: float = 1.0,
) -> Simulation:
    """
    Create a MOS capacitor simulation.

    Creates an oxide-semiconductor structure for C-V analysis.

    Parameters
    ----------
    oxide_thickness : float
        Gate oxide thickness in microns (default: 0.002 = 2nm)
    silicon_thickness : float, optional
        Silicon substrate thickness in microns. Defaults to 5.0 for
        single-gate (matching the nanoHUB reference tool) and 0.03 for
        double-gate (thin body). For single-gate it must comfortably
        exceed the maximum depletion width for the chosen doping or the
        back contact clips the depletion region and distorts the C-V
        minimum; a warning is emitted when silicon_thickness < 2×Wd,max.
    device_width : float
        Device width in microns (default: 1.0)
    device_z_width : float
        Depth of the device in the third dimension in microns (default:
        1.0). Capacitances scale linearly with this value.
    ny_oxide : int
        Mesh points in oxide layer (default: 100)
    ny_silicon : int
        Mesh points in silicon (default: 200)
    nx : int
        Mesh points in x direction (default: 3)
    substrate_doping : float
        Substrate doping concentration in cm^-3 (default: 1e16)
    substrate_type : str
        Substrate doping type: "p" or "n" (default: "p")
    oxide_permittivity : float
        Relative permittivity of oxide (default: 3.9)
    oxide_qf : float
        Fixed bulk charge density in oxide in cm^-3 (default: 0)
    oxide_qftrap : float
        Interface trap charge density at oxide-semiconductor interface in cm^-2
        (default: 0). Shifts the flat-band voltage.
    taun0 : float
        Electron minority carrier lifetime in seconds (default: 1e-9,
        matching the reference tool). Short lifetimes let the inversion
        layer respond during the sweep so the HF curve saturates at Cmin
        instead of drifting into deep depletion, and the LF curve reaches
        equilibrium inversion.
    taup0 : float
        Hole minority carrier lifetime in seconds (default: 1e-9).
    temperature : float
        Simulation temperature in Kelvin (default: 300)
    conmob : bool
        Enable concentration-dependent mobility (default: True)
    fldmob : bool
        Enable field-dependent mobility (default: True)
    gate_type : str
        Top gate contact type: "n_poly", "p_poly", "aluminum", "tungsten",
        or "metal" for custom workfunction (default: "n_poly").
    gate_workfunction : float, optional
        Custom gate workfunction in eV. Used when gate_type="metal".
        Standard values: aluminum=4.17, tungsten≈4.55.
    gate_config : str
        Gate configuration: "single" (one top gate + ohmic back contact) or
        "double" (gate-oxide-Si-oxide-gate stack). (default: "single")
    back_oxide_thickness : float
        Bottom oxide thickness in microns (double-gate only, default: 0.002).
    back_gate_type : str
        Bottom gate type: "n_poly", "p_poly", "aluminum", "tungsten", or
        "metal" (double-gate only, default: "n_poly").
    back_gate_workfunction : float, optional
        Custom bottom gate workfunction in eV (double-gate only).
    title : str, optional
        Simulation title
    log_cv : bool
        If True, add high-frequency AC C-V logging (default: False)
    cv_file : str
        Filename for high-frequency C-V data (default: "cv_data")
    log_cv_lf : bool
        If True, add a second low-frequency AC C-V solve (default: False)
    cv_lf_file : str
        Filename for low-frequency C-V data (default: "cv_lf_data")
    log_bands_eq : bool
        If True, log band diagrams at equilibrium (default: False)
    log_bands_bias : bool
        If True, log band diagrams at each bias point during sweep (default: False)
    log_qf_eq : bool
        If True, also log quasi-Fermi levels (Efn, Efp) at equilibrium
        alongside the band diagram. Requires log_bands_eq=True or is added
        automatically when True (default: False)
    log_qf_bias : bool
        If True, also log quasi-Fermi levels at each bias point (default: False)
    log_profiles_eq : bool
        If True, log carrier densities, potential, and electric field at
        equilibrium (default: False)
    log_profiles_bias : bool
        If True, log carrier densities, potential, and electric field at
        the last bias point of the sweep (default: False)
    vg_sweep : tuple (v_start, v_end, v_step), optional
        Gate voltage sweep for C-V characteristic with AC analysis.
        Example: (-2.0, 2.0, 0.2) sweeps from -2V to 2V
    ac_frequency : float
        High-frequency AC analysis frequency in Hz (default: 1e6 = 1 MHz)
    ac_frequency_lf : float
        Low-frequency AC analysis frequency in Hz (default: 1.0 = 1 Hz).
        Only used when log_cv_lf=True.

    Returns
    -------
    Simulation
        Configured MOS capacitor simulation

    Example
    -------
    >>> # Basic MOS capacitor - add your own solve commands
    >>> sim = create_mos_capacitor(oxide_thickness=0.005, substrate_doping=1e17)
    >>> sim.add_solve(Solve(initial=True))
    >>> print(sim.generate_deck())
    >>>
    >>> # C-V characteristic with HF and LF curves + profiles
    >>> sim = create_mos_capacitor(
    ...     log_cv=True, log_cv_lf=True,
    ...     log_profiles_eq=True, log_profiles_bias=True,
    ...     vg_sweep=(-2.0, 2.0, 0.1),
    ... )
    >>> result = sim.run()
    >>>
    >>> # Double-gate with custom workfunction
    >>> sim = create_mos_capacitor(
    ...     gate_config="double",
    ...     gate_type="metal", gate_workfunction=4.5,
    ...     back_gate_type="metal", back_gate_workfunction=4.5,
    ... )
    """
    is_double = gate_config.lower() == "double"
    if silicon_thickness is None:
        silicon_thickness = 0.03 if is_double else 5.0
    sim = Simulation(title=title or ("Double-Gate MOS Capacitor" if is_double else "MOS Capacitor"))
    sim._device_type = "mos_capacitor"
    sim._device_kwargs = dict(
        oxide_thickness=oxide_thickness, silicon_thickness=silicon_thickness,
        device_width=device_width, device_z_width=device_z_width,
        ny_oxide=ny_oxide, ny_silicon=ny_silicon,
        nx=nx, substrate_doping=substrate_doping, substrate_type=substrate_type,
        oxide_permittivity=oxide_permittivity, oxide_qf=oxide_qf,
        oxide_qftrap=oxide_qftrap, taun0=taun0, taup0=taup0,
        temperature=temperature, conmob=conmob, fldmob=fldmob,
        gate_type=gate_type, gate_workfunction=gate_workfunction,
        gate_config=gate_config,
        back_oxide_thickness=back_oxide_thickness, back_gate_type=back_gate_type,
        back_gate_workfunction=back_gate_workfunction,
        title=title, log_cv=log_cv, cv_file=cv_file,
        log_cv_lf=log_cv_lf, cv_lf_file=cv_lf_file,
        log_bands_eq=log_bands_eq, log_bands_bias=log_bands_bias,
        log_qf_eq=log_qf_eq, log_qf_bias=log_qf_bias,
        log_profiles_eq=log_profiles_eq, log_profiles_bias=log_profiles_bias,
        vg_sweep=vg_sweep, ac_frequency=ac_frequency,
        ac_frequency_lf=ac_frequency_lf,
    )

    # Warn when the silicon body cannot contain the depletion region:
    # the ohmic back contact would clip it and distort the C-V minimum.
    wd_max_um = max_depletion_width_um(substrate_doping, temperature)
    if not is_double and silicon_thickness < 2 * wd_max_um:
        warnings.warn(
            f"create_mos_capacitor: silicon_thickness={silicon_thickness} µm "
            f"is less than twice the maximum depletion width "
            f"(Wd,max ≈ {wd_max_um:.3f} µm at {substrate_doping:.1e} cm^-3). "
            f"The back contact will truncate the depletion region and "
            f"distort the C-V curve; increase silicon_thickness or the "
            f"doping.",
            UserWarning,
            stacklevel=2,
        )

    if is_double:
        # Double-gate mesh mirrors Rappture's 4-segment approach:
        #   top oxide → silicon (expanding) → silicon (compressing) → back oxide
        # This naturally densifies near both oxide-silicon interfaces.
        ny_back_oxide = ny_oxide
        ny_si_half = ny_silicon // 2
        total_ny = ny_oxide + ny_silicon + ny_back_oxide
        total_thickness = oxide_thickness + silicon_thickness + back_oxide_thickness
        ny_si_start = ny_oxide               # top oxide-silicon boundary
        ny_si_mid   = ny_oxide + ny_si_half  # silicon midplane
        ny_si_end   = ny_oxide + ny_silicon  # silicon-back oxide boundary

        sim.mesh = Mesh(nx=nx, ny=total_ny,
                        width=None if device_z_width == 1.0 else device_z_width)
        sim.mesh.add_y_mesh(1, 0, ratio=1)
        sim.mesh.add_y_mesh(ny_oxide, oxide_thickness, ratio=0.9)
        sim.mesh.add_y_mesh(ny_si_mid, oxide_thickness + silicon_thickness / 2, ratio=1.1)
        sim.mesh.add_y_mesh(ny_si_end, oxide_thickness + silicon_thickness, ratio=0.9)
        sim.mesh.add_y_mesh(total_ny, total_thickness, ratio=1.0)
    else:
        # Single-gate: matches Rappture's 5-point mesh exactly.
        # Silicon is split into a near-interface zone (ny_oxide nodes, 0.02 µm)
        # and a bulk zone (ny_silicon nodes), giving total = ny_oxide + ny_silicon nodes.
        # Cap the near-interface width at 10% of silicon thickness for thin bodies.
        near_interface_width = min(0.02, silicon_thickness * 0.1)
        # The near-interface zone resolves the inversion layer, so it gets a
        # share of the silicon nodes rather than being tied to the oxide
        # count.  Previously ny_near = ny_oxide = 100 put 100 lines across a
        # 2 nm oxide (0.02 nm cells) and, with RATIO=0.8 compounding over 50
        # intervals, drove the smallest cell to 3.6e-9 um -- 3.6 femtometres.
        ny_near = max(8, min(ny_silicon // 4, ny_silicon - 4))
        total_ny = ny_oxide + ny_silicon
        total_thickness = oxide_thickness + silicon_thickness
        ny_mid_oxide = max(1, ny_oxide // 2)
        near_end = oxide_thickness + near_interface_width

        h_ox = oxide_thickness / max(ny_oxide - 1, 1)
        h_near = near_interface_width / max(ny_near, 1)
        if h_ox < 1e-4:
            warnings.warn(
                f"create_mos_capacitor: ny_oxide={ny_oxide} puts {h_ox * 1e4:.3g} A "
                f"cells across a {oxide_thickness * 1e4:.1f} A oxide, below atomic "
                f"scale. Reduce ny_oxide.", UserWarning, stacklevel=2)

        sim.mesh = Mesh(nx=nx, ny=total_ny,
                        width=None if device_z_width == 1.0 else device_z_width)
        sim.mesh.add_y_mesh(1, 0)
        sim.mesh.add_y_mesh(ny_mid_oxide, oxide_thickness / 2, ratio=1)
        sim.mesh.add_y_mesh(ny_oxide, oxide_thickness, ratio=1)
        sim.mesh.add_y_mesh(ny_oxide + ny_near, near_end, ratio=1)
        sim.mesh.add_y_mesh(total_ny, total_thickness,
                            ratio=max(expand_ratio(
                                silicon_thickness - near_interface_width,
                                total_ny - ny_oxide - ny_near, h_near), 1.02))

    check_mesh_size(nx, total_ny, "create_mos_capacitor")

    sim.mesh.add_x_mesh(1, 0.001)
    sim.mesh.add_x_mesh(nx, device_width, ratio=1)

    # Regions
    if is_double:
        sim.add_region(Region(1, ix_low=1, ix_high=nx, iy_low=1, iy_high=ny_si_start,
                              material="sio2.1", insulator=True))
        sim.add_region(Region(2, ix_low=1, ix_high=nx, iy_low=ny_si_start, iy_high=ny_si_end,
                              material="silicon", semiconductor=True))
        sim.add_region(Region(3, ix_low=1, ix_high=nx, iy_low=ny_si_end, iy_high=total_ny,
                              material="sio2.2", insulator=True))
    else:
        sim.add_region(Region(1, ix_low=1, ix_high=nx, iy_low=1, iy_high=ny_oxide,
                              material="sio2", insulator=True))
        sim.add_region(Region(2, ix_low=1, ix_high=nx, iy_low=ny_oxide, iy_high=total_ny,
                              material="silicon", semiconductor=True))

    # Electrodes: electrode 1 = top gate, electrode 2 = back contact / bottom gate
    sim.add_electrode(Electrode(1, ix_low=1, ix_high=nx, iy_low=1, iy_high=1))       # Top gate
    sim.add_electrode(Electrode(2, ix_low=1, ix_high=nx, iy_low=total_ny, iy_high=total_ny))  # Back

    # Surface interface(s) between oxide and silicon — after electrodes, before doping
    sim.add_surface(Surface(number=1, interface=True, reg1=1, reg2=2,
                            x_min=0, x_max=device_width,
                            y_min=oxide_thickness, y_max=oxide_thickness))
    if is_double:
        back_interface_y = oxide_thickness + silicon_thickness
        sim.add_surface(Surface(number=2, interface=True, reg1=2, reg2=3,
                                x_min=0, x_max=device_width,
                                y_min=back_interface_y, y_max=back_interface_y))

    # Doping
    p_type = substrate_type.lower() == "p"
    si_region = 2
    sim.add_doping(Doping(region=si_region, p_type=p_type, n_type=not p_type,
                          concentration=substrate_doping, uniform=True))

    # Contacts — set all to neutral first, then override gate(s)
    sim.add_contact(Contact(all_contacts=True, neutral=True))

    def _gate_contact(number: int, gtype: str, gwf: Optional[float]) -> Contact:
        """Build a Contact for a gate electrode."""
        if gtype == "n_poly":
            return Contact(number=number, n_polysilicon=True)
        elif gtype == "p_poly":
            return Contact(number=number, p_polysilicon=True)
        elif gtype == "aluminum":
            return Contact(number=number, aluminum=True)
        elif gtype == "tungsten":
            return Contact(number=number, tungsten=True)
        else:  # "metal" or anything else — use explicit workfunction if given
            if gwf is not None:
                return Contact(number=number, workfunction=gwf)
            return Contact(number=number, neutral=True)  # fallback: neutral metal

    sim.add_contact(_gate_contact(1, gate_type, gate_workfunction))
    if is_double:
        sim.add_contact(_gate_contact(2, back_gate_type, back_gate_workfunction))

    # Materials — include carrier lifetimes
    sim.add_material(Material(name="silicon", taun0=taun0, taup0=taup0))
    if is_double:
        sim.add_material(Material(name="sio2.1", permittivity=oxide_permittivity, qf=oxide_qf))
        sim.add_material(Material(name="sio2.2", permittivity=oxide_permittivity, qf=oxide_qf))
    else:
        sim.add_material(Material(name="sio2", permittivity=oxide_permittivity, qf=oxide_qf))

    # Interface trap charge at oxide-semiconductor interface(s).
    # Always emit "interface num=N qf=..." — Rappture always includes this line.
    sim.add_interface(Interface(number=1, qf=oxide_qftrap))
    if is_double:
        sim.add_interface(Interface(number=2, qf=oxide_qftrap))

    # Models — matches Rappture reference deck (includes print flag)
    sim.models = Models(temperature=temperature, srh=True, conmob=conmob, fldmob=fldmob,
                        print_models=True)
    sim.system = System(electrons=True, holes=True, newton=True)

    # C-V logging (high frequency) — issued before the HF sweep
    if log_cv:
        sim.add_log(Log(acfile=cv_file))
    # Note: LF log is inserted later, after the HF sweep, when vg_sweep is built

    # Line cut for profiles: vertical through oxide and silicon at x = mid
    x_mid = device_width / 2

    needs_solve = (vg_sweep is not None or log_bands_eq or log_qf_eq
                   or log_profiles_eq or log_profiles_bias)

    if needs_solve:
        # Always start with equilibrium solve
        sim.add_solve(Solve(initial=True, outfile="eq"))
        n_prior = 1        # solutions computed so far
        v_gate = 0.0       # last solved gate bias

        # Equilibrium band diagram
        if log_bands_eq or log_qf_eq:
            sim.log_band_diagram(
                outfile_prefix="eq",
                x_start=x_mid, x_end=x_mid,
                y_start=0.0, y_end=total_thickness,
                include_qf=log_qf_eq,
            )

        # Equilibrium profiles: carriers, potential, E-field
        if log_profiles_eq:
            sim.log_carriers("eq", x_start=x_mid, x_end=x_mid,
                             y_start=0.0, y_end=total_thickness)
            sim.log_potential("pot_eq", x_start=x_mid, x_end=x_mid,
                              y_start=0.0, y_end=total_thickness)
            sim.log_efield("ef_eq", x_start=x_mid, x_end=x_mid,
                           y_start=0.0, y_end=total_thickness)

        # Gate voltage sweep — high-frequency C-V
        if vg_sweep is not None:
            v_start, v_end, v_step = vg_sweep
            nsteps, v_step, v_final = sweep_steps(
                v_start, v_end, v_step, "create_mos_capacitor vg_sweep")

            # Ramp the gate to the sweep starting voltage first — the
            # reference deck never jumps to the start bias in one solve.
            # Ramp solves carry no AC card, so they leave the AC log clean.
            if abs(v_start - v_gate) > 1e-10:
                n_prior = add_bias_ramp(sim, 1, v_gate, v_start, n_prior,
                                        max_step=GATE_STEP,
                                        outfile="vg_ramp")
                v_gate = v_start

            sim.add_solve(Solve(
                **solve_guess(n_prior),
                v1=v_start,
                vstep=v_step,
                nsteps=nsteps,
                electrode=1,
                ac_analysis=True,
                frequency=ac_frequency,
                outfile="cv",
                save=1 if (log_bands_bias or log_profiles_bias) else None,
            ))
            n_prior += nsteps + 1
            v_gate = v_final

            if log_bands_bias or log_qf_bias:
                sim.log_band_diagram(
                    outfile_prefix="bias",
                    x_start=x_mid, x_end=x_mid,
                    y_start=0.0, y_end=total_thickness,
                    include_qf=log_qf_bias,
                )

            if log_profiles_bias:
                sim.log_carriers("bias", x_start=x_mid, x_end=x_mid,
                                 y_start=0.0, y_end=total_thickness)
                sim.log_potential("pot_bias", x_start=x_mid, x_end=x_mid,
                                  y_start=0.0, y_end=total_thickness)
                sim.log_efield("ef_bias", x_start=x_mid, x_end=x_mid,
                               y_start=0.0, y_end=total_thickness)

            # Low-frequency C-V: ramp the gate back to the sweep start
            # (the HF sweep left it at v_end — never jump the full range
            # in one solve), redirect the AC log, then repeat the sweep
            # at the low frequency.
            if log_cv_lf:
                if abs(v_start - v_gate) > 1e-10:
                    n_prior = add_bias_ramp(sim, 1, v_gate, v_start, n_prior,
                                            max_step=GATE_STEP,
                                            outfile="vg_ramp_lf")
                    v_gate = v_start
                sim.add_log(Log(acfile=cv_lf_file))
                sim.add_solve(Solve(
                    **solve_guess(n_prior),
                    v1=v_start,
                    vstep=v_step,
                    nsteps=nsteps,
                    electrode=1,
                    ac_analysis=True,
                    frequency=ac_frequency_lf,
                    outfile="cv_lf",
                ))
                n_prior += nsteps + 1
                v_gate = v_final

    return sim


# Alias
mos_capacitor = create_mos_capacitor
