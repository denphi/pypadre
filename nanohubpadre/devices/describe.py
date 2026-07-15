"""
Device parameter description system.

Provides structured parameter documentation for all device factory functions,
with formatted output for both terminal and Jupyter environments.
"""

from collections import OrderedDict


# Parameter registry: device_name -> list of (group, params)
# Each param is (name, type_str, default, unit, description)
DEVICE_PARAMS = {
    "pn_diode": {
        "summary": "PN Junction Diode",
        "factory": "create_pn_diode",
        "groups": OrderedDict([
            ("Geometry", [
                ("length", "float", 1.0, "um", "Total device length"),
                ("width", "float", 1.0, "um", "Device width (y extent)"),
                ("device_z_width", "float", 1.0, "um", "Depth in z; terminal currents scale with it"),
                ("junction_position", "float", 0.5, "", "Junction position as fraction of length (0-1)"),
            ]),
            ("Mesh", [
                ("nx", "int", 200, "", "Mesh points in x direction"),
                ("ny", "int", 3, "", "Mesh points in y direction"),
            ]),
            ("Doping", [
                ("p_doping", "float", 1e17, "cm^-3", "P-type doping concentration"),
                ("n_doping", "float", 1e17, "cm^-3", "N-type doping concentration"),
            ]),
            ("Physical Models", [
                ("temperature", "float", 300, "K", "Simulation temperature"),
                ("srh", "bool", True, "", "Shockley-Read-Hall recombination"),
                ("conmob", "bool", True, "", "Concentration-dependent mobility"),
                ("fldmob", "bool", True, "", "Field-dependent mobility"),
                ("impact", "bool", False, "", "Impact ionization"),
            ]),
            ("Material", [
                ("taun0", "float", 1e-6, "s", "Electron lifetime"),
                ("taup0", "float", 1e-6, "s", "Hole lifetime"),
            ]),
            ("Simulation Options", [
                ("title", "str", None, "", "Simulation title"),
                ("postscript", "bool", False, "", "Enable PostScript output"),
            ]),
            ("Output Logging", [
                ("log_iv", "bool", False, "", "Enable I-V logging"),
                ("iv_file", "str", "iv", "", "I-V log filename"),
                ("log_bands_eq", "bool", False, "", "Log band diagrams at equilibrium"),
                ("log_bands_bias", "bool", False, "", "Log band diagrams during sweeps"),
            ]),
            ("Voltage Sweep", [
                ("forward_sweep", "tuple", None, "V", "Forward bias sweep (v_start, v_end, v_step)"),
                ("reverse_sweep", "tuple", None, "V", "Reverse bias sweep (v_start, v_end, v_step)"),
                ("sweep_electrode", "int", 1, "", "Electrode for voltage sweeps (1 = P side/anode)"),
                ("log_physics_at", "list", None, "V", "Bias points for physics profiles (must start at 0.0)"),
            ]),
        ]),
    },
    "mosfet": {
        "summary": "Metal-Oxide-Semiconductor FET",
        "factory": "create_mosfet",
        "groups": OrderedDict([
            ("Geometry", [
                ("channel_length", "float", 0.15, "um", "Gate/channel length"),
                ("gate_oxide_thickness", "float", 0.002, "um", "Gate oxide thickness"),
                ("junction_depth", "float", 0.02, "um", "Source/drain junction depth"),
                ("device_width", "float", 0.25, "um", "Total device width"),
                ("device_depth", "float", 0.05, "um", "Substrate depth"),
                ("device_z_width", "float", 1.0, "um", "Depth in z; terminal currents scale with it"),
            ]),
            ("Mesh", [
                ("nx", "int", 53, "", "Mesh points in x direction"),
                ("ny", "int", 46, "", "Mesh points in y direction"),
            ]),
            ("Doping", [
                ("channel_doping", "float", 1e18, "cm^-3", "Channel doping concentration"),
                ("substrate_doping", "float", 5e16, "cm^-3", "Substrate doping concentration"),
                ("source_drain_doping", "float", 2e20, "cm^-3", "Source/drain doping concentration"),
                ("device_type", "str", "nmos", "", "Device type: 'nmos' or 'pmos'"),
            ]),
            ("Physical Models", [
                ("temperature", "float", 300, "K", "Simulation temperature"),
                ("bgn", "bool", True, "", "Band-gap narrowing"),
                ("conmob", "bool", True, "", "Concentration-dependent mobility"),
                ("fldmob", "bool", True, "", "Field-dependent mobility (velocity saturation)"),
                ("gatmob", "bool", True, "", "Gate-field mobility degradation"),
                ("carriers", "int", 1, "", "Number of carriers to solve (1 or 2)"),
            ]),
            ("Simulation Options", [
                ("title", "str", None, "", "Simulation title"),
            ]),
            ("Output Logging", [
                ("log_iv", "bool", False, "", "Enable I-V logging"),
                ("iv_file", "str", "idvg", "", "I-V log filename"),
                ("log_bands_eq", "bool", False, "", "Log band diagrams at equilibrium"),
                ("log_bands_bias", "bool", False, "", "Log band diagrams during sweeps"),
            ]),
            ("Voltage Sweep", [
                ("vgs_sweep", "tuple", None, "V", "Gate voltage sweep for Id-Vg (v_start, v_end, v_step)"),
                ("vds", "float", 0.0, "V", "Fixed drain voltage during transfer characteristic"),
                ("vds_sweep", "tuple", None, "V", "Drain voltage sweep for Id-Vd (v_start, v_end, v_step)"),
                ("vgs", "float", 0.0, "V", "Fixed gate voltage during output characteristic"),
            ]),
        ]),
    },
    "bjt": {
        "summary": "Bipolar Junction Transistor",
        "factory": "create_bjt",
        "groups": OrderedDict([
            ("Geometry", [
                ("emitter_width", "float", 1.0, "um", "Emitter region width"),
                ("base_width", "float", 0.5, "um", "Base region width"),
                ("collector_width", "float", 2.0, "um", "Collector region width"),
                ("device_depth", "float", 1.0, "um", "Device depth"),
                ("device_z_width", "float", 1.0, "um", "Depth in z; terminal currents scale with it"),
            ]),
            ("Mesh", [
                ("nx", "int", 60, "", "Mesh points in x direction"),
                ("ny", "int", 30, "", "Mesh points in y direction"),
            ]),
            ("Doping", [
                ("emitter_doping", "float", 1e20, "cm^-3", "Emitter doping concentration"),
                ("base_doping", "float", 1e17, "cm^-3", "Base doping concentration"),
                ("collector_doping", "float", 1e16, "cm^-3", "Collector doping concentration"),
                ("device_type", "str", "npn", "", "Device type: 'npn' or 'pnp'"),
                ("emitter_tau", "float", 1e-7, "s", "Emitter minority-carrier lifetime"),
                ("base_tau", "float", 1e-6, "s", "Base minority-carrier lifetime"),
                ("collector_tau", "float", 1e-7, "s", "Collector minority-carrier lifetime"),
            ]),
            ("Physical Models", [
                ("temperature", "float", 300, "K", "Simulation temperature"),
                ("srh", "bool", True, "", "Shockley-Read-Hall recombination"),
                ("auger", "bool", True, "", "Auger recombination"),
                ("bgn", "bool", True, "", "Band-gap narrowing"),
                ("conmob", "bool", True, "", "Concentration-dependent mobility"),
                ("fldmob", "bool", True, "", "Field-dependent mobility"),
                ("surf_rec_contacts", "bool", True, "", "Reference-style surface-recombination contacts"),
            ]),
            ("Simulation Options", [
                ("title", "str", None, "", "Simulation title"),
            ]),
            ("Output Logging", [
                ("log_iv", "bool", False, "", "Enable I-V logging"),
                ("iv_file", "str", "ic_vce.log", "", "I-V log filename"),
                ("log_bands_eq", "bool", False, "", "Log band diagrams at equilibrium"),
            ]),
            ("Voltage Sweep", [
                ("vbe", "float", 0.0, "V", "Base-emitter voltage for output characteristics"),
                ("vce_sweep", "tuple", None, "V", "Collector-emitter sweep for Ic-Vce (v_start, v_end, v_step)"),
                ("gummel_sweep", "tuple", None, "V", "Base-emitter sweep for Gummel plot (v_start, v_end, v_step)"),
                ("gummel_vce", "float", 2.0, "V", "Fixed Vce for Gummel plot"),
            ]),
        ]),
    },
    "mesfet": {
        "summary": "Metal-Semiconductor FET",
        "factory": "create_mesfet",
        "groups": OrderedDict([
            ("Geometry", [
                ("channel_length", "float", 0.2, "um", "Source-gate and gate-drain spacing"),
                ("gate_length", "float", 0.2, "um", "Gate length"),
                ("device_width", "float", 0.6, "um", "Total device width"),
                ("channel_depth", "float", 0.2, "um", "Channel depth"),
                ("substrate_depth", "float", 0.8, "um", "Substrate depth below channel"),
                ("device_z_width", "float", 1.0, "um", "Depth in z; terminal currents scale with it"),
            ]),
            ("Mesh", [
                ("nx", "int", 55, "", "Mesh points in x direction"),
                ("ny", "int", 43, "", "Mesh points in y direction"),
            ]),
            ("Doping", [
                ("channel_doping", "float", 1e17, "cm^-3", "Channel doping concentration"),
                ("substrate_doping", "float", 1e10, "cm^-3", "Substrate doping (semi-insulating default)"),
                ("substrate_type", "str", "same", "", "Substrate type vs channel: 'same' or 'opposite'"),
                ("contact_doping", "float", 1e20, "cm^-3", "Source/drain contact doping"),
                ("device_type", "str", "n", "", "Channel type: 'n' or 'p'"),
            ]),
            ("Contact", [
                ("gate_workfunction", "float", 4.87, "V", "Gate metal workfunction (Schottky barrier)"),
            ]),
            ("Physical Models", [
                ("temperature", "float", 300, "K", "Simulation temperature"),
                ("bgn", "bool", True, "", "Band-gap narrowing"),
                ("conmob", "bool", True, "", "Concentration-dependent mobility"),
                ("fldmob", "bool", True, "", "Field-dependent mobility"),
                ("statistics", "str", "fermi", "", "Carrier statistics: 'fermi' or 'boltzmann'"),
            ]),
            ("Simulation Options", [
                ("title", "str", None, "", "Simulation title"),
            ]),
            ("Output Logging", [
                ("log_iv", "bool", False, "", "Enable I-V logging"),
                ("iv_file", "str", "idvd", "", "I-V log filename"),
                ("log_bands_eq", "bool", False, "", "Log band diagrams at equilibrium"),
            ]),
            ("Voltage Sweep", [
                ("vgs", "float", 0.0, "V", "Gate-source voltage for output characteristic"),
                ("vds_sweep", "tuple", None, "V", "Drain-source sweep for Id-Vds (v_start, v_end, v_step)"),
            ]),
        ]),
    },
    "mos_capacitor": {
        "summary": "MOS Capacitor",
        "factory": "create_mos_capacitor",
        "groups": OrderedDict([
            ("Geometry", [
                ("oxide_thickness", "float", 0.002, "um", "Gate oxide thickness"),
                ("silicon_thickness", "float", 5.0, "um", "Silicon thickness (default 5.0 single-gate, 0.03 double-gate)"),
                ("device_z_width", "float", 1.0, "um", "Depth in z; capacitances scale with it"),
                ("device_width", "float", 1.0, "um", "Device width"),
            ]),
            ("Mesh", [
                ("ny_oxide", "int", 100, "", "Mesh points in oxide layer"),
                ("ny_silicon", "int", 200, "", "Mesh points in silicon"),
                ("nx", "int", 3, "", "Mesh points in x direction"),
            ]),
            ("Doping", [
                ("substrate_doping", "float", 1e16, "cm^-3", "Substrate doping concentration"),
                ("substrate_type", "str", "p", "", "Substrate doping type: 'p' or 'n'"),
            ]),
            ("Material", [
                ("oxide_permittivity", "float", 3.9, "", "Relative permittivity of oxide"),
                ("oxide_qf", "float", 0, "cm^-3", "Fixed bulk charge density in oxide"),
                ("oxide_qftrap", "float", 0, "cm^-2", "Interface trap charge density at oxide-Si interface"),
                ("taun0", "float", 1e-9, "s", "Electron minority carrier lifetime"),
                ("taup0", "float", 1e-9, "s", "Hole minority carrier lifetime"),
            ]),
            ("Contact", [
                ("gate_type", "str", "n_poly", "", "Top gate type: 'n_poly', 'p_poly', 'aluminum', 'tungsten', or 'metal'"),
                ("gate_workfunction", "float", None, "eV", "Custom top gate workfunction (used when gate_type='metal')"),
                ("gate_config", "str", "single", "", "Gate configuration: 'single' or 'double'"),
                ("back_oxide_thickness", "float", 0.002, "um", "Bottom oxide thickness (double-gate only)"),
                ("back_gate_type", "str", "n_poly", "", "Bottom gate type: 'n_poly', 'p_poly', 'aluminum', 'tungsten', or 'metal' (double-gate only)"),
                ("back_gate_workfunction", "float", None, "eV", "Custom bottom gate workfunction (double-gate + gate_type='metal' only)"),
            ]),
            ("Physical Models", [
                ("temperature", "float", 300, "K", "Simulation temperature"),
                ("conmob", "bool", True, "", "Concentration-dependent mobility"),
                ("fldmob", "bool", True, "", "Field-dependent mobility"),
            ]),
            ("Simulation Options", [
                ("title", "str", None, "", "Simulation title"),
            ]),
            ("Output Logging", [
                ("log_cv", "bool", False, "", "Enable high-frequency AC C-V logging"),
                ("cv_file", "str", "cv_data", "", "High-frequency C-V data filename"),
                ("log_cv_lf", "bool", False, "", "Enable low-frequency AC C-V sweep"),
                ("cv_lf_file", "str", "cv_lf_data", "", "Low-frequency C-V data filename"),
                ("log_bands_eq", "bool", False, "", "Log band diagrams (Ec, Ev) at equilibrium"),
                ("log_bands_bias", "bool", False, "", "Log band diagrams during sweeps"),
                ("log_qf_eq", "bool", False, "", "Also log quasi-Fermi levels (Efn, Efp) at equilibrium"),
                ("log_qf_bias", "bool", False, "", "Also log quasi-Fermi levels during sweeps"),
                ("log_profiles_eq", "bool", False, "", "Log carrier densities, potential, E-field at equilibrium"),
                ("log_profiles_bias", "bool", False, "", "Log carrier densities, potential, E-field at last bias"),
            ]),
            ("Voltage Sweep", [
                ("vg_sweep", "tuple", None, "V", "Gate voltage sweep for C-V (v_start, v_end, v_step)"),
                ("ac_frequency", "float", 1e6, "Hz", "High-frequency AC analysis frequency"),
                ("ac_frequency_lf", "float", 1.0, "Hz", "Low-frequency AC analysis frequency"),
            ]),
        ]),
    },
    "schottky_diode": {
        "summary": "Schottky Barrier Diode",
        "factory": "create_schottky_diode",
        "groups": OrderedDict([
            ("Geometry", [
                ("length", "float", 2.0, "um", "Device length"),
                ("width", "float", 1.0, "um", "Device width (y extent)"),
                ("device_z_width", "float", 1.0, "um", "Depth in z; terminal currents scale with it"),
            ]),
            ("Mesh", [
                ("nx", "int", 100, "", "Mesh points in x direction"),
                ("ny", "int", 20, "", "Mesh points in y direction"),
            ]),
            ("Doping", [
                ("doping", "float", 1e16, "cm^-3", "Semiconductor doping concentration"),
                ("doping_type", "str", "n", "", "Semiconductor doping type: 'n' or 'p'"),
            ]),
            ("Contact", [
                ("workfunction", "float", 4.8, "V", "Metal workfunction"),
                ("barrier_lowering", "bool", False, "", "Image-force barrier lowering"),
                ("surf_rec", "bool", True, "", "Thermionic-emission boundary (finite surface recombination)"),
                ("vsurfn", "float", 2.2e6, "cm/s", "Electron thermionic surface recombination velocity"),
                ("vsurfp", "float", 1.9e6, "cm/s", "Hole thermionic surface recombination velocity"),
            ]),
            ("Physical Models", [
                ("temperature", "float", 300, "K", "Simulation temperature"),
                ("srh", "bool", False, "", "Shockley-Read-Hall recombination"),
                ("conmob", "bool", True, "", "Concentration-dependent mobility"),
                ("fldmob", "bool", True, "", "Field-dependent mobility"),
            ]),
            ("Simulation Options", [
                ("title", "str", None, "", "Simulation title"),
            ]),
            ("Output Logging", [
                ("log_iv", "bool", False, "", "Enable I-V logging"),
                ("iv_file", "str", "iv", "", "I-V log filename"),
                ("log_bands_eq", "bool", False, "", "Log band diagrams at equilibrium"),
                ("log_bands_bias", "bool", False, "", "Log band diagrams during sweeps"),
            ]),
            ("Voltage Sweep", [
                ("forward_sweep", "tuple", None, "V", "Forward bias sweep (v_start, v_end, v_step)"),
                ("reverse_sweep", "tuple", None, "V", "Reverse bias sweep (v_start, v_end, v_step)"),
            ]),
        ]),
    },
    "solar_cell": {
        "summary": "Solar Cell (PN Junction) [EXPERIMENTAL — IV output unreliable, see warnings]",
        "factory": "create_solar_cell",
        "groups": OrderedDict([
            ("Geometry", [
                ("emitter_depth", "float", 0.5, "um", "Emitter junction depth"),
                ("base_thickness", "float", 200.0, "um", "Base (substrate) thickness"),
                ("device_width", "float", 1.0, "um", "Device width"),
                ("device_z_width", "float", 1.0, "um", "Depth in z; terminal currents scale with it"),
            ]),
            ("Mesh", [
                ("nx", "int", 3, "", "Mesh points in x direction"),
                ("ny", "int", 100, "", "Mesh points in y direction"),
            ]),
            ("Doping", [
                ("emitter_doping", "float", 1e19, "cm^-3", "Emitter doping (Gaussian profile)"),
                ("base_doping", "float", 1e16, "cm^-3", "Base doping (uniform)"),
                ("device_type", "str", "n_on_p", "", "Structure: 'n_on_p' or 'p_on_n'"),
            ]),
            ("Physical Models", [
                ("temperature", "float", 300, "K", "Simulation temperature"),
                ("srh", "bool", True, "", "Shockley-Read-Hall recombination"),
                ("auger", "bool", True, "", "Auger recombination"),
                ("conmob", "bool", True, "", "Concentration-dependent mobility"),
                ("fldmob", "bool", True, "", "Field-dependent mobility"),
            ]),
            ("Material", [
                ("taun0", "float", 1e-5, "s", "Electron lifetime"),
                ("taup0", "float", 1e-5, "s", "Hole lifetime"),
            ]),
            ("Surface Recombination", [
                ("front_surface_velocity", "float", 1e4, "cm/s", "Front surface recombination velocity"),
                ("back_surface_velocity", "float", 1e7, "cm/s", "Back surface recombination velocity"),
            ]),
            ("Simulation Options", [
                ("title", "str", None, "", "Simulation title"),
            ]),
            ("Output Logging", [
                ("log_iv", "bool", False, "", "Enable I-V logging"),
                ("iv_file", "str", "iv_dark", "", "I-V log filename"),
                ("log_bands_eq", "bool", False, "", "Log band diagrams at equilibrium"),
            ]),
            ("Voltage Sweep", [
                ("forward_sweep", "tuple", None, "V", "Forward voltage sweep (v_start, v_end, v_step)"),
                ("sweep_electrode", "int", 1, "", "Electrode for voltage sweep (1 = front)"),
            ]),
        ]),
    },
}


def _format_default(val):
    """Format a default value for display."""
    if val is None:
        return "None"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, float):
        if val == 0.0:
            return "0.0"
        if abs(val) >= 1e4 or abs(val) < 0.01:
            return f"{val:.0e}"
        return str(val)
    return str(val)


def _is_jupyter():
    """Check if running in a Jupyter environment."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except ImportError:
        return False


def _format_text(device_name, info):
    """Format parameter description as plain text."""
    lines = []
    lines.append(f"  {info['summary']}")
    lines.append(f"  Factory: {info['factory']}()")
    lines.append("")

    for group_name, params in info["groups"].items():
        lines.append(f"  {group_name}")
        lines.append(f"  {'-' * len(group_name)}")
        for name, type_str, default, unit, desc in params:
            default_str = _format_default(default)
            unit_str = f" [{unit}]" if unit else ""
            lines.append(f"    {name:30s} {type_str:6s}  {default_str:>10s}{unit_str}")
            lines.append(f"    {'':30s} {desc}")
        lines.append("")

    return "\n".join(lines)


def _format_html(device_name, info):
    """Format parameter description as HTML for Jupyter."""
    html = []
    html.append(f"<h3>{info['summary']}</h3>")
    html.append(f"<p>Factory: <code>{info['factory']}()</code></p>")

    for group_name, params in info["groups"].items():
        html.append(f"<h4>{group_name}</h4>")
        html.append("<table style='border-collapse: collapse; width: 100%; font-size: 13px;'>")
        html.append("<tr style='background: #f0f0f0; border-bottom: 2px solid #ccc;'>")
        html.append("<th style='text-align: left; padding: 4px 8px;'>Parameter</th>")
        html.append("<th style='text-align: left; padding: 4px 8px;'>Type</th>")
        html.append("<th style='text-align: left; padding: 4px 8px;'>Default</th>")
        html.append("<th style='text-align: left; padding: 4px 8px;'>Unit</th>")
        html.append("<th style='text-align: left; padding: 4px 8px;'>Description</th>")
        html.append("</tr>")

        for i, (name, type_str, default, unit, desc) in enumerate(params):
            bg = "#ffffff" if i % 2 == 0 else "#f9f9f9"
            default_str = _format_default(default)
            html.append(f"<tr style='background: {bg}; border-bottom: 1px solid #eee;'>")
            html.append(f"<td style='padding: 4px 8px;'><code>{name}</code></td>")
            html.append(f"<td style='padding: 4px 8px;'>{type_str}</td>")
            html.append(f"<td style='padding: 4px 8px;'><code>{default_str}</code></td>")
            html.append(f"<td style='padding: 4px 8px;'>{unit}</td>")
            html.append(f"<td style='padding: 4px 8px;'>{desc}</td>")
            html.append("</tr>")

        html.append("</table>")

    return "\n".join(html)


def list_devices():
    """Return list of available device names.

    Returns
    -------
    list of str
        Device names that can be passed to describe() or device_schematic()
    """
    return list(DEVICE_PARAMS.keys())


def describe(device_name=None):
    """Show parameters and descriptions for a device factory function.

    Parameters
    ----------
    device_name : str, optional
        Device name (e.g., 'pn_diode', 'mosfet'). If None, lists all
        available devices.

    Returns
    -------
    In Jupyter: displays an HTML table and returns None
    In terminal: prints formatted text and returns None

    Examples
    --------
    >>> describe()  # list all devices
    >>> describe('pn_diode')  # show PN diode parameters
    >>> describe('mosfet')  # show MOSFET parameters
    """
    if device_name is None:
        # List all devices
        lines = ["Available devices:"]
        lines.append("")
        for name, info in DEVICE_PARAMS.items():
            lines.append(f"  {name:20s} {info['summary']:30s} {info['factory']}()")
        lines.append("")
        lines.append("Use describe('<device_name>') for parameter details.")

        if _is_jupyter():
            from IPython.display import display, HTML
            html = ["<h3>Available Devices</h3>"]
            html.append("<table style='border-collapse: collapse; width: 100%; font-size: 13px;'>")
            html.append("<tr style='background: #f0f0f0; border-bottom: 2px solid #ccc;'>")
            html.append("<th style='text-align: left; padding: 4px 8px;'>Name</th>")
            html.append("<th style='text-align: left; padding: 4px 8px;'>Description</th>")
            html.append("<th style='text-align: left; padding: 4px 8px;'>Factory Function</th>")
            html.append("</tr>")
            for i, (name, info) in enumerate(DEVICE_PARAMS.items()):
                bg = "#ffffff" if i % 2 == 0 else "#f9f9f9"
                html.append(f"<tr style='background: {bg}; border-bottom: 1px solid #eee;'>")
                html.append(f"<td style='padding: 4px 8px;'><code>{name}</code></td>")
                html.append(f"<td style='padding: 4px 8px;'>{info['summary']}</td>")
                html.append(f"<td style='padding: 4px 8px;'><code>{info['factory']}()</code></td>")
                html.append("</tr>")
            html.append("</table>")
            html.append("<p>Use <code>describe('device_name')</code> for parameter details.</p>")
            display(HTML("\n".join(html)))
        else:
            print("\n".join(lines))
        return

    if device_name not in DEVICE_PARAMS:
        available = ", ".join(DEVICE_PARAMS.keys())
        raise ValueError(f"Unknown device '{device_name}'. Available: {available}")

    info = DEVICE_PARAMS[device_name]

    if _is_jupyter():
        from IPython.display import display, HTML
        display(HTML(_format_html(device_name, info)))
    else:
        print(_format_text(device_name, info))
