"""
Analytic first-order device estimates.

These textbook formulas help pick sensible sweep ranges and geometry
before running PADRE (e.g. "will this MOSFET turn on inside my gate
sweep?", "is my silicon body thick enough for the depletion region?").
The device factories use them internally for sanity warnings.
"""

import math

# Physical constants (silicon, near 300 K)
Q = 1.602e-19            # C
EPS_SI = 1.04e-12        # F/cm
EPS_OX = 3.45e-13        # F/cm (SiO2)
NI_SI = 1.5e10           # cm^-3


def estimate_mosfet_vt(channel_doping: float, gate_oxide_thickness: float,
                       temperature: float = 300.0,
                       is_nmos: bool = True) -> float:
    """
    Textbook long-channel threshold voltage for a poly-gate MOSFET with a
    uniform body (n+ poly on p-body for NMOS, p+ poly on n-body for PMOS).

    Parameters
    ----------
    channel_doping : float
        Body doping in cm^-3.
    gate_oxide_thickness : float
        Oxide thickness in microns.
    temperature : float
        Temperature in Kelvin (default 300).
    is_nmos : bool
        True for NMOS (positive Vt), False for PMOS (negative Vt).

    Returns
    -------
    float
        Estimated threshold voltage in volts.
    """
    kt_q = 8.617e-5 * temperature
    na = max(channel_doping, NI_SI * 10)
    phi_f = kt_q * math.log(na / NI_SI)
    cox = EPS_OX / (gate_oxide_thickness * 1e-4)  # F/cm^2
    q_dep = math.sqrt(2 * Q * EPS_SI * na * 2 * phi_f)
    half_gap = 0.56
    if is_nmos:
        vfb = -(half_gap + phi_f)   # n+ poly gate over p-type body
        return vfb + 2 * phi_f + q_dep / cox
    vfb = half_gap + phi_f          # p+ poly gate over n-type body
    return vfb - 2 * phi_f - q_dep / cox


def max_depletion_width_um(doping: float, temperature: float = 300.0) -> float:
    """
    Maximum depletion width Wd,max = sqrt(4·εs·φF / (q·N)) in microns for
    a uniformly doped silicon body (strong-inversion limit).

    Parameters
    ----------
    doping : float
        Body doping in cm^-3.
    temperature : float
        Temperature in Kelvin (default 300).

    Returns
    -------
    float
        Maximum depletion width in microns.
    """
    kt_q = 8.617e-5 * temperature
    n = max(doping, NI_SI * 10)
    phi_f = kt_q * math.log(n / NI_SI)
    wd_cm = math.sqrt(4 * EPS_SI * phi_f / (Q * n))
    return wd_cm * 1e4
