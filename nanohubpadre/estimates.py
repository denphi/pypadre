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


def pn_depletion_width_um(p_doping: float, n_doping: float,
                          reverse_bias: float = 0.0,
                          temperature: float = 300.0,
                          ni: float = NI_SI) -> float:
    """
    Zero- or reverse-bias depletion width of an abrupt PN junction, in microns.

    W = sqrt( 2·εs·(Vbi + Vr)/q · (1/Na + 1/Nd) ), with
    Vbi = kT/q · ln(Na·Nd / ni²).

    Used by the device factories to size mesh refinement around a junction:
    the grid has to resolve this width, and grading it by a fixed ratio
    without reference to it is how meshes end up with femtometre cells.

    Parameters
    ----------
    p_doping, n_doping : float
        Acceptor and donor concentrations in cm^-3.
    reverse_bias : float
        Reverse bias magnitude in volts (default 0 = equilibrium).
    temperature : float
        Temperature in Kelvin (default 300).
    ni : float
        Intrinsic carrier concentration in cm^-3.

    Returns
    -------
    float
        Depletion width in microns.
    """
    kt_q = 8.617e-5 * temperature
    na = max(p_doping, ni * 10)
    nd = max(n_doping, ni * 10)
    vbi = kt_q * math.log(na * nd / (ni * ni))
    v = max(vbi + max(reverse_bias, 0.0), kt_q)
    w_cm = math.sqrt(2 * EPS_SI * v / Q * (1.0 / na + 1.0 / nd))
    return w_cm * 1e4


def debye_length_um(doping: float, temperature: float = 300.0) -> float:
    """
    Extrinsic Debye length LD = sqrt(εs·kT/q / (q·N)) in microns.

    This is the screening length that sets how far the built-in step at an
    isotype (n+/n-, p+/p-) interface spreads into the lightly doped side.
    Unlike a PN junction there is no depletion approximation to fall back
    on, so device factories for NIN/PIP structures size their mesh
    refinement from this instead.

    Parameters
    ----------
    doping : float
        Net doping of the lightly doped side in cm^-3.
    temperature : float
        Temperature in Kelvin (default 300).

    Returns
    -------
    float
        Debye length in microns.
    """
    kt_q = 8.617e-5 * temperature
    n = max(doping, NI_SI)
    return math.sqrt(EPS_SI * kt_q / (Q * n)) * 1e4
