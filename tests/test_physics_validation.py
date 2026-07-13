"""
Physics validation tests — run PADRE and check the numbers.

These tests are skipped automatically when no ``padre`` executable is on
PATH (e.g. on developer machines); they run on nanoHUB or wherever PADRE
is installed.  Tolerances are generous: they catch broken physics (wrong
models, non-turning-on devices, clipped depletion regions), not small
numerical drift.

All PADRE terminal quantities are per micron of device depth (mesh
width=1), which cancels out of the ratios and slopes tested here except
where noted.
"""

import math
import os
import shutil

import numpy as np
import pytest

from nanohubpadre.devices import (
    create_bjt,
    create_mos_capacitor,
    create_mosfet,
    create_pn_diode,
)
from nanohubpadre.parser import parse_ac_file, parse_iv_file

PADRE = shutil.which("padre")

pytestmark = pytest.mark.skipif(
    PADRE is None, reason="PADRE executable not available"
)

KT_Q = 0.02585  # V at 300 K


def _run(sim):
    result = sim.run()
    assert result.returncode == 0, f"PADRE failed: {result.stderr}"
    return sim


def _iv_path(sim, name):
    return os.path.join(sim.working_dir, name)


class TestPnDiodePhysics:
    def test_forward_ideality_factor(self):
        """ln(I) vs V slope in mid-forward bias gives 1 <= n <= 2."""
        sim = _run(create_pn_diode(
            log_iv=True, forward_sweep=(0.0, 0.6, 0.025),
        ))
        iv = parse_iv_file(_iv_path(sim, "iv"))
        v = iv.get_voltages(2)
        i = np.abs(iv.get_currents(2))
        # mid-bias window: above SRH-dominated region, below series resistance
        mask = (v >= 0.3) & (v <= 0.5) & (i > 0)
        assert mask.sum() >= 4, "not enough valid bias points"
        slope = np.polyfit(v[mask], np.log(i[mask]), 1)[0]
        n = 1.0 / (slope * KT_Q)
        assert 0.8 < n < 2.2, f"ideality factor n={n:.2f} out of range"

    def test_rectification(self):
        """Forward current at 0.6 V must exceed reverse at -1 V by >> 1e3."""
        sim = _run(create_pn_diode(
            log_iv=True,
            forward_sweep=(0.0, 0.6, 0.05),
            reverse_sweep=(0.0, -1.0, -0.1),
        ))
        iv = parse_iv_file(_iv_path(sim, "iv"))
        v = iv.get_voltages(2)
        i = np.abs(iv.get_currents(2))
        i_fwd = i[np.argmin(np.abs(v - 0.6))]
        i_rev = max(i[np.argmin(np.abs(v + 1.0))], 1e-25)
        assert i_fwd / i_rev > 1e3


class TestMosfetPhysics:
    def test_default_device_turns_on(self):
        """Id-Vg must show >1e3 on/off ratio and a sub-volt Vt."""
        sim = _run(create_mosfet(
            log_iv=True, vgs_sweep=(0.0, 1.5, 0.1), vds=0.1,
        ))
        iv = parse_iv_file(_iv_path(sim, "idvg"))
        vg = iv.get_voltages(3)
        idr = np.abs(iv.get_currents(2))
        i_on = idr[np.argmin(np.abs(vg - 1.5))]
        i_off = max(idr[np.argmin(np.abs(vg - 0.0))], 1e-25)
        assert i_on > 1e-8, f"on-current {i_on:.3e} A/um is noise-level"
        assert i_on / i_off > 1e3, "no gate control (device never turns on)"

    def test_drain_current_saturates(self):
        """With fldmob, Id at high Vd must flatten (finite output cond.)."""
        sim = _run(create_mosfet(
            log_iv=True, vds_sweep=(0.0, 1.5, 0.05), vgs=1.2,
        ))
        iv = parse_iv_file(_iv_path(sim, "idvd"))
        vd = iv.get_voltages(2)
        idr = np.abs(iv.get_currents(2))
        order = np.argsort(vd)
        vd, idr = vd[order], idr[order]
        # slope in the last third (saturation) must be far below the
        # slope in the first third (linear region)
        third = len(vd) // 3
        g_lin = np.polyfit(vd[:third], idr[:third], 1)[0]
        g_sat = np.polyfit(vd[-third:], idr[-third:], 1)[0]
        assert g_sat < 0.3 * g_lin, "no drain-current saturation"


class TestBjtPhysics:
    def test_gummel_beta(self):
        """Current gain Ic/Ib in mid-injection must exceed 5."""
        sim = _run(create_bjt(
            log_iv=True, iv_file="gummel",
            gummel_sweep=(0.2, 0.8, 0.025), gummel_vce=2.0,
        ))
        iv = parse_iv_file(_iv_path(sim, "gummel"))
        vbe = iv.get_voltages(2)
        ib = np.abs(iv.get_currents(2))
        ic = np.abs(iv.get_currents(3))
        mask = (vbe >= 0.55) & (vbe <= 0.7) & (ib > 0)
        assert mask.sum() >= 2
        beta = np.max(ic[mask] / ib[mask])
        assert beta > 5, f"beta={beta:.1f} too low (base current corrupted?)"


class TestMosCapPhysics:
    def test_accumulation_capacitance_matches_cox(self):
        """C in accumulation must equal eps_ox*A/tox within 25%."""
        tox_um = 0.002
        sim = _run(create_mos_capacitor(
            oxide_thickness=tox_um,
            log_cv=True, vg_sweep=(-2.0, 2.0, 0.2),
        ))
        ac = parse_ac_file(_iv_path(sim, "cv_data"))
        vg, c = ac.get_cv_data(gate_electrode=1)
        assert len(c) > 0, "no C-V data parsed"
        c_acc = np.max(np.abs(c))
        # area: device_width (1 um) x mesh width (1 um) = 1e-8 cm^2
        eps_ox = 3.45e-13  # F/cm
        c_ox = eps_ox * 1e-8 / (tox_um * 1e-4)
        assert abs(c_acc - c_ox) / c_ox < 0.25, (
            f"C_acc={c_acc:.3e} F vs Cox={c_ox:.3e} F"
        )

    def test_hf_cv_shows_depletion(self):
        """Cmin (inversion, HF) must be well below Cox."""
        sim = _run(create_mos_capacitor(
            log_cv=True, vg_sweep=(-2.0, 2.0, 0.2),
        ))
        ac = parse_ac_file(_iv_path(sim, "cv_data"))
        _, c = ac.get_cv_data(gate_electrode=1)
        assert len(c) > 0
        c = np.abs(c)
        assert np.min(c) < 0.7 * np.max(c), "no depletion dip in C-V"
