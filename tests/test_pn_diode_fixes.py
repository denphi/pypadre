"""
Regression tests for the 2026-09-06 PN-diode review.

Every expectation here was checked against PADRE 2.4E-r15 running on nanoHUB;
the comments name the observed behaviour that motivated each test.
"""

import warnings

import numpy as np
import pytest

from nanohubpadre import Log, Simulation, create_pn_diode
from nanohubpadre.devices._common import sweep_steps
from nanohubpadre.estimates import pn_depletion_width_um
from nanohubpadre.parser import IVData


def deck_lines(sim):
    """Deck lines with PADRE continuation lines folded back in."""
    out = []
    for raw in sim.generate_deck().splitlines():
        if raw.startswith("+") and out:
            out[-1] += " " + raw.lstrip("+ ").strip()
        else:
            out.append(raw)
    return out


def kv(line):
    """key=value pairs from a deck line, ignoring bare flags like no.append."""
    return dict(tok.split("=", 1) for tok in line.split() if "=" in tok)


def x_mesh_nodes(sim):
    """Node positions PADRE will build from the deck's X.MESH lines.

    Reproduces PADRE's geometric RATIO expansion, verified to six
    significant figures against PADRE's own mesh output file.
    """
    spec = []
    for line in deck_lines(sim):
        if not line.startswith("x.m"):
            continue
        parts = kv(line)
        spec.append((int(parts["n"]), float(parts["l"]),
                     float(parts.get("r", 1.0))))
    x = np.zeros(spec[-1][0])
    for (n0, l0, _), (n1, l1, r) in zip(spec, spec[1:]):
        k, seg = n1 - n0, l1 - l0
        h = (np.full(k, seg / k) if abs(r - 1) < 1e-12
             else seg * (1 - r) / (1 - r ** k) * r ** np.arange(k))
        x[n0:n1] = l0 + np.cumsum(h)
    return x


class TestSweepArithmetic:
    def test_endpoint_is_not_truncated_by_float_error(self):
        """0.6/0.05 == 11.999999999999998, so int() stopped at 0.55 V."""
        nsteps, step, v_end = sweep_steps(0.0, 0.6, 0.05)
        assert nsteps == 12
        assert v_end == pytest.approx(0.6)

    @pytest.mark.parametrize("v_end,v_step", [(0.6, 0.05), (0.7, 0.05),
                                              (1.2, 0.2), (0.8, 0.05)])
    def test_sweeps_reach_their_requested_endpoint(self, v_end, v_step):
        assert sweep_steps(0.0, v_end, v_step)[2] == pytest.approx(v_end)

    def test_step_sign_follows_the_sweep_direction(self):
        """A negative step on an upward sweep used to run it backwards."""
        with pytest.warns(UserWarning, match="disagrees in sign"):
            nsteps, step, v_end = sweep_steps(0.0, 0.8, -0.05)
        assert step > 0 and v_end == pytest.approx(0.8)

    def test_non_integer_step_count_warns(self):
        with pytest.warns(UserWarning, match="not a whole number"):
            assert sweep_steps(0.0, 0.7, 0.03)[2] == pytest.approx(0.69)

    def test_zero_step_is_rejected(self):
        with pytest.raises(ValueError):
            sweep_steps(0.0, 0.8, 0.0)

    def test_deck_uses_the_corrected_step_count(self):
        sim = create_pn_diode(log_iv=True, forward_sweep=(0.0, 0.6, 0.05))
        assert any("nsteps=12" in l for l in deck_lines(sim)
                   if l.startswith("solve"))


class TestGeometryValidation:
    def test_intrinsic_region_past_the_contact_is_rejected(self):
        """PADRE aborted: 'Illegal or ambiguous mesh line defn!'"""
        with pytest.raises(ValueError, match="runs off the end"):
            create_pn_diode(junction_position=0.8, intrinsic_width=0.5)

    @pytest.mark.parametrize("jp", [0.0, 1.0, -0.1, 1.5])
    def test_junction_position_must_be_interior(self, jp):
        with pytest.raises(ValueError, match="junction_position"):
            create_pn_diode(junction_position=jp)

    def test_valid_pin_still_builds(self):
        sim = create_pn_diode(junction_position=0.3, intrinsic_width=0.4)
        assert sim.generate_deck().strip().endswith("end")

    def test_bad_sweep_electrode_is_rejected(self):
        with pytest.raises(ValueError, match="sweep_electrode"):
            create_pn_diode(sweep_electrode=3, forward_sweep=(0.0, 0.5, 0.1))

    def test_mesh_node_numbers_stay_in_range_and_ascending(self):
        for kwargs in ({}, {"junction_position": 0.3, "intrinsic_width": 0.4},
                       {"junction_position": 0.1}, {"length": 9.0, "nx": 180}):
            sim = create_pn_diode(**kwargs)
            nodes = [int(kv(l)["n"]) for l in deck_lines(sim)
                     if l.startswith("x.m")]
            assert nodes == sorted(nodes), kwargs
            assert nodes[-1] == sim._device_kwargs["nx"], kwargs


class TestSweepPolarity:
    def test_forward_sweep_on_the_cathode_warns(self):
        """This produced examples/iv: leakage floor logged as 'fwd'."""
        with pytest.warns(UserWarning, match="REVERSE-biases"):
            create_pn_diode(forward_sweep=(0.0, 0.8, 0.05), sweep_electrode=2)

    def test_correct_polarity_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            create_pn_diode(forward_sweep=(0.0, 0.8, 0.05), sweep_electrode=1)
            create_pn_diode(forward_sweep=(0.0, -0.8, -0.05), sweep_electrode=2)


class TestMeshQuality:
    def test_default_mesh_is_not_degenerate(self):
        """The old r=0.8-over-99-intervals mesh reached h_min = 3.2e-11 um
        and made forward-then-reverse sweeps fail to converge."""
        x = x_mesh_nodes(create_pn_diode())
        h = np.diff(x)
        assert h.min() > 1e-4, f"h_min={h.min():.3g} um is unphysically small"
        assert h.max() / h.min() < 50, f"spacing ratio {h.max() / h.min():.3g}"

    @pytest.mark.parametrize("kwargs", [
        {},
        {"length": 9.0, "nx": 180, "p_doping": 2e15, "n_doping": 1e15},
        {"junction_position": 0.3, "intrinsic_width": 0.4},
        {"length": 0.2, "nx": 100, "p_doping": 1e18, "n_doping": 1e18},
    ])
    def test_mesh_stays_sane_across_geometries(self, kwargs):
        x = x_mesh_nodes(create_pn_diode(**kwargs))
        h = np.diff(x)
        assert h.min() > 1e-4
        assert h.max() / h.min() < 50

    def test_depletion_region_is_resolved(self):
        x = x_mesh_nodes(create_pn_diode())
        w = pn_depletion_width_um(1e17, 1e17)
        inside = int((np.abs(x - 0.5) < w).sum())
        assert inside > 20, f"only {inside} nodes across the depletion region"

    def test_junction_lands_exactly_on_a_mesh_node(self):
        x = x_mesh_nodes(create_pn_diode(junction_position=0.3))
        assert np.abs(x - 0.3).min() < 1e-9

    def test_pin_resolves_both_junctions(self):
        x = x_mesh_nodes(create_pn_diode(junction_position=0.3,
                                         intrinsic_width=0.4))
        assert np.abs(x - 0.3).min() < 1e-9
        assert np.abs(x - 0.7).min() < 1e-9


class TestSolveOutput:
    def test_stepped_solves_do_not_write_a_file_per_bias_point(self):
        """A default run wrote 31 solution files into the working dir."""
        sim = create_pn_diode(log_iv=True, forward_sweep=(0.0, 0.8, 0.05),
                              reverse_sweep=(0.0, -5.0, -0.5))
        stepped = [l for l in deck_lines(sim)
                   if l.startswith("solve") and "vstep=" in l]
        assert stepped
        assert all("no.append" in l for l in stepped)

    def test_reverse_ramp_uses_junction_sized_steps(self):
        """Ramping down off forward bias at 0.2 V/step failed to converge."""
        sim = create_pn_diode(log_iv=True, forward_sweep=(0.0, 0.6, 0.05),
                              reverse_sweep=(0.0, -5.0, -0.5))
        ramp = next(l for l in deck_lines(sim) if "rev_ramp" in l)
        assert abs(float(kv(ramp)["vstep"])) <= 0.1 + 1e-9


class TestLogPhysicsAt:
    @pytest.mark.parametrize("bad", [[], [0.2, 0.4], [0.0, 0.4, 0.2]])
    def test_invalid_lists_are_rejected(self, bad):
        with pytest.raises(ValueError):
            create_pn_diode(log_physics_at=bad)

    def test_large_steps_are_split(self):
        """0.2 V in one solve exceeds the module's own JUNCTION_STEP."""
        sim = create_pn_diode(log_physics_at=[0.0, 0.2, 0.4])
        for line in deck_lines(sim):
            if line.startswith("solve") and "vstep=" in line:
                assert abs(float(kv(line)["vstep"])) <= 0.1 + 1e-9


class TestSimulationLog:
    def test_log_assignment_reaches_the_deck(self):
        """`sim.log = Log(...)` used to be a silent no-op."""
        sim = Simulation(title="t")
        sim.log = Log(ivfile="pn_iv.log")
        assert any(l.startswith("log") for l in sim.generate_deck().splitlines())
        assert sim.log is not None

    def test_log_assignment_replaces_rather_than_appends(self):
        sim = Simulation(title="t")
        sim.log = Log(ivfile="a")
        sim.log = Log(ivfile="b")
        logs = [l for l in sim.generate_deck().splitlines() if l.startswith("log")]
        assert logs == ["log outf=b"]

    def test_log_rejects_non_log(self):
        with pytest.raises(TypeError):
            Simulation(title="t").log = "iv"


class TestContinuityCheck:
    def _iv(self, pairs):
        d = IVData(num_electrodes=2)
        for v, i1, i2 in pairs:
            d.bias_points.append({"voltages": {1: v, 2: 0.0},
                                  "currents": {1: {"total": i1},
                                               2: {"total": i2}}})
        return d

    def test_converged_data_passes(self):
        iv = self._iv([(0.5, 1e-9, -1e-9), (0.6, 1e-8, -1e-8)])
        assert not iv.check_continuity().any()
        assert np.allclose(iv.continuity_error(), 0.0)

    def test_noise_floor_data_is_flagged(self):
        iv = self._iv([(0.5, 1e-17, -1.3e-17), (0.6, 2e-17, -2.6e-17)])
        with pytest.warns(UserWarning, match="continuity"):
            assert iv.check_continuity().all()

    def test_empty_data_is_safe(self):
        assert len(IVData(num_electrodes=2).check_continuity()) == 0


# ---------------------------------------------------------------------------
# Cross-device regressions from the 2026-09-06 all-device review.
# Every number below was measured against PADRE 2.4E-r15 on nanoHUB.
# ---------------------------------------------------------------------------

from nanohubpadre.devices import (create_bjt, create_mesfet,  # noqa: E402
                                  create_nin_diode,
                                  create_mos_capacitor, create_mosfet,
                                  create_schottky_diode, create_solar_cell)

ALL_FACTORIES = {
    "mosfet_vgs": lambda **k: create_mosfet(log_iv=True, vgs_sweep=(0.0, 0.7, 0.1), **k),
    "mosfet_vds": lambda **k: create_mosfet(log_iv=True, vds_sweep=(0.0, 0.7, 0.1), vgs=1.2, **k),
    "mesfet": lambda **k: create_mesfet(log_iv=True, vds_sweep=(0.0, 0.7, 0.1), **k),
    "bjt_vce": lambda **k: create_bjt(log_iv=True, vce_sweep=(0.0, 0.7, 0.1), vbe=0.7, **k),
    "bjt_gummel": lambda **k: create_bjt(log_iv=True, gummel_sweep=(0.0, 0.7, 0.1), **k),
    "moscap": lambda **k: create_mos_capacitor(log_cv=True, vg_sweep=(0.0, 0.7, 0.1), **k),
    "schottky": lambda **k: create_schottky_diode(log_iv=True, forward_sweep=(0.0, 0.7, 0.1), **k),
    "solar": lambda **k: create_solar_cell(log_iv=True, forward_sweep=(0.0, 0.7, 0.1), **k),
}


class TestSweepEndpointsEverywhere:
    """0.7/0.1 == 6.999999999999999, so int() lost the last step in every
    factory.  Observed live: create_solar_cell stopped at 0.6 V."""

    @pytest.mark.parametrize("name", sorted(ALL_FACTORIES))
    def test_sweep_reaches_requested_endpoint(self, name):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = ALL_FACTORIES[name]()
        ends = []
        for line in deck_lines(sim):
            if line.startswith("solve") and "vstep=" in line:
                p = kv(line)
                v0 = next((float(p[k]) for k in ("v1", "v2", "v3", "v4") if k in p), 0.0)
                ends.append(round(v0 + float(p["vstep"]) * int(p["nsteps"]), 6))
        assert ends, name
        assert any(abs(e - 0.7) < 1e-6 for e in ends), f"{name}: endpoints {ends}"

    def test_no_factory_still_truncates_with_int(self):
        import pathlib
        src = pathlib.Path("nanohubpadre/devices")
        hits = [p.name for p in src.glob("*.py")
                if "int(abs(v_end" in p.read_text()]
        assert hits == [], f"unconverted sweep sites in {hits}"


class TestMeshSanityEverywhere:
    """Cells far below the silicon lattice constant (5.43 A = 5.43e-4 um)
    stall PADRE: the MOSFET y-mesh reached 8e-6 um and its drain sweep
    stopped at Vd = 0.093 V of a requested 1.5 V."""

    @pytest.mark.parametrize("name", sorted(ALL_FACTORIES))
    def test_no_subatomic_cells(self, name):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = ALL_FACTORIES[name]()
        for tag in ("x.m", "y.m"):
            spec = [(int(kv(l)["n"]), float(kv(l)["l"]), float(kv(l).get("r", 1.0)))
                    for l in deck_lines(sim) if l.startswith(tag)]
            if len(spec) < 2:
                continue
            hs = []
            for (n0, l0, _), (n1, l1, r) in zip(spec, spec[1:]):
                k, seg = n1 - n0, l1 - l0
                h = (np.full(k, seg / k) if abs(r - 1) < 1e-12
                     else seg * (1 - r) / (1 - r ** k) * r ** np.arange(k))
                hs.append(np.abs(h))
            h = np.concatenate(hs)
            assert h.min() > 1e-4, (
                f"{name} {tag}: h_min={h.min():.3g} um is below atomic scale")


class TestMosfetMeshRegression:
    def test_interface_cell_is_physical(self):
        """Was 7.99e-06 um (0.08 A); the drain sweep stalled at Vd=0.093 V."""
        sim = create_mosfet()
        y = [(int(kv(l)["n"]), float(kv(l)["l"]), float(kv(l).get("r", 1.0)))
             for l in deck_lines(sim) if l.startswith("y.m")]
        hs = []
        for (n0, l0, _), (n1, l1, r) in zip(y, y[1:]):
            k, seg = n1 - n0, l1 - l0
            hs.append(np.abs(np.full(k, seg / k) if abs(r - 1) < 1e-12
                             else seg * (1 - r) / (1 - r ** k) * r ** np.arange(k)))
        h = np.concatenate(hs)
        assert h.min() > 2e-4
        assert h.max() / h.min() < 100

    def test_grading_directions_still_match_the_reference(self):
        """The reference deck refines into the Si/SiO2 interface and
        expands away from it through the oxide; only the magnitude changed."""
        y = [l for l in deck_lines(create_mosfet()) if l.startswith("y.m")]
        assert float(kv(y[-2])["r"]) < 1.0
        assert float(kv(y[-1])["r"]) > 1.0


class TestSolarCellWarning:
    def test_warning_does_not_recommend_device_z_width(self):
        """Scaling z multiplies signal and noise identically; verified on
        nanoHUB that z=1 and z=1e4 give the same currents after scaling."""
        with pytest.warns(UserWarning) as rec:
            create_solar_cell()
        msg = str(rec[0].message)
        assert "does NOT help" in msg
        assert "Increase device_z_width" not in msg


class TestMosCapMeshRegression:
    def test_oxide_cells_are_physical(self):
        """ny_oxide=100 across a 2 nm oxide gave 0.02 nm cells, and RATIO=0.8
        compounding over 50 intervals drove h_min to 3.6e-9 um."""
        sim = create_mos_capacitor()
        y = [(int(kv(l)["n"]), float(kv(l)["l"]), float(kv(l).get("r", 1.0)))
             for l in deck_lines(sim) if l.startswith("y.m")]
        hs = []
        for (n0, l0, _), (n1, l1, r) in zip(y, y[1:]):
            k, seg = n1 - n0, l1 - l0
            hs.append(np.abs(np.full(k, seg / k) if abs(r - 1) < 1e-12
                             else seg * (1 - r) / (1 - r ** k) * r ** np.arange(k)))
        assert np.concatenate(hs).min() > 1e-4

    def test_over_meshed_oxide_warns(self):
        with pytest.warns(UserWarning, match="below atomic scale"):
            create_mos_capacitor(ny_oxide=100)

    def test_cv_geometry_is_unchanged(self):
        """Region and electrode indices must still line up after the
        node reallocation, or the C-V is measured across the wrong stack.
        Verified on PADRE: Cox and Cmin match analytic within 5%."""
        deck = deck_lines(create_mos_capacitor())
        ny = int(kv(next(l for l in deck if l.startswith("mesh")))["ny"])
        back = next(l for l in deck if l.startswith("elec num=2"))
        assert int(kv(back)["iy.l"]) == ny
        regions = [l for l in deck if l.startswith("region")]
        assert int(kv(regions[-1])["iy.h"]) == ny


class TestNinDiode:
    """NIN / PIP isotype diodes.  Verified on PADRE 2.4E-r15 (nanoHUB):
    exit 0, no convergence problems, terminal-current continuity 0.000%,
    and I-V antisymmetry to 0.001% over a +-2 V sweep."""

    def test_structure_is_isotype(self):
        """All three regions must be the same polarity - no PN junction."""
        for dtype, want, other in (("nin", "n.type", "p.type"),
                                   ("pip", "p.type", "n.type")):
            deck = deck_lines(create_nin_diode(device_type=dtype))
            dops = [l for l in deck if l.startswith("dop")]
            assert len(dops) == 3, dtype
            assert all(want in l for l in dops), dtype
            assert not any(other in l for l in dops), dtype

    def test_barrier_is_more_lightly_doped_than_the_contacts(self):
        deck = deck_lines(create_nin_diode())
        concs = [float(kv(l)["conc"]) for l in deck if l.startswith("dop")]
        assert concs[1] < concs[0] and concs[1] < concs[2]

    def test_mesh_resolves_both_interfaces(self):
        x = x_mesh_nodes(create_nin_diode())
        assert np.abs(x - 0.6).min() < 1e-9
        assert np.abs(x - 1.4).min() < 1e-9

    def test_mesh_has_no_subatomic_cells(self):
        h = np.diff(x_mesh_nodes(create_nin_diode()))
        assert h.min() > 1e-4
        assert h.max() / h.min() < 50

    def test_symmetric_sweep_ramps_instead_of_jumping(self):
        """A (-2, 2) sweep starts 2 V away from equilibrium."""
        deck = deck_lines(create_nin_diode(log_iv=True, bias_sweep=(-2.0, 2.0, 0.25)))
        ramp = next((l for l in deck if "outf=ramp" in l), None)
        assert ramp is not None
        assert abs(float(kv(ramp)["vstep"])) <= 0.1 + 1e-9

    def test_sweep_reaches_its_endpoint(self):
        deck = deck_lines(create_nin_diode(log_iv=True, bias_sweep=(-2.0, 2.0, 0.25)))
        sweep = next(l for l in deck if "outf=sweep" in l)
        p = kv(sweep)
        assert round(float(p["v1"]) + float(p["vstep"]) * int(p["nsteps"]), 6) == 2.0

    @pytest.mark.parametrize("kwargs,match", [
        ({"device_type": "npn"}, "device_type"),
        ({"junction_position": 0.8, "intrinsic_width": 0.5, "length": 1.0}, "runs off the end"),
        ({"intrinsic_width": 0.0}, "intrinsic_width"),
        ({"sweep_electrode": 3}, "sweep_electrode"),
    ])
    def test_invalid_geometry_is_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            create_nin_diode(**kwargs)

    def test_inverted_doping_warns(self):
        with pytest.warns(UserWarning, match="no high-resistivity barrier"):
            create_nin_diode(contact_doping=1e14, intrinsic_doping=1e18)

    def test_registered_everywhere(self, capsys):
        import nanohubpadre
        from nanohubpadre.devices import describe, list_devices
        assert "nin_diode" in list_devices()
        assert hasattr(nanohubpadre, "create_nin_diode")
        assert hasattr(nanohubpadre, "nin_diode")
        assert hasattr(nanohubpadre, "pip_diode")
        describe("nin_diode")          # prints; must not raise
        assert "create_nin_diode" in capsys.readouterr().out
        assert create_nin_diode().device_schematic() is not None
