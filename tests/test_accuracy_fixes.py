"""
Deck-level tests for the device-accuracy fixes (see TODO.md).

These tests validate the *generated decks* against the physical setup of
the reference nanoHUB Rappture tools: mobility models, material cards,
mesh refinement direction, contact placement, solve sequencing, and
node-count limits.  They do not run PADRE (see test_physics_validation.py
for the executable-gated physics checks).
"""

import re
import warnings

import pytest

from nanohubpadre import Simulation, Mesh, Region, Electrode, Doping, Solve
from nanohubpadre.devices import (
    create_bjt,
    create_mesfet,
    create_mos_capacitor,
    create_mosfet,
    create_pn_diode,
    create_schottky_diode,
)
from nanohubpadre.devices._common import NANOHUB_NODE_LIMIT


def _no_proj_warnings(sim):
    """Generate the deck and return any PROJ-sequencing warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.generate_deck()
    return [w for w in caught if "SOLVE PROJ" in str(w.message)]


class TestMosfetModels:
    def test_mobility_models_enabled(self):
        deck = create_mosfet().generate_deck().lower()
        models_line = next(l for l in deck.splitlines() if l.startswith("models"))
        assert "conmob" in models_line
        assert "fldmob" in models_line
        assert "gatmob" in models_line

    def test_silicon_material_card(self):
        deck = create_mosfet().generate_deck().lower()
        silicon = next(l for l in deck.splitlines()
                       if l.startswith("material") and "name=silicon" in l)
        # continuation lines belong to the same card in the raw deck
        idx = deck.splitlines().index(silicon)
        card = silicon
        for line in deck.splitlines()[idx + 1:]:
            if line.startswith("+"):
                card += line
            else:
                break
        assert "klaassen" in card
        assert "vsatn" in card
        assert "mun=1400" in card

    def test_sio2_material_card(self):
        deck = create_mosfet().generate_deck().lower()
        assert any(l.startswith("material") and "name=sio2" in l
                   for l in deck.splitlines())

    def test_default_device_turns_on(self):
        """Default doping/oxide must give a Vt inside a normal sweep."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            create_mosfet(vgs_sweep=(0.0, 1.5, 0.1))
        assert not any("threshold" in str(w.message) for w in caught)

    def test_vt_warning_for_broken_defaults(self):
        """The old defaults (1e19 channel, 12nm oxide) must warn."""
        with pytest.warns(UserWarning, match="threshold"):
            create_mosfet(channel_doping=1e19, gate_oxide_thickness=0.012,
                          vgs_sweep=(0.0, 1.5, 0.1))

    def test_ymesh_refines_into_interface(self):
        """Silicon spacing must contract toward the Si/SiO2 interface."""
        sim = create_mosfet()
        deck = sim.generate_deck().lower()
        y_lines = [l for l in deck.splitlines() if l.startswith("y.m")]
        # The segment ending at the interface (device_depth+junction_depth)
        # is the second-to-last y.m line; its ratio must be < 1.
        interface_seg = y_lines[-2]
        ratio = float(re.search(r"r=([\d.]+)", interface_seg).group(1))
        assert ratio < 1.0
        # The oxide segment (last) expands away from the interface.
        oxide_seg = y_lines[-1]
        ratio_ox = float(re.search(r"r=([\d.]+)", oxide_seg).group(1))
        assert ratio_ox > 1.0


class TestBjtStructure:
    def test_base_electrode_inset_from_junctions(self):
        sim = create_bjt()
        deck = sim.generate_deck().lower()
        base_region = next(l for l in deck.splitlines()
                           if l.startswith("region num=2"))
        r_lo = int(re.search(r"ix\.l=(\d+)", base_region).group(1))
        r_hi = int(re.search(r"ix\.h=(\d+)", base_region).group(1))
        base_elec = next(l for l in deck.splitlines()
                         if l.startswith("elec num=2"))
        e_lo = int(re.search(r"ix\.l=(\d+)", base_elec).group(1))
        e_hi = int(re.search(r"ix\.h=(\d+)", base_elec).group(1))
        # electrode must not touch the junction columns r_lo / r_hi
        assert e_lo > r_lo
        assert e_hi < r_hi

    def test_per_region_lifetime_materials(self):
        deck = create_bjt().generate_deck().lower()
        assert "name=emat" in deck and "taup0=1e-07" in deck
        assert "name=bmat" in deck and "taun0=1e-06" in deck
        assert "name=cmat" in deck

    def test_base_contact_blocks_minority_recombination(self):
        deck = create_bjt().generate_deck().lower()
        base_contact = next(l for l in deck.splitlines()
                            if l.startswith("contact") and "num=2" in l)
        assert "vsurfn=0" in base_contact
        # PNP mirrors with vsurfp
        deck_pnp = create_bjt(device_type="pnp").generate_deck().lower()
        base_contact_pnp = next(l for l in deck_pnp.splitlines()
                                if l.startswith("contact") and "num=2" in l)
        assert "vsurfp=0" in base_contact_pnp

    def test_mesh_refines_into_both_junctions(self):
        deck = create_bjt().generate_deck().lower()
        x_lines = [l for l in deck.splitlines() if l.startswith("x.m")]
        ratios = [float(re.search(r"r=([\d.]+)", l).group(1)) for l in x_lines]
        # segments: [start, E-B junction, base mid, B-C junction, collector end]
        assert ratios[1] < 1.0   # contract into E-B
        assert ratios[2] > 1.0   # expand away from E-B
        assert ratios[3] < 1.0   # contract into B-C
        assert ratios[4] > 1.0   # expand away from B-C

    def test_default_mesh_under_node_limit(self):
        kwargs = create_bjt()._device_kwargs
        assert kwargs["nx"] * kwargs["ny"] <= NANOHUB_NODE_LIMIT

    def test_degenerate_mesh_raises(self):
        with pytest.raises(ValueError):
            create_bjt(nx=5)


class TestMesfetStructure:
    def test_semi_insulating_substrate(self):
        deck = create_mesfet().generate_deck().lower()
        sub_dop = next(l for l in deck.splitlines()
                       if l.startswith("dop") and "reg=1" in l)
        assert "n.type" in sub_dop
        assert "1.000000e10" in sub_dop

    def test_opposite_substrate_option(self):
        deck = create_mesfet(substrate_type="opposite",
                             substrate_doping=1e17).generate_deck().lower()
        sub_dop = next(l for l in deck.splitlines()
                       if l.startswith("dop") and "reg=1" in l)
        assert "p.type" in sub_dop

    def test_gate_gap_from_contact_regions(self):
        deck = create_mesfet().generate_deck().lower()
        src_region = next(l for l in deck.splitlines()
                          if l.startswith("region num=2"))
        drn_region = next(l for l in deck.splitlines()
                          if l.startswith("region num=4"))
        src_hi = int(re.search(r"ix\.h=(\d+)", src_region).group(1))
        drn_lo = int(re.search(r"ix\.l=(\d+)", drn_region).group(1))
        gate = next(l for l in deck.splitlines() if l.startswith("elec num=3"))
        g_lo = int(re.search(r"ix\.l=(\d+)", gate).group(1))
        g_hi = int(re.search(r"ix\.h=(\d+)", gate).group(1))
        assert g_lo > src_hi   # gap between n+ source region and gate
        assert g_hi < drn_lo   # gap between gate and n+ drain region

    def test_fermi_statistics_and_caughey(self):
        deck = create_mesfet().generate_deck().lower()
        assert "statistics=fermi" in deck
        assert "en.mod=caughey" in deck

    def test_default_mesh_under_node_limit(self):
        kwargs = create_mesfet()._device_kwargs
        assert kwargs["nx"] * kwargs["ny"] <= NANOHUB_NODE_LIMIT


class TestMosCapacitor:
    def test_reference_defaults(self):
        kwargs = create_mos_capacitor()._device_kwargs
        assert kwargs["silicon_thickness"] == 5.0
        assert kwargs["substrate_doping"] == 1e16
        assert kwargs["taun0"] == 1e-9
        assert kwargs["taup0"] == 1e-9

    def test_thin_body_warns(self):
        with pytest.warns(UserWarning, match="depletion"):
            create_mos_capacitor(silicon_thickness=0.03, substrate_doping=1e18)

    def test_lf_sweep_ramps_back(self):
        sim = create_mos_capacitor(log_cv=True, log_cv_lf=True,
                                   vg_sweep=(-2.0, 2.0, 0.2))
        deck = sim.generate_deck().lower()
        lines = [l for l in deck.splitlines() if l.startswith(("solve", "log"))]
        idx_init = next(i for i, l in enumerate(lines) if "init" in l)
        idx_hf = next(i for i, l in enumerate(lines) if "ac.analysis" in l)
        idx_lf = next(i for i, l in enumerate(lines)
                      if "ac.analysis" in l and i > idx_hf)
        # ramp up to the sweep start before the HF sweep
        assert any("vg_ramp" in l for l in lines[idx_init:idx_hf])
        # ramp back down, then switch the AC log, before the LF sweep
        between = lines[idx_hf + 1:idx_lf]
        assert any("vg_ramp_lf" in l for l in between)
        assert any(l.startswith("log") and "cv_lf" in l for l in between)

    def test_no_proj_after_init(self):
        sim = create_mos_capacitor(log_cv=True, vg_sweep=(-2.0, 2.0, 0.2))
        assert _no_proj_warnings(sim) == []


class TestSchottkyDiode:
    def test_thermionic_emission_default(self):
        deck = create_schottky_diode().generate_deck().lower()
        contact = next(l for l in deck.splitlines()
                       if l.startswith("contact") and "num=1" in l)
        idx = deck.splitlines().index(contact)
        for line in deck.splitlines()[idx + 1:]:
            if line.startswith("+"):
                contact += line
            else:
                break
        assert "surf" in contact
        assert "vsurfn" in contact
        assert "vsurfp" in contact

    def test_mesh_refines_into_contact(self):
        deck = create_schottky_diode().generate_deck().lower()
        y_lines = [l for l in deck.splitlines() if l.startswith("y.m")]
        # segment from the contact (y=0) outward must expand (ratio > 1)
        near = next(l for l in y_lines if re.search(r"r=([\d.]+)", l))
        ratio = float(re.search(r"r=([\d.]+)", near).group(1))
        assert ratio > 1.0


class TestSolveSequencing:
    """No factory may emit SOLVE PROJ with fewer than two prior solutions."""

    def test_all_factories_sequence_correctly(self):
        sims = [
            create_pn_diode(log_iv=True, forward_sweep=(0.0, 0.8, 0.05)),
            create_pn_diode(log_iv=True, reverse_sweep=(0.0, -5.0, -0.5)),
            create_pn_diode(log_iv=True, forward_sweep=(0.0, 0.8, 0.05),
                            reverse_sweep=(0.0, -5.0, -0.5)),
            create_pn_diode(log_physics_at=[0.0, 0.2, 0.4]),
            create_mosfet(log_iv=True, vgs_sweep=(0.0, 1.5, 0.1)),  # vds=0
            create_mosfet(log_iv=True, vds_sweep=(0.0, 1.0, 0.05)),  # vgs=0
            create_mosfet(contour_maps=True, contour_vgs=0.0),
            create_bjt(log_iv=True, vce_sweep=(0.0, 3.0, 0.1)),  # vbe=0
            create_bjt(log_iv=True, gummel_sweep=(0.0, 0.8, 0.05),
                       gummel_vce=0.0),
            create_bjt(contour_maps=True, contour_vbe=0.0),
            create_mesfet(log_iv=True, vds_sweep=(0.0, 2.0, 0.1)),  # vgs=0
            create_mos_capacitor(log_cv=True, vg_sweep=(-2.0, 2.0, 0.2)),
        ]
        for sim in sims:
            assert _no_proj_warnings(sim) == [], sim.title

    def test_proj_after_init_is_detected(self):
        sim = Simulation(title="bad sequencing")
        sim.mesh = Mesh(nx=3, ny=3)
        sim.mesh.add_x_mesh(1, 0)
        sim.mesh.add_x_mesh(3, 1)
        sim.mesh.add_y_mesh(1, 0)
        sim.mesh.add_y_mesh(3, 1)
        sim.add_region(Region(1, ix_low=1, ix_high=3, iy_low=1, iy_high=3,
                              silicon=True))
        sim.add_electrode(Electrode(1, ix_low=1, ix_high=1, iy_low=1, iy_high=3))
        sim.add_doping(Doping(region=1, n_type=True, uniform=True,
                              concentration=1e16))
        sim.add_solve(Solve(initial=True))
        sim.add_solve(Solve(project=True, v1=0.5, electrode=1))
        with pytest.warns(UserWarning, match="SOLVE PROJ"):
            sim.generate_deck()

    def test_bjt_vbe_is_ramped(self):
        deck = create_bjt(log_iv=True, vbe=0.7,
                          vce_sweep=(0.0, 3.0, 0.1)).generate_deck().lower()
        ramp = next(l for l in deck.splitlines()
                    if l.startswith("solve") and "vbe_set" in l)
        nsteps = int(re.search(r"nsteps=(\d+)", ramp).group(1))
        vstep = abs(float(re.search(r"vstep=([-\d.e]+)", ramp).group(1)))
        assert nsteps >= 5           # 0.7 V in steps of <= 0.1 V
        assert vstep <= 0.1 + 1e-9
        assert "prev" in ramp

    def test_iv_log_excludes_prebias_points(self):
        """LOG must come after the ramp solves, right before the sweep."""
        deck = create_mosfet(log_iv=True, vgs_sweep=(0.0, 1.5, 0.1),
                             vds=0.5).generate_deck().lower()
        lines = [l for l in deck.splitlines()
                 if l.startswith(("solve", "log"))]
        log_idx = next(i for i, l in enumerate(lines) if l.startswith("log"))
        # everything before the log is init/ramp; the sweep follows the log
        assert any("vd_set" in l for l in lines[:log_idx])
        assert "idvg" in lines[log_idx + 1]


class TestNodeLimit:
    def test_oversize_mesh_warns(self):
        with pytest.warns(UserWarning, match="node"):
            create_pn_diode(nx=100, ny=30)

    def test_default_factories_do_not_warn_about_size(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            create_mosfet()
            create_bjt()
            create_mesfet()
            create_mos_capacitor()
            create_pn_diode()
            create_schottky_diode()
        assert not any("node" in str(w.message).lower() and
                       "limit" in str(w.message).lower() for w in caught)
