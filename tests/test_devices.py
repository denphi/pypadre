"""
Unit tests for device factory functions.

Tests the nanohubpadre.devices module factory functions for creating
common semiconductor device simulations.
"""

import unittest

from nanohubpadre import (
    create_pn_diode,
    create_nin_diode,
    create_mos_capacitor,
    create_mosfet,
    create_mesfet,
    create_bjt,
    create_schottky_diode,
    create_solar_cell,
    Simulation,
    Solve,
    Log,
)


class TestPNDiodeFactory(unittest.TestCase):
    """Tests for create_pn_diode factory function."""

    def test_default_creation(self):
        """Test creating a PN diode with default parameters."""
        sim = create_pn_diode()
        self.assertIsInstance(sim, Simulation)
        self.assertIsNotNone(sim.mesh)
        self.assertEqual(sim.title, "PN Junction Diode")

    def test_custom_parameters(self):
        """Test creating a PN diode with custom parameters."""
        sim = create_pn_diode(
            length=2.0,
            width=1.5,
            junction_position=0.3,
            p_doping=1e16,
            n_doping=1e18,
            temperature=350,
            title="Custom PN Diode"
        )
        deck = sim.generate_deck()
        self.assertIn("Custom PN Diode", deck)
        self.assertIn("temp=350", deck.lower())
        # Check concentration format (may be 1e16 or 1.000000e16)
        self.assertIn("e16", deck.lower())
        self.assertIn("e18", deck.lower())

    def test_deck_generation(self):
        """Test that generated deck contains required elements."""
        sim = create_pn_diode()
        deck = sim.generate_deck().lower()
        self.assertIn("mesh", deck)
        self.assertIn("region", deck)
        self.assertIn("elec", deck)
        self.assertIn("dop", deck)
        self.assertIn("models", deck)
        self.assertIn("system", deck)

    def test_with_solve_commands(self):
        """Test adding solve commands to PN diode."""
        sim = create_pn_diode()
        sim.add_solve(Solve(initial=True))
        sim.add_log(Log(ivfile="iv"))
        sim.add_solve(Solve(project=True, vstep=0.05, nsteps=10, electrode=1))
        deck = sim.generate_deck().lower()
        self.assertIn("init", deck)
        self.assertIn("log", deck)
        self.assertIn("proj", deck)


class TestMOSCapacitorFactory(unittest.TestCase):
    """Tests for create_mos_capacitor factory function."""

    def test_default_creation(self):
        """Test creating a MOS capacitor with default parameters."""
        sim = create_mos_capacitor()
        self.assertIsInstance(sim, Simulation)
        self.assertIsNotNone(sim.mesh)

    def test_custom_oxide_thickness(self):
        """Test creating MOS capacitor with custom oxide thickness."""
        sim = create_mos_capacitor(oxide_thickness=0.005)
        deck = sim.generate_deck()
        self.assertIn("0.005", deck)

    def test_p_type_substrate(self):
        """Test creating MOS capacitor with p-type substrate."""
        sim = create_mos_capacitor(substrate_type="p")
        deck = sim.generate_deck().lower()
        self.assertIn("p.type", deck)

    def test_n_type_substrate(self):
        """Test creating MOS capacitor with n-type substrate."""
        sim = create_mos_capacitor(substrate_type="n")
        deck = sim.generate_deck().lower()
        self.assertIn("n.type", deck)

    def test_gate_types(self):
        """Test different gate contact types."""
        sim_npoly = create_mos_capacitor(gate_type="n_poly")
        deck_npoly = sim_npoly.generate_deck().lower()
        self.assertIn("n.polysilicon", deck_npoly)

        sim_ppoly = create_mos_capacitor(gate_type="p_poly")
        deck_ppoly = sim_ppoly.generate_deck().lower()
        self.assertIn("p.polysilicon", deck_ppoly)

    def test_oxide_regions(self):
        """Test that oxide regions are created correctly."""
        sim = create_mos_capacitor()
        deck = sim.generate_deck().lower()
        self.assertIn("sio2", deck)
        self.assertIn("ins", deck)


class TestMOSFETFactory(unittest.TestCase):
    """Tests for create_mosfet factory function."""

    def test_default_nmos_creation(self):
        """Test creating an NMOS transistor with defaults."""
        sim = create_mosfet()
        self.assertIsInstance(sim, Simulation)
        self.assertEqual(sim.title, "NMOS MOSFET")

    def test_pmos_creation(self):
        """Test creating a PMOS transistor."""
        sim = create_mosfet(device_type="pmos")
        deck = sim.generate_deck().lower()
        self.assertIn("pmos", deck)
        # PMOS should have p+ S/D
        self.assertIn("p.type", deck)

    def test_regions_structure(self):
        """Test that MOSFET has correct region structure."""
        sim = create_mosfet()
        deck = sim.generate_deck().lower()
        # Should have substrate, source, drain, channel, gate oxide regions
        self.assertGreater(deck.count("region"), 4)

    def test_electrodes_structure(self):
        """Test that MOSFET has 4 electrodes."""
        sim = create_mosfet()
        deck = sim.generate_deck().lower()
        # Should have source, drain, gate, substrate contacts
        # Count "elec num=" to avoid matching "electrons"
        self.assertEqual(deck.count("elec num="), 4)

    def test_custom_doping(self):
        """Test MOSFET with custom doping levels."""
        sim = create_mosfet(
            source_drain_doping=5e19,
            channel_doping=5e18,
            substrate_doping=1e17
        )
        deck = sim.generate_deck().lower()
        # Check concentration format (may be 5e19 or 5.000000e19)
        self.assertIn("5", deck)
        self.assertIn("e19", deck)
        self.assertIn("e18", deck)
        self.assertIn("e17", deck)


class TestMESFETFactory(unittest.TestCase):
    """Tests for create_mesfet factory function."""

    def test_default_creation(self):
        """Test creating a MESFET with default parameters."""
        sim = create_mesfet()
        self.assertIsInstance(sim, Simulation)
        self.assertIn("MESFET", sim.title)

    def test_gate_workfunction(self):
        """Test MESFET gate workfunction setting."""
        sim = create_mesfet(gate_workfunction=4.9)
        deck = sim.generate_deck().lower()
        self.assertIn("workfunction=4.9", deck)

    def test_n_channel(self):
        """Test n-channel MESFET."""
        sim = create_mesfet(device_type="n")
        deck = sim.generate_deck().lower()
        self.assertIn("n.type", deck)

    def test_p_channel(self):
        """Test p-channel MESFET."""
        sim = create_mesfet(device_type="p")
        deck = sim.generate_deck().lower()
        self.assertIn("p.type", deck)

    def test_schottky_gate(self):
        """Test that MESFET has Schottky gate contact."""
        sim = create_mesfet()
        deck = sim.generate_deck().lower()
        self.assertIn("workfunction", deck)


class TestBJTFactory(unittest.TestCase):
    """Tests for create_bjt factory function."""

    def test_default_npn_creation(self):
        """Test creating an NPN BJT with defaults."""
        sim = create_bjt()
        self.assertIsInstance(sim, Simulation)
        self.assertIn("NPN", sim.title)

    def test_pnp_creation(self):
        """Test creating a PNP BJT."""
        sim = create_bjt(device_type="pnp")
        deck = sim.generate_deck().lower()
        self.assertIn("pnp", deck)

    def test_three_regions(self):
        """Test that BJT has three regions (E, B, C)."""
        sim = create_bjt()
        deck = sim.generate_deck().lower()
        # Should have exactly 3 regions
        self.assertEqual(deck.count("region"), 3)

    def test_three_electrodes(self):
        """Test that BJT has three electrodes."""
        sim = create_bjt()
        deck = sim.generate_deck().lower()
        # Count "elec num=" to avoid matching "electrons"
        self.assertEqual(deck.count("elec num="), 3)

    def test_doping_profiles_npn(self):
        """Test NPN doping profiles (n-p-n)."""
        sim = create_bjt(device_type="npn")
        deck = sim.generate_deck().lower()
        # Should have both N.TYPE and P.TYPE for NPN
        self.assertIn("n.type", deck)
        self.assertIn("p.type", deck)

    def test_models_enabled(self):
        """Test that BJT has SRH, Auger, and BGN models."""
        sim = create_bjt()
        deck = sim.generate_deck().lower()
        self.assertIn("srh", deck)
        self.assertIn("auger", deck)
        self.assertIn("bgn", deck)


class TestSchottkyDiodeFactory(unittest.TestCase):
    """Tests for create_schottky_diode factory function."""

    def test_default_creation(self):
        """Test creating a Schottky diode with defaults."""
        sim = create_schottky_diode()
        self.assertIsInstance(sim, Simulation)
        self.assertEqual(sim.title, "Schottky Diode")

    def test_workfunction_setting(self):
        """Test Schottky barrier workfunction."""
        sim = create_schottky_diode(workfunction=4.7)
        deck = sim.generate_deck().lower()
        self.assertIn("workfunction=4.7", deck)

    def test_barrier_lowering(self):
        """Test barrier lowering option."""
        sim = create_schottky_diode(barrier_lowering=True)
        deck = sim.generate_deck().lower()
        self.assertIn("barrierl", deck)

    def test_surface_recombination(self):
        """Test surface recombination at Schottky contact."""
        sim = create_schottky_diode(surf_rec=True)
        deck = sim.generate_deck().lower()
        self.assertIn("surf.rec", deck)

    def test_n_type_semiconductor(self):
        """Test n-type Schottky diode."""
        sim = create_schottky_diode(doping_type="n")
        deck = sim.generate_deck().lower()
        self.assertIn("n.type", deck)

    def test_two_electrodes(self):
        """Test Schottky diode has two electrodes."""
        sim = create_schottky_diode()
        deck = sim.generate_deck().lower()
        # Count "elec num=" to avoid matching "electrons"
        self.assertEqual(deck.count("elec num="), 2)


class TestSolarCellFactory(unittest.TestCase):
    """Tests for create_solar_cell factory function."""

    def test_default_creation(self):
        """Test creating a solar cell with defaults."""
        sim = create_solar_cell()
        self.assertIsInstance(sim, Simulation)
        self.assertIn("Solar Cell", sim.title)

    def test_n_on_p_structure(self):
        """Test N-on-P solar cell structure."""
        sim = create_solar_cell(device_type="n_on_p")
        deck = sim.generate_deck().lower()
        # Should have both doping types
        self.assertIn("n.type", deck)
        self.assertIn("p.type", deck)
        # Should have Gaussian emitter
        self.assertIn("gaussian", deck)

    def test_p_on_n_structure(self):
        """Test P-on-N solar cell structure."""
        sim = create_solar_cell(device_type="p_on_n")
        deck = sim.generate_deck().lower()
        self.assertIn("n.type", deck)
        self.assertIn("p.type", deck)

    def test_surface_recombination_velocities(self):
        """Test surface recombination velocity settings."""
        sim = create_solar_cell(
            front_surface_velocity=1e3,
            back_surface_velocity=1e6
        )
        deck = sim.generate_deck().lower()
        self.assertIn("vsurfn", deck)
        self.assertIn("vsurfp", deck)

    def test_material_lifetimes(self):
        """Test material lifetime settings."""
        sim = create_solar_cell(taun0=1e-4, taup0=1e-4)
        deck = sim.generate_deck().lower()
        self.assertIn("taun0", deck)
        self.assertIn("taup0", deck)


class TestDeviceAliases(unittest.TestCase):
    """Tests for device factory function aliases."""

    def test_pn_diode_alias(self):
        """Test that pn_diode alias works."""
        from nanohubpadre import pn_diode
        sim = pn_diode()
        self.assertIsInstance(sim, Simulation)

    def test_mos_capacitor_alias(self):
        """Test that mos_capacitor alias works."""
        from nanohubpadre import mos_capacitor
        sim = mos_capacitor()
        self.assertIsInstance(sim, Simulation)

    def test_mosfet_alias(self):
        """Test that mosfet alias works."""
        from nanohubpadre import mosfet
        sim = mosfet()
        self.assertIsInstance(sim, Simulation)

    def test_mesfet_alias(self):
        """Test that mesfet alias works."""
        from nanohubpadre import mesfet
        sim = mesfet()
        self.assertIsInstance(sim, Simulation)

    def test_bjt_alias(self):
        """Test that bjt alias works."""
        from nanohubpadre import bjt
        sim = bjt()
        self.assertIsInstance(sim, Simulation)

    def test_schottky_diode_alias(self):
        """Test that schottky_diode alias works."""
        from nanohubpadre import schottky_diode
        sim = schottky_diode()
        self.assertIsInstance(sim, Simulation)

    def test_solar_cell_alias(self):
        """Test that solar_cell alias works."""
        from nanohubpadre import solar_cell
        sim = solar_cell()
        self.assertIsInstance(sim, Simulation)


class TestDeviceDeckStructure(unittest.TestCase):
    """Tests for proper deck structure in all device types."""

    def test_all_devices_have_end(self):
        """Test that all device decks end with 'end'."""
        devices = [
            create_pn_diode(),
            create_mos_capacitor(),
            create_mosfet(),
            create_mesfet(),
            create_bjt(),
            create_schottky_diode(),
            create_solar_cell(),
            create_nin_diode(),
        ]
        for sim in devices:
            deck = sim.generate_deck()
            self.assertTrue(deck.strip().endswith("end"))

    def test_all_devices_have_mesh(self):
        """Test that all devices have mesh specification."""
        devices = [
            create_pn_diode(),
            create_mos_capacitor(),
            create_mosfet(),
            create_mesfet(),
            create_bjt(),
            create_schottky_diode(),
            create_solar_cell(),
            create_nin_diode(),
        ]
        for sim in devices:
            deck = sim.generate_deck().lower()
            self.assertIn("mesh", deck)

    def test_all_devices_have_models(self):
        """Test that all devices have models specification."""
        devices = [
            create_pn_diode(),
            create_mos_capacitor(),
            create_mosfet(),
            create_mesfet(),
            create_bjt(),
            create_schottky_diode(),
            create_solar_cell(),
            create_nin_diode(),
        ]
        for sim in devices:
            deck = sim.generate_deck().lower()
            self.assertIn("models", deck)


if __name__ == "__main__":
    unittest.main()
