"""
Golden-deck tests: compare factory decks against the reference nanoHUB
Rappture decks embedded in ``rappture/*.xml``.

Each Rappture XML contains the PADRE deck the production nanoHUB tool ran
(line-numbered inside the simulation log).  These tests extract the
physics-bearing statements (MODELS flags, material model choices, doping
types) from the reference and assert the corresponding factory emits the
same physics.  They skip when the reference files are not present.
"""

import os
import re

import pytest

from nanohubpadre.devices import (
    create_bjt,
    create_mesfet,
    create_mos_capacitor,
    create_mosfet,
)

RAPPTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "rappture")


def _reference_text(name):
    path = os.path.join(RAPPTURE_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"reference file {name} not available")
    with open(path, errors="replace") as f:
        return f.read()


def _reference_deck_line(text, pattern):
    """
    Find a deck statement (numbered 'NN... <stmt>') matching pattern.

    Some reference XMLs store the whole log on a single line, so the
    statement ends at the next numbered statement, not at a newline.
    """
    m = re.search(
        r"\d+\.\.\.\s*(" + pattern + r".*?)(?=\s+\d+\.\.\.|\n|$)",
        text, re.IGNORECASE | re.DOTALL,
    )
    assert m, f"pattern {pattern!r} not found in reference deck"
    return m.group(1).lower()


def _models_flags(line):
    """Extract bare flags from a MODELS statement (skip key=value pairs)."""
    tokens = line.split()[1:]
    return {t for t in tokens if "=" not in t}


class TestMosfetGolden:
    def test_models_flags_match_reference(self):
        ref = _reference_deck_line(
            _reference_text("rappture_mosfet.xml"), r"MODELS\s")
        deck = create_mosfet().generate_deck().lower()
        gen = next(l for l in deck.splitlines() if l.startswith("models"))
        missing = _models_flags(ref) - _models_flags(gen)
        assert not missing, f"reference MODELS flags missing: {missing}"

    def test_silicon_material_matches_reference(self):
        text = _reference_text("rappture_mosfet.xml")
        # reference: In.Model=klaassen MUn=1400 VSATn=1.03e+07
        assert re.search(r"in\.model=klaassen", text, re.I)
        deck = create_mosfet().generate_deck().lower()
        assert "in.mod=klaassen" in deck
        assert "mun=1400" in deck
        assert re.search(r"vsatn=1\.03", deck)

    def test_doping_polarity_matches_reference(self):
        """Reference: p substrate/channel, n+ S/D for the NMOS tool."""
        deck = create_mosfet().generate_deck().lower()
        assert re.search(r"dop uniform n\.type conc=2\.0+e20 reg=2,3", deck)
        assert re.search(r"dop uniform p\.type .*reg=4", deck)


class TestMesfetGolden:
    def test_models_flags_match_reference(self):
        ref = _reference_deck_line(
            _reference_text("rappture_mesfet.xml"), r"MODELS\s")
        deck = create_mesfet().generate_deck().lower()
        gen = next(l for l in deck.splitlines() if l.startswith("models"))
        # reference: MODELS TEMP=300 bgn conmob fldmob st=fermi print=true
        missing = {f for f in _models_flags(ref) if f != "print=true"} \
            - _models_flags(gen)
        assert not missing, f"reference MODELS flags missing: {missing}"
        assert "statistics=fermi" in gen or "st=fermi" in gen

    def test_caughey_material_matches_reference(self):
        text = _reference_text("rappture_mesfet.xml")
        assert re.search(r"en\.model=caughey", text, re.I)
        deck = create_mesfet().generate_deck().lower()
        assert "en.mod=caughey" in deck

    def test_semi_insulating_substrate_matches_reference(self):
        text = _reference_text("rappture_mesfet.xml")
        # reference: dop reg=1 n.type conc=1e+10 uniform
        assert re.search(r"dop\s+reg=1\s+n\.type\s+conc=1e\+?10", text, re.I)
        deck = create_mesfet().generate_deck().lower()
        sub = next(l for l in deck.splitlines()
                   if l.startswith("dop") and "reg=1" in l)
        assert "n.type" in sub and "e10" in sub


class TestMosCapGolden:
    def test_models_flags_match_reference(self):
        ref = _reference_deck_line(
            _reference_text("rappture_moscap.xml"), r"models\s")
        deck = create_mos_capacitor().generate_deck().lower()
        gen = next(l for l in deck.splitlines() if l.startswith("models"))
        missing = _models_flags(ref) - _models_flags(gen)
        assert not missing, f"reference MODELS flags missing: {missing}"

    def test_lifetimes_match_reference(self):
        text = _reference_text("rappture_moscap.xml")
        # reference: material name=silicon taun0=1e-09 taup0=1e-09
        assert re.search(r"taun0=1e-09", text, re.I)
        deck = create_mos_capacitor().generate_deck().lower()
        assert "taun0=1e-09" in deck
        assert "taup0=1e-09" in deck

    def test_contact_structure_matches_reference(self):
        """Reference: contact all neutral, then the gate override."""
        deck = create_mos_capacitor().generate_deck().lower()
        lines = [l for l in deck.splitlines() if l.startswith("contact")]
        assert lines[0] == "contact all neutral"
        assert any("polysilicon" in l and "num=1" in l for l in lines[1:])


class TestBjtGolden:
    def test_models_flags_match_reference(self):
        ref = _reference_deck_line(
            _reference_text("rappture_bjt.xml"), r"models\s")
        deck = create_bjt().generate_deck().lower()
        gen = next(l for l in deck.splitlines() if l.startswith("models"))
        # reference: models srh conmob fldmob auger temperature=300
        missing = _models_flags(ref) - _models_flags(gen)
        # bgn is a deliberate addition (1e20 emitter needs it); nothing
        # from the reference may be missing.
        assert not missing, f"reference MODELS flags missing: {missing}"

    def test_surf_rec_contacts_match_reference(self):
        text = _reference_text("rappture_bjt.xml")
        # reference blocks electron recombination at the base contact
        assert re.search(r"contact\s+num=2\s+n\.surf\.rec\s+p\.surf\.rec"
                         r"\s+vsurfn=0", text, re.I)
        deck = create_bjt().generate_deck().lower()
        base = next(l for l in deck.splitlines()
                    if l.startswith("contact") and "num=2" in l)
        assert "n.surf.rec" in base and "vsurfn=0" in base

    def test_per_region_materials_match_reference(self):
        text = _reference_text("rappture_bjt.xml")
        # reference: Material name=Bmat DEF=silicon taun0=1e-06
        assert re.search(r"name=bmat\s+def=silicon\s+taun0=1e-06", text, re.I)
        deck = create_bjt().generate_deck().lower()
        assert re.search(r"material name=bmat def=silicon .*taun0=1e-06", deck)
