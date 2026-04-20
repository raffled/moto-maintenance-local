"""
Unit tests for ingestion/parse.py pure functions.
No PDF file or external dependencies required.
"""

import pytest
from ingestion.parse import _clean_text, _extract_torque_specs


# ---------------------------------------------------------------------------
# _clean_text — inline ref expansion
# ---------------------------------------------------------------------------

class TestInlineRefExpansion:
    def test_lowercase_word_plus_digit(self):
        assert "screws (1)" in _clean_text("tighten screws1 firmly")

    def test_lowercase_word_plus_two_digits(self):
        assert "bolt (12)" in _clean_text("remove bolt12 carefully")

    def test_preposition_plus_uppercase(self):
        assert "of (A)" in _clean_text("placed at a distance ofA from center")

    def test_word_plus_two_uppercase(self):
        assert "position (AB)" in _clean_text("set to positionAB")

    def test_bolt_spec_unchanged(self):
        # M8, M10 end in uppercase — must not be expanded
        result = _clean_text("torque M8 to 20 Nm and M10 to 40 Nm")
        assert "M8" in result
        assert "M10" in result
        assert "(8)" not in result
        assert "(10)" not in result

    def test_all_caps_word_unchanged(self):
        result = _clean_text("WARNING do not exceed torque")
        assert "WARNING" in result
        assert "(G)" not in result

    def test_multiple_refs_on_same_line(self):
        result = _clean_text("Remove screws1 and bracket2 then lift coverA")
        assert "screws (1)" in result
        assert "bracket (2)" in result
        assert "cover (A)" in result

    def test_no_refs_unchanged(self):
        text = "Torque the bolt to specification."
        assert _clean_text(text) == text


class TestCleanTextLicense:
    def test_license_watermark_removed(self):
        text = (
            "Some manual text\n"
            "Lizenziert für | Licensed for: Douglas Raffle, raffled@gmail.com, 069969\n"
            "More manual text"
        )
        result = _clean_text(text)
        assert "Licensed for" not in result
        assert "raffled@gmail.com" not in result
        assert "Some manual text" in result
        assert "More manual text" in result

    def test_blank_line_normalization(self):
        text = "line one\n\n\n\nline two\n\n\n\nline three"
        result = _clean_text(text)
        assert "\n\n\n" not in result


# ---------------------------------------------------------------------------
# _extract_torque_specs
# ---------------------------------------------------------------------------

class TestExtractTorqueSpecs:
    def _spec_text(self, desc, bolt, nm, ftlbf, note=""):
        lines = [desc, f"{bolt} {nm} Nm", f"({ftlbf} ft·lbf)"]
        if note:
            lines.append(note)
        return "\n".join(lines)

    def test_basic_spec(self):
        text = self._spec_text("Screw, handlebar mount", "M10", "40", "29.5")
        specs = _extract_torque_specs(text)
        assert len(specs) == 1
        assert specs[0].bolt == "M10"
        assert specs[0].nm == 40.0
        assert specs[0].ftlbf == 29.5
        assert specs[0].description == "Screw, handlebar mount"

    def test_spec_with_loctite_note(self):
        text = self._spec_text(
            "Screw, handlebar mount", "M10", "40", "29.5", "Loctite® 243"
        )
        specs = _extract_torque_specs(text)
        assert specs[0].note == "Loctite® 243"

    def test_spec_without_note(self):
        text = self._spec_text("Rear brake lever stop nut", "M8", "20", "14.8")
        specs = _extract_torque_specs(text)
        assert specs[0].note == ""

    def test_multiple_specs(self):
        # Specs in the manual are separated by procedural prose, not adjacent
        text = (
            self._spec_text("Handlebar mount", "M10", "40", "29.5")
            + "\n‒ Tighten in a cross pattern.\n"
            + self._spec_text("Clamp screw", "M8", "20", "14.8")
        )
        specs = _extract_torque_specs(text)
        assert len(specs) == 2
        assert {s.bolt for s in specs} == {"M10", "M8"}

    def test_bolt_with_thread_pitch(self):
        text = self._spec_text("Oil drain plug", "M12x1.5", "25", "18.4")
        specs = _extract_torque_specs(text)
        assert specs[0].bolt == "M12x1.5"

    def test_decimal_nm(self):
        text = self._spec_text("Screw, throttle grip", "M6", "5.0", "3.7")
        specs = _extract_torque_specs(text)
        assert specs[0].nm == 5.0

    def test_no_specs_returns_empty(self):
        assert _extract_torque_specs("No torque specs here.") == []

    def test_uses_middle_dot_variant(self):
        # Manual uses both ft⋅lbf and ft·lbf
        text = "Pivot bolt\nM10 40 Nm\n(29.5 ft⋅lbf)"
        specs = _extract_torque_specs(text)
        assert len(specs) == 1
        assert specs[0].nm == 40.0
