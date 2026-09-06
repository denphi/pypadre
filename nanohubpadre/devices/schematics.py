"""
SVG device cross-section schematics.

Generates SVG representations of semiconductor device structures based on
the actual geometry, doping, and electrode parameters passed by the user.
"""

import math

# Color scheme
COLORS = {
    "n_light": "#B3D9FF",    # N-type silicon (light)
    "p_light": "#FFB3B3",    # P-type silicon (light)
    "n_heavy": "#6699CC",    # N+ heavily doped
    "p_heavy": "#CC6666",    # P+ heavily doped
    "oxide": "#FFFFCC",      # SiO2 / insulator
    "metal": "#999999",      # Metal electrode
    "schottky": "#DAA520",   # Schottky contact (gold)
    "border": "#333333",     # Region borders
    "dim_line": "#666666",   # Dimension lines
    "text": "#222222",       # Text color
    "bg": "#FFFFFF",         # Background
}


def _fmt_doping(conc):
    """Format doping concentration as e.g. '1e17'."""
    exp = int(math.floor(math.log10(abs(conc))))
    mantissa = conc / (10 ** exp)
    if abs(mantissa - 1.0) < 0.01:
        return f"10<tspan baseline-shift='super' font-size='10'>{exp}</tspan>"
    return f"{mantissa:.0f}&times;10<tspan baseline-shift='super' font-size='10'>{exp}</tspan>"


def _fmt_dim(val, unit="um"):
    """Format a dimension value."""
    if val >= 1:
        return f"{val:g} {unit}"
    return f"{val:g} {unit}"


def _svg_rect(x, y, w, h, fill, stroke=None, stroke_width=1, rx=0):
    stroke_attr = f' stroke="{stroke}" stroke-width="{stroke_width}"' if stroke else ' stroke="none"'
    rx_attr = f' rx="{rx}"' if rx else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}"{stroke_attr}{rx_attr}/>'


def _svg_text(x, y, text, size=12, anchor="middle", color=None, bold=False, data_attrs=None):
    color = color or COLORS["text"]
    weight = ' font-weight="bold"' if bold else ""
    extra = ""
    if data_attrs:
        for k, v in data_attrs.items():
            extra += f' {k}="{v}"'
    return f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" fill="{color}" text-anchor="{anchor}"{weight}{extra}>{text}</text>'


def _svg_param_label(x, y, text, param_name, value, size=10, color=None, anchor="middle"):
    """Render a clickable/editable parameter label for interactive mode."""
    color = color or COLORS["dim_line"]
    return (
        f'<g class="param-label" data-param="{param_name}" data-value="{value}" '
        f'style="cursor:pointer">'
        f'<rect x="{x - 30}" y="{y - size - 2}" width="60" height="{size + 6}" '
        f'fill="rgba(255,255,255,0.7)" rx="3" stroke="{color}" stroke-width="0.5"/>'
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" '
        f'fill="{color}" text-anchor="{anchor}" text-decoration="underline dotted">{text}</text>'
        f'</g>'
    )


def _svg_line(x1, y1, x2, y2, color=None, width=1, dash=False):
    color = color or COLORS["border"]
    dash_attr = ' stroke-dasharray="4,3"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{dash_attr}/>'


def _svg_arrow_h(x1, x2, y, color=None, label="", param_name=None, value=None):
    """Horizontal dimension arrow with label."""
    color = color or COLORS["dim_line"]
    parts = []
    parts.append(_svg_line(x1, y, x2, y, color=color, width=1))
    # Arrowheads
    parts.append(f'<polygon points="{x1},{y} {x1+5},{y-3} {x1+5},{y+3}" fill="{color}"/>')
    parts.append(f'<polygon points="{x2},{y} {x2-5},{y-3} {x2-5},{y+3}" fill="{color}"/>')
    if label:
        mid = (x1 + x2) / 2
        if param_name is not None:
            parts.append(_svg_param_label(mid, y - 5, label, param_name, value, size=10, color=color))
        else:
            parts.append(_svg_text(mid, y - 5, label, size=10, color=color))
    return "\n".join(parts)


def _svg_arrow_v(x, y1, y2, color=None, label="", side="right", param_name=None, value=None):
    """Vertical dimension arrow with label."""
    color = color or COLORS["dim_line"]
    parts = []
    parts.append(_svg_line(x, y1, x, y2, color=color, width=1))
    # Arrowheads
    parts.append(f'<polygon points="{x},{y1} {x-3},{y1+5} {x+3},{y1+5}" fill="{color}"/>')
    parts.append(f'<polygon points="{x},{y2} {x-3},{y2-5} {x+3},{y2-5}" fill="{color}"/>')
    if label:
        mid = (y1 + y2) / 2
        offset = 8 if side == "right" else -8
        anchor = "start" if side == "right" else "end"
        if param_name is not None:
            parts.append(_svg_param_label(x + offset, mid + 4, label, param_name, value, size=10, color=color, anchor=anchor))
        else:
            parts.append(_svg_text(x + offset, mid + 4, label, size=10, color=color, anchor=anchor))
    return "\n".join(parts)


def _svg_electrode(x, y, w, h, label, color=None):
    """Draw an electrode with label."""
    color = color or COLORS["metal"]
    parts = []
    parts.append(_svg_rect(x, y, w, h, fill=color, stroke=COLORS["border"]))
    parts.append(_svg_text(x + w / 2, y + h / 2 + 4, label, size=11, bold=True, color="#FFFFFF"))
    return "\n".join(parts)


def _wrap_svg(content, width=600, height=350):
    """Wrap content in SVG element."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" style="background: {COLORS["bg"]}">\n'
        f'{content}\n'
        f'</svg>'
    )


class DeviceSchematic:
    """Wrapper for SVG device schematics with Jupyter display support."""

    def __init__(self, svg_string):
        self._svg = svg_string

    def _repr_svg_(self):
        return self._svg

    def _repr_html_(self):
        return self._svg

    def __str__(self):
        return self._svg

    def save(self, filepath):
        """Save SVG to file."""
        with open(filepath, "w") as f:
            f.write(self._svg)


# ─────────────────────────────────────────────────────────
# Device-specific drawing functions
# ─────────────────────────────────────────────────────────

def draw_pn_diode(length=1.0, width=1.0, junction_position=0.5,
                  p_doping=1e17, n_doping=1e17, interactive=False, **kwargs):
    """Draw PN diode cross-section based on user parameters."""
    # Layout constants
    margin = 60
    elec_h = 22
    dw = 480  # drawing width
    dh = 180  # drawing height (device body)
    top = margin + elec_h + 5
    left = margin

    junction_frac = junction_position
    jx = left + dw * junction_frac

    parts = []

    # Title
    parts.append(_svg_text(left + dw / 2, 20, "PN Junction Diode", size=16, bold=True))

    # P-type region (left of junction)
    parts.append(_svg_rect(left, top, jx - left, dh, fill=COLORS["p_light"], stroke=COLORS["border"]))
    parts.append(_svg_text(left + (jx - left) / 2, top + dh / 2 - 10, "P", size=20, bold=True, color="#993333"))
    if interactive:
        parts.append(_svg_param_label(left + (jx - left) / 2, top + dh / 2 + 15,
                                      _fmt_doping(p_doping), "p_doping", p_doping, size=11, color="#993333"))
    else:
        parts.append(_svg_text(left + (jx - left) / 2, top + dh / 2 + 15,
                               _fmt_doping(p_doping), size=11, color="#993333"))

    # N-type region (right of junction)
    rw = dw - (jx - left)
    parts.append(_svg_rect(jx, top, rw, dh, fill=COLORS["n_light"], stroke=COLORS["border"]))
    parts.append(_svg_text(jx + rw / 2, top + dh / 2 - 10, "N", size=20, bold=True, color="#336699"))
    if interactive:
        parts.append(_svg_param_label(jx + rw / 2, top + dh / 2 + 15,
                                      _fmt_doping(n_doping), "n_doping", n_doping, size=11, color="#336699"))
    else:
        parts.append(_svg_text(jx + rw / 2, top + dh / 2 + 15,
                               _fmt_doping(n_doping), size=11, color="#336699"))

    # Junction line
    parts.append(_svg_line(jx, top, jx, top + dh, color="#FF0000", width=2, dash=True))
    parts.append(_svg_text(jx, top + dh + 15, "junction", size=10, color="#FF0000"))

    # Electrodes
    parts.append(_svg_electrode(left, top - elec_h, 60, elec_h, "Anode"))
    parts.append(_svg_electrode(left + dw - 60, top - elec_h, 60, elec_h, "Cathode"))

    # Dimensions
    _ip = interactive
    parts.append(_svg_arrow_h(left, left + dw, top + dh + 35, label=_fmt_dim(length),
                              param_name="length" if _ip else None, value=length))
    parts.append(_svg_arrow_v(left + dw + 15, top, top + dh, label=_fmt_dim(width), side="right",
                              param_name="width" if _ip else None, value=width))
    parts.append(_svg_arrow_h(left, jx, top + dh + 55, label=_fmt_dim(junction_position * length),
                              param_name="junction_position" if _ip else None, value=junction_position))

    return _wrap_svg("\n".join(parts), width=620, height=top + dh + 80)



def draw_nin_diode(length=2.0, width=1.0, junction_position=0.3,
                   intrinsic_width=0.8, contact_doping=1e18,
                   intrinsic_doping=1e14, device_type="nin",
                   interactive=False, **kwargs):
    """Draw NIN/PIP isotype diode cross-section based on user parameters."""
    is_n = str(device_type).lower() == "nin"
    label = "N" if is_n else "P"
    fill = COLORS["n_light"] if is_n else COLORS["p_light"]
    color = "#336699" if is_n else "#993333"

    margin = 60
    elec_h = 22
    dw = 480
    dh = 180
    top = margin + elec_h + 5
    left = margin

    f1 = junction_position
    f2 = min(1.0, junction_position + intrinsic_width / max(length, 1e-12))
    x1 = left + dw * f1
    x2 = left + dw * f2

    parts = []
    parts.append(_svg_text(left + dw / 2, 20,
                           f"{label}I{label} Isotype Diode "
                           f"(no junction - symmetric I-V)", size=16, bold=True))

    # Heavily doped end regions
    for a, b in ((left, x1), (x2, left + dw)):
        parts.append(_svg_rect(a, top, b - a, dh, fill=fill, stroke=COLORS["border"]))
        parts.append(_svg_text((a + b) / 2, top + dh / 2 - 10, label + "+",
                               size=20, bold=True, color=color))
        txt = _fmt_doping(contact_doping)
        if interactive:
            parts.append(_svg_param_label((a + b) / 2, top + dh / 2 + 15, txt,
                                          "contact_doping", contact_doping,
                                          size=11, color=color))
        else:
            parts.append(_svg_text((a + b) / 2, top + dh / 2 + 15, txt,
                                   size=11, color=color))

    # Lightly doped barrier
    parts.append(_svg_rect(x1, top, x2 - x1, dh, fill="#F2F2F2",
                           stroke=COLORS["border"]))
    parts.append(_svg_text((x1 + x2) / 2, top + dh / 2 - 10,
                           label + "-", size=20, bold=True, color="#666666"))
    txt = _fmt_doping(intrinsic_doping)
    if interactive:
        parts.append(_svg_param_label((x1 + x2) / 2, top + dh / 2 + 15, txt,
                                      "intrinsic_doping", intrinsic_doping,
                                      size=11, color="#666666"))
    else:
        parts.append(_svg_text((x1 + x2) / 2, top + dh / 2 + 15, txt,
                               size=11, color="#666666"))

    # Isotype interfaces (not junctions - dashed grey, not red)
    for x in (x1, x2):
        parts.append(_svg_line(x, top, x, top + dh, color="#888888",
                               width=2, dash=True))
    parts.append(_svg_text((x1 + x2) / 2, top + dh + 15,
                           "isotype interfaces (no depletion region)",
                           size=10, color="#888888"))

    parts.append(_svg_electrode(left, top - elec_h, 60, elec_h, "Contact 1"))
    parts.append(_svg_electrode(left + dw - 60, top - elec_h, 60, elec_h, "Contact 2"))

    _ip = interactive
    parts.append(_svg_arrow_h(left, left + dw, top + dh + 35, label=_fmt_dim(length),
                              param_name="length" if _ip else None, value=length))
    parts.append(_svg_arrow_v(left + dw + 15, top, top + dh, label=_fmt_dim(width),
                              side="right", param_name="width" if _ip else None,
                              value=width))
    parts.append(_svg_arrow_h(x1, x2, top + dh + 55, label=_fmt_dim(intrinsic_width),
                              param_name="intrinsic_width" if _ip else None,
                              value=intrinsic_width))

    return _wrap_svg("\n".join(parts), width=620, height=top + dh + 80)


def draw_mosfet(channel_length=0.15, gate_oxide_thickness=0.002,
                junction_depth=0.02, device_width=0.25, device_depth=0.05,
                channel_doping=1e18, substrate_doping=5e16,
                source_drain_doping=2e20, device_type="nmos", interactive=False, **kwargs):
    """Draw MOSFET cross-section based on user parameters."""
    is_nmos = device_type.lower() == "nmos"
    margin = 60
    elec_h = 22
    dw = 480
    total_h = device_depth + junction_depth + gate_oxide_thickness
    # Vertical scale
    dh_total = 220
    ox_frac = gate_oxide_thickness / total_h
    jd_frac = junction_depth / total_h
    sub_frac = device_depth / total_h
    sd_w_frac = (device_width - channel_length) / 2 / device_width

    top = margin + elec_h + 5
    left = margin

    ox_h = max(15, dh_total * ox_frac)
    jd_h = max(25, dh_total * jd_frac)
    sub_h = dh_total - ox_h - jd_h
    sd_w = dw * sd_w_frac
    ch_w = dw - 2 * sd_w

    # Doping label colors
    sd_type = "N+" if is_nmos else "P+"
    sd_color_fill = COLORS["n_heavy"] if is_nmos else COLORS["p_heavy"]
    sd_text_color = "#336699" if is_nmos else "#993333"
    ch_type = "P" if is_nmos else "N"
    ch_fill = COLORS["p_light"] if is_nmos else COLORS["n_light"]
    ch_text_color = "#993333" if is_nmos else "#336699"
    sub_fill = ch_fill

    parts = []
    parts.append(_svg_text(left + dw / 2, 20,
                           f"{'NMOS' if is_nmos else 'PMOS'} MOSFET", size=16, bold=True))

    # Gate oxide (across channel only)
    parts.append(_svg_rect(left + sd_w, top, ch_w, ox_h, fill=COLORS["oxide"], stroke=COLORS["border"]))
    parts.append(_svg_text(left + sd_w + ch_w / 2, top + ox_h / 2 + 4, "SiO2", size=9, color="#999900"))

    # Source region
    parts.append(_svg_rect(left, top + ox_h, sd_w, jd_h, fill=sd_color_fill, stroke=COLORS["border"]))
    parts.append(_svg_text(left + sd_w / 2, top + ox_h + jd_h / 2 - 5, sd_type, size=14, bold=True, color=sd_text_color))
    parts.append(_svg_text(left + sd_w / 2, top + ox_h + jd_h / 2 + 12, _fmt_doping(source_drain_doping), size=9, color=sd_text_color))

    # Drain region
    parts.append(_svg_rect(left + sd_w + ch_w, top + ox_h, sd_w, jd_h, fill=sd_color_fill, stroke=COLORS["border"]))
    parts.append(_svg_text(left + sd_w + ch_w + sd_w / 2, top + ox_h + jd_h / 2 - 5, sd_type, size=14, bold=True, color=sd_text_color))
    parts.append(_svg_text(left + sd_w + ch_w + sd_w / 2, top + ox_h + jd_h / 2 + 12, _fmt_doping(source_drain_doping), size=9, color=sd_text_color))

    # Channel region
    parts.append(_svg_rect(left + sd_w, top + ox_h, ch_w, jd_h, fill=ch_fill, stroke=COLORS["border"]))
    parts.append(_svg_text(left + sd_w + ch_w / 2, top + ox_h + jd_h / 2 - 5, ch_type, size=14, bold=True, color=ch_text_color))

    # Substrate
    parts.append(_svg_rect(left, top + ox_h + jd_h, dw, sub_h, fill=sub_fill, stroke=COLORS["border"]))
    parts.append(_svg_text(left + dw / 2, top + ox_h + jd_h + sub_h / 2 - 5,
                           f"{ch_type}-substrate", size=14, bold=True, color=ch_text_color))
    if interactive:
        parts.append(_svg_param_label(left + dw / 2, top + ox_h + jd_h + sub_h / 2 + 12,
                                      _fmt_doping(substrate_doping), "substrate_doping", substrate_doping,
                                      size=9, color=ch_text_color))
        parts.append(_svg_param_label(left + sd_w + ch_w / 2, top + ox_h + jd_h / 2 + 12,
                                      _fmt_doping(channel_doping), "channel_doping", channel_doping,
                                      size=9, color=ch_text_color))
    else:
        parts.append(_svg_text(left + dw / 2, top + ox_h + jd_h + sub_h / 2 + 12,
                               _fmt_doping(substrate_doping), size=9, color=ch_text_color))
        parts.append(_svg_text(left + sd_w + ch_w / 2, top + ox_h + jd_h / 2 + 12,
                               _fmt_doping(channel_doping), size=9, color=ch_text_color))

    # Filler oxide over S/D (visual only, thin)
    parts.append(_svg_rect(left, top, sd_w, ox_h, fill=COLORS["oxide"], stroke=COLORS["border"]))
    parts.append(_svg_rect(left + sd_w + ch_w, top, sd_w, ox_h, fill=COLORS["oxide"], stroke=COLORS["border"]))

    # Electrodes
    parts.append(_svg_electrode(left, top - elec_h, sd_w * 0.8, elec_h, "Source"))
    parts.append(_svg_electrode(left + sd_w + ch_w + sd_w * 0.2, top - elec_h, sd_w * 0.8, elec_h, "Drain"))
    parts.append(_svg_electrode(left + sd_w, top - elec_h, ch_w, elec_h, "Gate"))
    # Substrate contact (bottom)
    parts.append(_svg_electrode(left + dw * 0.3, top + ox_h + jd_h + sub_h, dw * 0.4, elec_h, "Substrate"))

    # Dimensions
    _ip = interactive
    dh_total_actual = ox_h + jd_h + sub_h
    parts.append(_svg_arrow_h(left, left + dw, top + dh_total_actual + elec_h + 20, label=_fmt_dim(device_width),
                              param_name="device_width" if _ip else None, value=device_width))
    parts.append(_svg_arrow_h(left + sd_w, left + sd_w + ch_w, top - elec_h - 15, label=_fmt_dim(channel_length),
                              param_name="channel_length" if _ip else None, value=channel_length))
    parts.append(_svg_arrow_v(left + dw + 15, top, top + ox_h, label=_fmt_dim(gate_oxide_thickness), side="right",
                              param_name="gate_oxide_thickness" if _ip else None, value=gate_oxide_thickness))
    parts.append(_svg_arrow_v(left + dw + 15, top + ox_h, top + ox_h + jd_h, label=_fmt_dim(junction_depth), side="right",
                              param_name="junction_depth" if _ip else None, value=junction_depth))

    total_svg_h = top + dh_total_actual + elec_h + 50
    return _wrap_svg("\n".join(parts), width=620, height=total_svg_h)


def draw_bjt(emitter_width=1.0, base_width=0.5, collector_width=2.0,
             device_depth=1.0, emitter_doping=1e20, base_doping=1e17,
             collector_doping=1e16, device_type="npn", interactive=False, **kwargs):
    """Draw BJT cross-section based on user parameters."""
    is_npn = device_type.lower() == "npn"
    margin = 60
    elec_h = 22
    dw = 480
    dh = 180
    top = margin + elec_h + 5
    left = margin

    total_w = emitter_width + base_width + collector_width
    ew = dw * emitter_width / total_w
    bw = dw * base_width / total_w
    cw = dw * collector_width / total_w

    # Colors based on NPN vs PNP
    if is_npn:
        e_fill, e_type, e_tc = COLORS["n_heavy"], "N+", "#336699"
        b_fill, b_type, b_tc = COLORS["p_light"], "P", "#993333"
        c_fill, c_type, c_tc = COLORS["n_light"], "N", "#336699"
    else:
        e_fill, e_type, e_tc = COLORS["p_heavy"], "P+", "#993333"
        b_fill, b_type, b_tc = COLORS["n_light"], "N", "#336699"
        c_fill, c_type, c_tc = COLORS["p_light"], "P", "#993333"

    parts = []
    parts.append(_svg_text(left + dw / 2, 20,
                           f"{'NPN' if is_npn else 'PNP'} BJT", size=16, bold=True))

    # Emitter
    parts.append(_svg_rect(left, top, ew, dh, fill=e_fill, stroke=COLORS["border"]))
    parts.append(_svg_text(left + ew / 2, top + dh / 2 - 10, e_type, size=18, bold=True, color=e_tc))
    parts.append(_svg_text(left + ew / 2, top + dh / 2 + 10, "Emitter", size=10, color=e_tc))
    if interactive:
        parts.append(_svg_param_label(left + ew / 2, top + dh / 2 + 25, _fmt_doping(emitter_doping), "emitter_doping", emitter_doping, size=9, color=e_tc))
    else:
        parts.append(_svg_text(left + ew / 2, top + dh / 2 + 25, _fmt_doping(emitter_doping), size=9, color=e_tc))

    # Base
    parts.append(_svg_rect(left + ew, top, bw, dh, fill=b_fill, stroke=COLORS["border"]))
    parts.append(_svg_text(left + ew + bw / 2, top + dh / 2 - 10, b_type, size=18, bold=True, color=b_tc))
    parts.append(_svg_text(left + ew + bw / 2, top + dh / 2 + 10, "Base", size=10, color=b_tc))
    if interactive:
        parts.append(_svg_param_label(left + ew + bw / 2, top + dh / 2 + 25, _fmt_doping(base_doping), "base_doping", base_doping, size=9, color=b_tc))
    else:
        parts.append(_svg_text(left + ew + bw / 2, top + dh / 2 + 25, _fmt_doping(base_doping), size=9, color=b_tc))

    # Collector
    parts.append(_svg_rect(left + ew + bw, top, cw, dh, fill=c_fill, stroke=COLORS["border"]))
    parts.append(_svg_text(left + ew + bw + cw / 2, top + dh / 2 - 10, c_type, size=18, bold=True, color=c_tc))
    parts.append(_svg_text(left + ew + bw + cw / 2, top + dh / 2 + 10, "Collector", size=10, color=c_tc))
    if interactive:
        parts.append(_svg_param_label(left + ew + bw + cw / 2, top + dh / 2 + 25, _fmt_doping(collector_doping), "collector_doping", collector_doping, size=9, color=c_tc))
    else:
        parts.append(_svg_text(left + ew + bw + cw / 2, top + dh / 2 + 25, _fmt_doping(collector_doping), size=9, color=c_tc))

    # Junction lines
    parts.append(_svg_line(left + ew, top, left + ew, top + dh, color="#FF0000", width=2, dash=True))
    parts.append(_svg_line(left + ew + bw, top, left + ew + bw, top + dh, color="#FF0000", width=2, dash=True))

    # Electrodes
    parts.append(_svg_electrode(left, top - elec_h, min(ew, 70), elec_h, "E"))
    parts.append(_svg_electrode(left + ew, top + dh, bw, elec_h, "B"))
    parts.append(_svg_electrode(left + dw - min(cw, 70), top - elec_h, min(cw, 70), elec_h, "C"))

    # Dimensions
    _ip = interactive
    parts.append(_svg_arrow_h(left, left + ew, top + dh + elec_h + 15, label=_fmt_dim(emitter_width),
                              param_name="emitter_width" if _ip else None, value=emitter_width))
    parts.append(_svg_arrow_h(left + ew, left + ew + bw, top + dh + elec_h + 15, label=_fmt_dim(base_width),
                              param_name="base_width" if _ip else None, value=base_width))
    parts.append(_svg_arrow_h(left + ew + bw, left + dw, top + dh + elec_h + 15, label=_fmt_dim(collector_width),
                              param_name="collector_width" if _ip else None, value=collector_width))
    parts.append(_svg_arrow_v(left + dw + 15, top, top + dh, label=_fmt_dim(device_depth), side="right",
                              param_name="device_depth" if _ip else None, value=device_depth))

    return _wrap_svg("\n".join(parts), width=620, height=top + dh + elec_h + 45)


def draw_mesfet(channel_length=0.2, gate_length=0.2, device_width=0.6,
                channel_depth=0.2, substrate_depth=0.8,
                channel_doping=1e17, substrate_doping=1e10,
                contact_doping=1e20, device_type="n",
                gate_workfunction=4.87, interactive=False, **kwargs):
    """Draw MESFET cross-section based on user parameters."""
    is_n = device_type.lower() == "n"
    margin = 60
    elec_h = 22
    dw = 480
    total_depth = substrate_depth + channel_depth
    top = margin + elec_h + 5
    left = margin

    # Vertical proportions
    dh_total = 200
    ch_h = max(30, dh_total * channel_depth / total_depth)
    sub_h = dh_total - ch_h

    # Horizontal proportions
    src_w = dw * channel_length / device_width
    drn_w = src_w
    gate_w = dw * gate_length / device_width
    # Gate centered between source and drain regions
    gap_total = dw - src_w - drn_w
    gate_start_px = src_w + (gap_total - gate_w) / 2

    ch_type = "N" if is_n else "P"
    ch_fill = COLORS["n_light"] if is_n else COLORS["p_light"]
    ch_tc = "#336699" if is_n else "#993333"
    sd_type = "N+" if is_n else "P+"
    sd_fill = COLORS["n_heavy"] if is_n else COLORS["p_heavy"]
    sd_tc = ch_tc
    sub_type = "P" if is_n else "N"
    sub_fill = COLORS["p_light"] if is_n else COLORS["n_light"]
    sub_tc = "#993333" if is_n else "#336699"

    parts = []
    parts.append(_svg_text(left + dw / 2, 20,
                           f"{'N' if is_n else 'P'}-channel MESFET", size=16, bold=True))

    # Substrate (bottom, full width)
    parts.append(_svg_rect(left, top + ch_h, dw, sub_h, fill=sub_fill, stroke=COLORS["border"]))
    parts.append(_svg_text(left + dw / 2, top + ch_h + sub_h / 2 - 5,
                           f"{sub_type}-substrate", size=14, bold=True, color=sub_tc))
    if not interactive:
        parts.append(_svg_text(left + dw / 2, top + ch_h + sub_h / 2 + 12,
                               _fmt_doping(substrate_doping), size=9, color=sub_tc))

    # Source contact region (top-left)
    parts.append(_svg_rect(left, top, src_w, ch_h, fill=sd_fill, stroke=COLORS["border"]))
    parts.append(_svg_text(left + src_w / 2, top + ch_h / 2, sd_type, size=12, bold=True, color=sd_tc))

    # Channel (top-center)
    ch_x = left + src_w
    ch_px_w = dw - src_w - drn_w
    parts.append(_svg_rect(ch_x, top, ch_px_w, ch_h, fill=ch_fill, stroke=COLORS["border"]))
    parts.append(_svg_text(ch_x + ch_px_w / 2, top + ch_h / 2 - 5,
                           f"{ch_type}-channel", size=12, bold=True, color=ch_tc))
    if not interactive:
        parts.append(_svg_text(ch_x + ch_px_w / 2, top + ch_h / 2 + 12,
                               _fmt_doping(channel_doping), size=9, color=ch_tc))

    # Drain contact region (top-right)
    parts.append(_svg_rect(left + dw - drn_w, top, drn_w, ch_h, fill=sd_fill, stroke=COLORS["border"]))
    parts.append(_svg_text(left + dw - drn_w / 2, top + ch_h / 2, sd_type, size=12, bold=True, color=sd_tc))

    # Electrodes
    parts.append(_svg_electrode(left, top - elec_h, src_w, elec_h, "Source"))
    parts.append(_svg_electrode(left + dw - drn_w, top - elec_h, drn_w, elec_h, "Drain"))
    # Schottky gate (gold color)
    parts.append(_svg_electrode(left + gate_start_px, top - elec_h, gate_w, elec_h, "Gate", color=COLORS["schottky"]))
    parts.append(_svg_text(left + gate_start_px + gate_w / 2, top - elec_h - 5,
                           f"WF={gate_workfunction}V", size=9, color=COLORS["schottky"]))

    # Dimensions
    _ip = interactive
    dh_total_actual = ch_h + sub_h
    parts.append(_svg_arrow_h(left, left + dw, top + dh_total_actual + 15, label=_fmt_dim(device_width),
                              param_name="device_width" if _ip else None, value=device_width))
    parts.append(_svg_arrow_h(left + gate_start_px, left + gate_start_px + gate_w,
                              top - elec_h - 18, label=_fmt_dim(gate_length),
                              param_name="gate_length" if _ip else None, value=gate_length))
    parts.append(_svg_arrow_v(left + dw + 15, top, top + ch_h, label=_fmt_dim(channel_depth), side="right",
                              param_name="channel_depth" if _ip else None, value=channel_depth))
    parts.append(_svg_arrow_v(left + dw + 15, top + ch_h, top + dh_total_actual,
                              label=_fmt_dim(substrate_depth), side="right",
                              param_name="substrate_depth" if _ip else None, value=substrate_depth))

    if interactive:
        parts.append(_svg_param_label(left + dw / 2, top + ch_h + sub_h / 2 + 12,
                                      _fmt_doping(substrate_doping), "substrate_doping", substrate_doping, size=9, color=sub_tc))
        parts.append(_svg_param_label(ch_x + ch_px_w / 2, top + ch_h / 2 + 12,
                                      _fmt_doping(channel_doping), "channel_doping", channel_doping, size=9, color=ch_tc))

    return _wrap_svg("\n".join(parts), width=620, height=top + dh_total_actual + 45)


def draw_mos_capacitor(oxide_thickness=0.002, silicon_thickness=5.0,
                       device_width=1.0, substrate_doping=1e16,
                       substrate_type="p", gate_type="n_poly",
                       gate_config="single", back_oxide_thickness=0.002,
                       back_gate_type="n_poly", interactive=False, **kwargs):
    """Draw MOS Capacitor cross-section based on user parameters."""
    is_double = gate_config.lower() == "double"
    margin = 60
    elec_h = 22
    dw = 300
    left = margin + 90  # center it

    total_phys = oxide_thickness + silicon_thickness + (back_oxide_thickness if is_double else 0)
    dh_total = 200
    ox_h = max(20, int(dh_total * oxide_thickness / total_phys))
    back_ox_h = max(20, int(dh_total * back_oxide_thickness / total_phys)) if is_double else 0
    si_h = dh_total - ox_h - back_ox_h

    top = margin + elec_h + 5

    is_p = substrate_type.lower() == "p"
    si_fill = COLORS["p_light"] if is_p else COLORS["n_light"]
    si_type = "P" if is_p else "N"
    si_tc = "#993333" if is_p else "#336699"

    gate_label = {"n_poly": "N+ Poly", "p_poly": "P+ Poly"}.get(gate_type, "Metal")
    back_label = {"n_poly": "N+ Poly", "p_poly": "P+ Poly"}.get(back_gate_type, "Metal")
    title_str = "Double-Gate MOS Capacitor" if is_double else "MOS Capacitor"

    parts = []
    parts.append(_svg_text(left + dw / 2, 20, title_str, size=16, bold=True))

    # Top gate electrode
    parts.append(_svg_electrode(left, top - elec_h, dw, elec_h, f"Gate 1 ({gate_label})" if is_double else f"Gate ({gate_label})"))

    # Top oxide
    parts.append(_svg_rect(left, top, dw, ox_h, fill=COLORS["oxide"], stroke=COLORS["border"]))
    parts.append(_svg_text(left + dw / 2, top + ox_h / 2 + 4, "SiO2", size=12, bold=True, color="#999900"))

    # Silicon substrate
    parts.append(_svg_rect(left, top + ox_h, dw, si_h, fill=si_fill, stroke=COLORS["border"]))
    parts.append(_svg_text(left + dw / 2, top + ox_h + si_h / 2 - 10,
                           f"{si_type}-Si", size=18, bold=True, color=si_tc))
    parts.append(_svg_text(left + dw / 2, top + ox_h + si_h / 2 + 12,
                           _fmt_doping(substrate_doping), size=11, color=si_tc))

    if is_double:
        # Bottom oxide
        back_y = top + ox_h + si_h
        parts.append(_svg_rect(left, back_y, dw, back_ox_h, fill=COLORS["oxide"], stroke=COLORS["border"]))
        parts.append(_svg_text(left + dw / 2, back_y + back_ox_h / 2 + 4, "SiO2", size=12, bold=True, color="#999900"))
        # Bottom gate electrode
        parts.append(_svg_electrode(left, back_y + back_ox_h, dw, elec_h, f"Gate 2 ({back_label})"))
        # Dimension arrows
        parts.append(_svg_arrow_v(left - 15, top, top + ox_h, label=_fmt_dim(oxide_thickness), side="left"))
        parts.append(_svg_arrow_v(left - 15, top + ox_h, top + ox_h + si_h, label=_fmt_dim(silicon_thickness), side="left"))
        parts.append(_svg_arrow_v(left - 15, top + ox_h + si_h, back_y + back_ox_h, label=_fmt_dim(back_oxide_thickness), side="left"))
        parts.append(_svg_arrow_h(left, left + dw, back_y + back_ox_h + elec_h + 15, label=_fmt_dim(device_width)))
        return _wrap_svg("\n".join(parts), width=560, height=top + ox_h + si_h + back_ox_h + elec_h + 45)
    else:
        # Back contact
        parts.append(_svg_electrode(left, top + ox_h + si_h, dw, elec_h, "Back Contact"))
        # Dimension arrows
        parts.append(_svg_arrow_v(left - 15, top, top + ox_h, label=_fmt_dim(oxide_thickness), side="left"))
        parts.append(_svg_arrow_v(left - 15, top + ox_h, top + ox_h + si_h,
                                  label=_fmt_dim(silicon_thickness), side="left"))
        parts.append(_svg_arrow_h(left, left + dw, top + ox_h + si_h + elec_h + 15,
                                  label=_fmt_dim(device_width)))
        return _wrap_svg("\n".join(parts), width=560, height=top + ox_h + si_h + elec_h + 45)


def draw_schottky_diode(length=2.0, width=1.0, doping=1e16,
                        doping_type="n", workfunction=4.8, interactive=False, **kwargs):
    """Draw Schottky Diode cross-section based on user parameters."""
    margin = 60
    elec_h = 22
    dw = 400
    dh = 200
    top = margin + elec_h + 5
    left = margin + 50

    is_n = doping_type.lower() == "n"
    si_fill = COLORS["n_light"] if is_n else COLORS["p_light"]
    si_type = "N" if is_n else "P"
    si_tc = "#336699" if is_n else "#993333"

    parts = []
    parts.append(_svg_text(left + dw / 2, 20, "Schottky Diode", size=16, bold=True))

    # Schottky contact (top)
    parts.append(_svg_electrode(left, top - elec_h, dw, elec_h, f"Schottky (WF={workfunction}V)", color=COLORS["schottky"]))

    # Semiconductor body
    parts.append(_svg_rect(left, top, dw, dh, fill=si_fill, stroke=COLORS["border"]))
    parts.append(_svg_text(left + dw / 2, top + dh / 2 - 15,
                           f"{si_type}-type Si", size=18, bold=True, color=si_tc))
    if interactive:
        parts.append(_svg_param_label(left + dw / 2, top + dh / 2 + 10,
                                      _fmt_doping(doping), "doping", doping, size=12, color=si_tc))
    else:
        parts.append(_svg_text(left + dw / 2, top + dh / 2 + 10,
                               _fmt_doping(doping), size=12, color=si_tc))

    # Ohmic contact (bottom)
    parts.append(_svg_electrode(left, top + dh, dw, elec_h, "Ohmic Contact"))

    # Dimensions
    _ip = interactive
    parts.append(_svg_arrow_v(left - 15, top, top + dh, label=_fmt_dim(width), side="left",
                              param_name="width" if _ip else None, value=width))
    parts.append(_svg_arrow_h(left, left + dw, top + dh + elec_h + 15, label=_fmt_dim(length),
                              param_name="length" if _ip else None, value=length))

    return _wrap_svg("\n".join(parts), width=580, height=top + dh + elec_h + 45)


def draw_solar_cell(emitter_depth=0.5, base_thickness=200.0, device_width=1.0,
                    emitter_doping=1e19, base_doping=1e16,
                    device_type="n_on_p",
                    front_surface_velocity=1e4, back_surface_velocity=1e7,
                    interactive=False, **kwargs):
    """Draw Solar Cell cross-section based on user parameters."""
    margin = 60
    elec_h = 22
    dw = 350
    top = margin + elec_h + 5
    left = margin + 60

    total = emitter_depth + base_thickness
    dh_total = 220
    # Emitter is usually thin — give it at least visible height
    em_frac = emitter_depth / total
    em_h = max(35, min(80, dh_total * em_frac))
    base_h = dh_total - em_h

    is_n_on_p = device_type.lower() == "n_on_p"
    if is_n_on_p:
        em_fill, em_type, em_tc = COLORS["n_heavy"], "N+", "#336699"
        base_fill, base_type, base_tc = COLORS["p_light"], "P", "#993333"
    else:
        em_fill, em_type, em_tc = COLORS["p_heavy"], "P+", "#993333"
        base_fill, base_type, base_tc = COLORS["n_light"], "N", "#336699"

    parts = []
    parts.append(_svg_text(left + dw / 2, 20,
                           f"Solar Cell ({'N-on-P' if is_n_on_p else 'P-on-N'})", size=16, bold=True))

    # Light arrows at top
    for i in range(5):
        ax = left + 40 + i * (dw - 80) / 4
        parts.append(_svg_line(ax, top - elec_h - 25, ax, top - elec_h - 5, color="#FFB800", width=2))
        parts.append(f'<polygon points="{ax},{top - elec_h - 5} {ax - 4},{top - elec_h - 12} {ax + 4},{top - elec_h - 12}" fill="#FFB800"/>')
    parts.append(_svg_text(left + dw / 2, top - elec_h - 30, "Light", size=10, color="#FFB800"))

    # Front contact
    parts.append(_svg_electrode(left, top - elec_h, dw, elec_h, "Front Contact"))

    # Emitter
    parts.append(_svg_rect(left, top, dw, em_h, fill=em_fill, stroke=COLORS["border"]))
    parts.append(_svg_text(left + dw / 2, top + em_h / 2 - 5,
                           f"{em_type} Emitter", size=14, bold=True, color=em_tc))

    # Junction line
    parts.append(_svg_line(left, top + em_h, left + dw, top + em_h, color="#FF0000", width=2, dash=True))

    # Base
    parts.append(_svg_rect(left, top + em_h, dw, base_h, fill=base_fill, stroke=COLORS["border"]))
    parts.append(_svg_text(left + dw / 2, top + em_h + base_h / 2 - 5,
                           f"{base_type} Base", size=14, bold=True, color=base_tc))

    # Back contact
    parts.append(_svg_electrode(left, top + em_h + base_h, dw, elec_h, "Back Contact"))

    # SRV annotations
    parts.append(_svg_text(left + dw + 10, top + 10,
                           f"S_front={front_surface_velocity:.0e} cm/s", size=9,
                           color="#666666", anchor="start"))
    parts.append(_svg_text(left + dw + 10, top + em_h + base_h + 10,
                           f"S_back={back_surface_velocity:.0e} cm/s", size=9,
                           color="#666666", anchor="start"))

    if interactive:
        parts.append(_svg_param_label(left + dw / 2, top + em_h / 2 + 12,
                                      _fmt_doping(emitter_doping), "emitter_doping", emitter_doping, size=10, color=em_tc))
        parts.append(_svg_param_label(left + dw / 2, top + em_h + base_h / 2 + 12,
                                      _fmt_doping(base_doping), "base_doping", base_doping, size=10, color=base_tc))
    else:
        parts.append(_svg_text(left + dw / 2, top + em_h / 2 + 12,
                               _fmt_doping(emitter_doping), size=10, color=em_tc))
        parts.append(_svg_text(left + dw / 2, top + em_h + base_h / 2 + 12,
                               _fmt_doping(base_doping), size=10, color=base_tc))

    # Dimensions
    _ip = interactive
    parts.append(_svg_arrow_v(left - 15, top, top + em_h, label=_fmt_dim(emitter_depth), side="left",
                              param_name="emitter_depth" if _ip else None, value=emitter_depth))
    parts.append(_svg_arrow_v(left - 15, top + em_h, top + em_h + base_h,
                              label=_fmt_dim(base_thickness), side="left",
                              param_name="base_thickness" if _ip else None, value=base_thickness))
    parts.append(_svg_arrow_h(left, left + dw, top + em_h + base_h + elec_h + 15,
                              label=_fmt_dim(device_width),
                              param_name="device_width" if _ip else None, value=device_width))

    return _wrap_svg("\n".join(parts), width=620, height=top + em_h + base_h + elec_h + 45)


def draw_pin_diode(length=2.0, width=1.0, p_width=0.5, i_width=1.0, n_width=0.5,
                   p_doping=1e18, i_doping=1e13, n_doping=1e18,
                   interactive=False, **kwargs):
    """Draw PiN diode cross-section with three distinct regions."""
    margin = 60
    elec_h = 22
    dw = 480
    dh = 180
    top = margin + elec_h + 5
    left = margin

    total = p_width + i_width + n_width
    if total <= 0:
        total = 1.0
    p_frac = p_width / total
    i_frac = i_width / total
    # n_frac = n_width / total  (remainder)

    p_w = dw * p_frac
    i_w = dw * i_frac
    n_w = dw - p_w - i_w

    p_x = left
    i_x = left + p_w
    n_x = i_x + i_w

    _ip = interactive
    parts = []

    # Title
    parts.append(_svg_text(left + dw / 2, 20, "PiN Diode", size=16, bold=True))

    # P region
    parts.append(_svg_rect(p_x, top, p_w, dh, fill=COLORS["p_light"], stroke=COLORS["border"]))
    parts.append(_svg_text(p_x + p_w / 2, top + dh / 2 - 10, "P", size=20, bold=True, color="#993333"))
    if _ip:
        parts.append(_svg_param_label(p_x + p_w / 2, top + dh / 2 + 15,
                                      _fmt_doping(p_doping), "p_doping", p_doping, size=10, color="#993333"))
    else:
        parts.append(_svg_text(p_x + p_w / 2, top + dh / 2 + 15,
                               _fmt_doping(p_doping), size=10, color="#993333"))

    # Intrinsic region (very light fill)
    i_fill = "#f0f4f8"
    parts.append(_svg_rect(i_x, top, i_w, dh, fill=i_fill, stroke=COLORS["border"]))
    parts.append(_svg_text(i_x + i_w / 2, top + dh / 2 - 10, "i", size=20, bold=True, color="#5a7a9a"))
    if _ip:
        parts.append(_svg_param_label(i_x + i_w / 2, top + dh / 2 + 15,
                                      _fmt_doping(i_doping), "i_doping", i_doping, size=10, color="#5a7a9a"))
    else:
        parts.append(_svg_text(i_x + i_w / 2, top + dh / 2 + 15,
                               _fmt_doping(i_doping), size=10, color="#5a7a9a"))

    # N region
    parts.append(_svg_rect(n_x, top, n_w, dh, fill=COLORS["n_light"], stroke=COLORS["border"]))
    parts.append(_svg_text(n_x + n_w / 2, top + dh / 2 - 10, "N", size=20, bold=True, color="#336699"))
    if _ip:
        parts.append(_svg_param_label(n_x + n_w / 2, top + dh / 2 + 15,
                                      _fmt_doping(n_doping), "n_doping", n_doping, size=10, color="#336699"))
    else:
        parts.append(_svg_text(n_x + n_w / 2, top + dh / 2 + 15,
                               _fmt_doping(n_doping), size=10, color="#336699"))

    # Junction lines
    parts.append(_svg_line(i_x, top, i_x, top + dh, color="#CC4444", width=2, dash=True))
    parts.append(_svg_line(n_x, top, n_x, top + dh, color="#4444CC", width=2, dash=True))

    # Electrodes
    parts.append(_svg_electrode(p_x, top - elec_h, 60, elec_h, "Anode"))
    parts.append(_svg_electrode(left + dw - 60, top - elec_h, 60, elec_h, "Cathode"))

    # Dimension arrows
    parts.append(_svg_arrow_h(left, left + dw, top + dh + 35,
                              label=_fmt_dim(length),
                              param_name="length" if _ip else None, value=length))
    parts.append(_svg_arrow_v(left + dw + 15, top, top + dh, label=_fmt_dim(width), side="right",
                              param_name="width" if _ip else None, value=width))
    parts.append(_svg_arrow_h(p_x, i_x, top + dh + 55,
                              label=_fmt_dim(p_width),
                              param_name="p_width" if _ip else None, value=p_width))
    parts.append(_svg_arrow_h(i_x, n_x, top + dh + 55,
                              label=_fmt_dim(i_width),
                              param_name="i_width" if _ip else None, value=i_width))

    return _wrap_svg("\n".join(parts), width=620, height=top + dh + 85)


# ─────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────

_DRAW_FUNCTIONS = {
    "pn_diode": draw_pn_diode,
    "nin_diode": draw_nin_diode,
    "pin_diode": draw_pin_diode,
    "mosfet": draw_mosfet,
    "bjt": draw_bjt,
    "mesfet": draw_mesfet,
    "mos_capacitor": draw_mos_capacitor,
    "schottky_diode": draw_schottky_diode,
    "solar_cell": draw_solar_cell,
}


def device_schematic(device_name, interactive=False, **kwargs):
    """Generate an SVG cross-section schematic for a device.

    Parameters
    ----------
    device_name : str
        Device name (e.g., 'pn_diode', 'mosfet', 'mesfet')
    interactive : bool
        If True, dimension/doping labels are wrapped in clickable groups with
        ``data-param`` attributes for use in web UIs. Default False.
    **kwargs
        Device geometry and doping parameters matching the factory function.
        Only geometry/doping/contact parameters affect the schematic.

    Returns
    -------
    DeviceSchematic
        SVG schematic with Jupyter display support (_repr_svg_)

    Examples
    --------
    >>> schematic = device_schematic('pn_diode', length=2.0, p_doping=1e16)
    >>> schematic  # displays inline in Jupyter
    >>> schematic.save('pn_diode.svg')
    """
    if device_name not in _DRAW_FUNCTIONS:
        available = ", ".join(_DRAW_FUNCTIONS.keys())
        raise ValueError(f"Unknown device '{device_name}'. Available: {available}")

    svg = _DRAW_FUNCTIONS[device_name](interactive=interactive, **kwargs)
    return DeviceSchematic(svg)
