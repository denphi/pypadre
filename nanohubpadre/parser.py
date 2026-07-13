"""
Parser for PADRE simulation output.

Extracts statistics, convergence data, and I-V characteristics from PADRE output.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from enum import Enum
import numpy as np


class ConvergenceStatus(Enum):
    """Convergence status for a bias point."""
    CONVERGED = "converged"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class BiasPoint:
    """Data for a single bias point solution."""
    voltages: Dict[int, float] = field(default_factory=dict)
    currents: Dict[int, Dict[str, float]] = field(default_factory=dict)
    flux: Dict[int, float] = field(default_factory=dict)
    displacement_current: Dict[int, float] = field(default_factory=dict)
    total_current: Dict[int, float] = field(default_factory=dict)
    iterations: int = 0
    cpu_time: float = 0.0
    convergence_status: ConvergenceStatus = ConvergenceStatus.UNKNOWN
    is_initial: bool = False


@dataclass
class MeshStatistics:
    """Mesh statistics from PADRE output."""
    total_grid_points: int = 0
    total_elements: int = 0
    min_grid_spacing: float = 0.0
    max_grid_spacing: float = 0.0
    spacing_ratio: float = 0.0
    obtuse_elements: int = 0
    obtuse_percentage: float = 0.0


@dataclass
class SimulationResult:
    """Complete parsed results from a PADRE simulation."""
    title: str = ""
    mesh_stats: Optional[MeshStatistics] = None
    materials: Dict[int, str] = field(default_factory=dict)
    regions_by_material: Dict[str, List[int]] = field(default_factory=dict)
    bias_points: List[BiasPoint] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    total_cpu_time: float = 0.0

    @property
    def num_bias_points(self) -> int:
        """Number of bias points solved."""
        return len(self.bias_points)

    @property
    def all_converged(self) -> bool:
        """Whether all bias points converged."""
        return all(bp.convergence_status == ConvergenceStatus.CONVERGED
                   for bp in self.bias_points)

    def get_iv_data(self, electrode: int) -> Tuple[List[float], List[float]]:
        """
        Extract I-V data for a specific electrode.

        Parameters
        ----------
        electrode : int
            Electrode number (1-indexed)

        Returns
        -------
        tuple
            (voltages, currents) lists
        """
        voltages = []
        currents = []
        for bp in self.bias_points:
            if electrode in bp.voltages and electrode in bp.total_current:
                voltages.append(bp.voltages[electrode])
                currents.append(bp.total_current[electrode])
        return np.array(voltages), np.array(currents)

    def get_transfer_characteristic(self, gate_electrode: int,
                                     drain_electrode: int) -> Tuple[List[float], List[float]]:
        """
        Extract transfer characteristic (Id vs Vg).

        Parameters
        ----------
        gate_electrode : int
            Gate electrode number
        drain_electrode : int
            Drain electrode number

        Returns
        -------
        tuple
            (gate_voltages, drain_currents) lists
        """
        vg = []
        id = []
        for bp in self.bias_points:
            if gate_electrode in bp.voltages and drain_electrode in bp.total_current:
                vg.append(bp.voltages[gate_electrode])
                id.append(abs(bp.total_current[drain_electrode]))
        return np.array(vg), np.array(id)

    def summary(self) -> str:
        """Generate a human-readable summary of the simulation results."""
        lines = []
        lines.append(f"PADRE Simulation Results")
        lines.append(f"=" * 50)

        if self.title:
            lines.append(f"Title: {self.title}")

        if self.mesh_stats:
            lines.append(f"\nMesh Statistics:")
            lines.append(f"  Grid points: {self.mesh_stats.total_grid_points}")
            lines.append(f"  Elements: {self.mesh_stats.total_elements}")
            lines.append(f"  Min spacing: {self.mesh_stats.min_grid_spacing:.4e} um")
            lines.append(f"  Max spacing: {self.mesh_stats.max_grid_spacing:.4e} um")

        lines.append(f"\nBias Points: {self.num_bias_points}")
        converged = sum(1 for bp in self.bias_points
                       if bp.convergence_status == ConvergenceStatus.CONVERGED)
        lines.append(f"  Converged: {converged}/{self.num_bias_points}")

        if self.warnings:
            lines.append(f"\nWarnings: {len(self.warnings)}")
            for w in self.warnings[:5]:  # Show first 5
                lines.append(f"  - {w}")
            if len(self.warnings) > 5:
                lines.append(f"  ... and {len(self.warnings) - 5} more")

        if self.errors:
            lines.append(f"\nErrors: {len(self.errors)}")
            for e in self.errors:
                lines.append(f"  - {e}")

        lines.append(f"\nTotal CPU time: {self.total_cpu_time:.4f} s")

        return "\n".join(lines)


class PadreOutputParser:
    """Parser for PADRE simulation output."""

    def __init__(self):
        self.result = SimulationResult()
        self._current_bias_point = None

    def parse(self, output: str) -> SimulationResult:
        """
        Parse PADRE output string.

        Parameters
        ----------
        output : str
            Complete PADRE stdout output

        Returns
        -------
        SimulationResult
            Parsed simulation results
        """
        self.result = SimulationResult()
        self._current_bias_point = None

        lines = output.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]

            # Parse title
            if line.strip().startswith('***********') and i + 1 < len(lines):
                title_line = lines[i + 1].strip()
                if title_line and not title_line.startswith('*'):
                    self.result.title = title_line
                i += 1

            # Parse mesh statistics
            elif 'Mesh statistics' in line:
                i = self._parse_mesh_stats(lines, i)

            # Parse material definitions
            elif 'Material Definitions' in line:
                i = self._parse_materials(lines, i)

            # Parse bias point solution
            elif 'Solution for bias:' in line:
                i = self._parse_bias_point(lines, i)

            # Parse warnings
            elif '** Warning' in line:
                warning = self._extract_warning(lines, i)
                if warning:
                    self.result.warnings.append(warning)

            # Parse total CPU time at end
            elif 'Total cpu time =' in line and 'bias point' not in line.lower():
                match = re.search(r'Total cpu time =\s+([\d.]+)', line)
                if match:
                    self.result.total_cpu_time = float(match.group(1))

            i += 1

        return self.result

    def _parse_mesh_stats(self, lines: List[str], start: int) -> int:
        """Parse mesh statistics section."""
        stats = MeshStatistics()
        i = start + 1

        while i < len(lines) and lines[i].strip():
            line = lines[i].strip()

            if 'Total grid points' in line:
                match = re.search(r'=\s*(\d+)', line)
                if match:
                    stats.total_grid_points = int(match.group(1))

            elif 'Total no. of elements' in line:
                match = re.search(r'=\s*(\d+)', line)
                if match:
                    stats.total_elements = int(match.group(1))

            elif 'Min grid spacing' in line:
                match = re.search(r'=\s*([\d.E+-]+)', line)
                if match:
                    stats.min_grid_spacing = float(match.group(1))

            elif 'Max grid spacing' in line:
                match = re.search(r'=\s*([\d.E+-]+).*r=\s*([\d.E+-]+)', line)
                if match:
                    stats.max_grid_spacing = float(match.group(1))
                    stats.spacing_ratio = float(match.group(2))

            elif 'Obtuse elements' in line:
                match = re.search(r'=\s*(\d+).*\(\s*([\d.]+)%\)', line)
                if match:
                    stats.obtuse_elements = int(match.group(1))
                    stats.obtuse_percentage = float(match.group(2))

            i += 1

        self.result.mesh_stats = stats
        return i

    def _parse_materials(self, lines: List[str], start: int) -> int:
        """Parse material definitions section."""
        i = start + 1

        # Skip header lines
        while i < len(lines) and ('Index' in lines[i] or '---' in lines[i] or not lines[i].strip()):
            i += 1

        # Parse material entries
        while i < len(lines) and lines[i].strip():
            line = lines[i].strip()
            match = re.match(r'(\d+)\s+(\w+)\s+([\d,]+)', line)
            if match:
                idx = int(match.group(1))
                name = match.group(2)
                regions = [int(r) for r in match.group(3).split(',')]
                self.result.materials[idx] = name
                if name not in self.result.regions_by_material:
                    self.result.regions_by_material[name] = []
                self.result.regions_by_material[name].extend(regions)
            i += 1

        return i

    def _parse_bias_point(self, lines: List[str], start: int) -> int:
        """Parse a bias point solution section."""
        bp = BiasPoint()
        i = start + 1

        # Parse voltages (V1, V2, etc.)
        while i < len(lines):
            line = lines[i]

            # Look for voltage specifications
            v_matches = re.findall(r'V(\d+)\s*=\s*([\d.E+-]+)', line)
            for electrode, voltage in v_matches:
                bp.voltages[int(electrode)] = float(voltage)

            if v_matches:
                i += 1
                continue

            # Check for initial solution
            if 'Initial solution' in line:
                bp.is_initial = True

            # Parse electrode currents table
            if 'Electrode   Voltage   Electron Current' in line:
                i = self._parse_current_table(lines, i, bp)
                continue

            # Parse flux table
            if 'Electrode          Flux' in line:
                i = self._parse_flux_table(lines, i, bp)
                continue

            # Check for convergence
            if 'Convergence criterion completely met' in line:
                bp.convergence_status = ConvergenceStatus.CONVERGED
            elif 'did not converge' in line.lower() or 'failed' in line.lower():
                bp.convergence_status = ConvergenceStatus.FAILED

            # Parse CPU time for this bias point
            if 'Total cpu time for bias point' in line:
                match = re.search(r'=\s+([\d.]+)', line)
                if match:
                    bp.cpu_time = float(match.group(1))

            # End of this bias point section
            if i + 1 < len(lines) and 'Solution for bias:' in lines[i + 1]:
                break
            if 'Caution!' in line or 'warnings cited' in line:
                break

            i += 1

        self.result.bias_points.append(bp)
        return i

    def _parse_current_table(self, lines: List[str], start: int, bp: BiasPoint) -> int:
        """Parse electrode current table."""
        i = start + 1

        # Skip header line
        while i < len(lines) and ('(Volts)' in lines[i] or '(Amps)' in lines[i]):
            i += 1

        # Parse current entries
        while i < len(lines) and lines[i].strip():
            line = lines[i].strip()
            parts = line.split()

            if len(parts) >= 5:
                try:
                    electrode = int(parts[0])
                    voltage = float(parts[1])
                    electron_current = float(parts[2])
                    hole_current = float(parts[3])
                    conduction_current = float(parts[4])

                    bp.currents[electrode] = {
                        'electron': electron_current,
                        'hole': hole_current,
                        'conduction': conduction_current
                    }
                except (ValueError, IndexError):
                    pass

            i += 1

        return i

    def _parse_flux_table(self, lines: List[str], start: int, bp: BiasPoint) -> int:
        """Parse electrode flux table."""
        i = start + 1

        # Skip header line
        while i < len(lines) and ('(Coul)' in lines[i] or '(Amps)' in lines[i]):
            i += 1

        # Parse flux entries
        while i < len(lines) and lines[i].strip():
            line = lines[i].strip()
            parts = line.split()

            if len(parts) >= 4:
                try:
                    electrode = int(parts[0])
                    flux = float(parts[1])
                    disp_current = float(parts[2])
                    total_current = float(parts[3])

                    bp.flux[electrode] = flux
                    bp.displacement_current[electrode] = disp_current
                    bp.total_current[electrode] = total_current
                except (ValueError, IndexError):
                    pass

            i += 1

        return i

    def _extract_warning(self, lines: List[str], start: int) -> Optional[str]:
        """Extract warning message."""
        warning_parts = []
        i = start

        while i < len(lines) and i < start + 5:
            line = lines[i].strip()
            if line and not line.startswith('**'):
                warning_parts.append(line)
            elif line.startswith('** Warning'):
                # Extract line number if present
                match = re.search(r'line #\s*(\d+)', line)
                if match:
                    warning_parts.append(f"Line {match.group(1)}:")
            i += 1

            # Stop at next warning or empty line after content
            if warning_parts and not line:
                break

        return ' '.join(warning_parts) if warning_parts else None


def parse_padre_output(output: str) -> SimulationResult:
    """
    Convenience function to parse PADRE output.

    Parameters
    ----------
    output : str
        Complete PADRE stdout output

    Returns
    -------
    SimulationResult
        Parsed simulation results

    Example
    -------
    >>> result = sim.run(padre_executable="padre")
    >>> parsed = parse_padre_output(result.stdout)
    >>> print(parsed.summary())
    >>> vg, id = parsed.get_transfer_characteristic(gate_electrode=3, drain_electrode=2)
    """
    parser = PadreOutputParser()
    return parser.parse(output)


@dataclass
class IVData:
    """
    Parsed I-V data from PADRE log file (ivfile).

    Units: PADRE solves a 2D cross-section, so terminal currents are in
    amperes **per micron of device depth**, scaled by the mesh ``width=``
    parameter (default 1 µm).  Multiply by the physical device depth in
    microns to obtain absolute amperes.

    Attributes
    ----------
    num_electrodes : int
        Number of electrodes in the simulation
    bias_points : List[dict]
        List of bias point data, each containing:
        - voltages: dict mapping electrode number to voltage (V)
        - currents: dict mapping electrode number to current components
          (A per µm of depth, times the mesh ``width``)
    """
    num_electrodes: int = 0
    bias_points: List[Dict] = field(default_factory=list)

    def get_voltages(self, electrode: int) -> np.ndarray:
        """Get all voltages for a specific electrode."""
        return np.array([bp['voltages'].get(electrode, 0.0) for bp in self.bias_points])

    def get_currents(self, electrode: int, component: str = 'total') -> np.ndarray:
        """
        Get currents for a specific electrode.

        Parameters
        ----------
        electrode : int
            Electrode number (1-indexed)
        component : str
            Current component: 'electron', 'hole', or 'total' (default)
        """
        return np.array([bp['currents'].get(electrode, {}).get(component, 0.0)
                for bp in self.bias_points])

    def get_iv_data(self, electrode: int) -> Tuple[List[float], List[float]]:
        """
        Get I-V data for a specific electrode.

        Parameters
        ----------
        electrode : int
            Electrode number (1-indexed)

        Returns
        -------
        tuple
            (voltages, currents) lists
        """
        voltages = self.get_voltages(electrode)
        currents = self.get_currents(electrode, 'total')
        return voltages, currents

    def get_transfer_characteristic(self, gate_electrode: int,
                                     drain_electrode: int) -> Tuple[List[float], List[float]]:
        """
        Extract transfer characteristic (Id vs Vg).

        Parameters
        ----------
        gate_electrode : int
            Gate electrode number
        drain_electrode : int
            Drain electrode number

        Returns
        -------
        tuple
            (gate_voltages, drain_currents) lists
        """
        vg = self.get_voltages(gate_electrode)
        id_vals = np.abs(self.get_currents(drain_electrode, 'total'))
        return vg, id_vals

    def get_output_characteristic(self, drain_electrode: int) -> Tuple[List[float], List[float]]:
        """
        Extract output characteristic (Id vs Vd).

        Parameters
        ----------
        drain_electrode : int
            Drain electrode number

        Returns
        -------
        tuple
            (drain_voltages, drain_currents) lists
        """
        vd = self.get_voltages(drain_electrode)
        id_vals = np.abs(self.get_currents(drain_electrode, 'total'))
        return vd, id_vals

    def get_gummel_data(self, base_electrode: int, collector_electrode: int) -> Tuple[List[float], List[float]]:
        """
        Extract Gummel plot data (Vbe vs Ic, Ib).

        Parameters
        ----------
        base_electrode : int
            Base electrode number
        collector_electrode : int
            Collector electrode number

        Returns
        -------
        tuple
            (base_voltages, current_values) lists
            If base_electrode == collector_electrode, returns (Vbe, Ib)
            If base_electrode != collector_electrode, returns (Vbe, Ic)
        """
        vbe = self.get_voltages(base_electrode)
        i_val = np.abs(self.get_currents(collector_electrode, 'total'))
        return vbe, i_val

    # -----------------------------------------------------------------------
    # Plotting methods
    # -----------------------------------------------------------------------

    def _find_swept_electrode(self) -> int:
        """
        Find the electrode with the largest voltage variation (the swept electrode).

        Returns
        -------
        int
            Electrode number of the swept electrode
        """
        max_range = 0.0
        swept_electrode = 1

        for elec in range(1, self.num_electrodes + 1):
            voltages = self.get_voltages(elec)
            if len(voltages) > 0:
                v_range = max(voltages) - min(voltages)
                if v_range > max_range:
                    max_range = v_range
                    swept_electrode = elec

        return swept_electrode

    def plot(
        self,
        current_electrode: int,
        voltage_electrode: Optional[int] = None,
        title: Optional[str] = None,
        log_scale: bool = False,
        backend: Optional[str] = None,
        show: bool = True,
        **kwargs
    ):
        """
        Plot I-V data for a specific electrode.

        Parameters
        ----------
        current_electrode : int
            Electrode number for current (y-axis)
        voltage_electrode : int, optional
            Electrode number for voltage (x-axis). If None, auto-detects
            the swept electrode (electrode with largest voltage variation).
        title : str, optional
            Plot title
        log_scale : bool
            Use log scale for current
        backend : str, optional
            'matplotlib' or 'plotly'
        show : bool
            Display plot immediately
        **kwargs
            Additional plotting arguments

        Returns
        -------
        Any
            Plot object (matplotlib Axes or plotly Figure)

        Example
        -------
        >>> iv_data = sim.get_iv_data()
        >>> # Plot drain current vs gate voltage (auto-detect gate as swept)
        >>> iv_data.plot(current_electrode=2)
        >>> # Explicitly specify both electrodes
        >>> iv_data.plot(current_electrode=2, voltage_electrode=3)
        """
        from .visualization import plot_iv

        # Get current data
        currents = self.get_currents(current_electrode, 'total')

        # Determine voltage electrode (use swept electrode if not specified)
        if voltage_electrode is None:
            voltage_electrode = self._find_swept_electrode()

        voltages = self.get_voltages(voltage_electrode)

        if voltage_electrode == current_electrode:
            title = title or f"I-V Characteristic - Electrode {current_electrode}"
        else:
            title = title or f"I{current_electrode} vs V{voltage_electrode}"

        return plot_iv(
            voltages, currents,
            title=title,
            log_scale=log_scale,
            backend=backend,
            show=show,
            **kwargs
        )

    def plot_transfer(
        self,
        gate_electrode: int,
        drain_electrode: int,
        title: Optional[str] = None,
        log_scale: bool = True,
        backend: Optional[str] = None,
        show: bool = True,
        **kwargs
    ):
        """
        Plot transfer characteristic (Id vs Vg).

        Parameters
        ----------
        gate_electrode : int
            Gate electrode number
        drain_electrode : int
            Drain electrode number
        title : str, optional
            Plot title
        log_scale : bool
            Use log scale for drain current (default True)
        backend : str, optional
            'matplotlib' or 'plotly'
        show : bool
            Display plot immediately
        **kwargs
            Additional plotting arguments

        Returns
        -------
        Any
            Plot object (matplotlib Axes or plotly Figure)

        Example
        -------
        >>> iv_data = sim.get_iv_data()
        >>> iv_data.plot_transfer(gate_electrode=3, drain_electrode=2)
        """
        from .visualization import plot_transfer_characteristic
        vg, id_vals = self.get_transfer_characteristic(gate_electrode, drain_electrode)
        title = title or f"Transfer Characteristic (Gate={gate_electrode}, Drain={drain_electrode})"
        return plot_transfer_characteristic(
            vg, id_vals,
            title=title,
            log_scale=log_scale,
            backend=backend,
            show=show,
            **kwargs
        )

    def plot_output(
        self,
        drain_electrode: int,
        title: Optional[str] = None,
        log_scale: bool = False,
        backend: Optional[str] = None,
        show: bool = True,
        **kwargs
    ):
        """
        Plot output characteristic (Id vs Vd).

        Parameters
        ----------
        drain_electrode : int
            Drain electrode number
        title : str, optional
            Plot title
        log_scale : bool
            Use log scale for drain current (default False)
        backend : str, optional
            'matplotlib' or 'plotly'
        show : bool
            Display plot immediately
        **kwargs
            Additional plotting arguments

        Returns
        -------
        Any
            Plot object (matplotlib Axes or plotly Figure)

        Example
        -------
        >>> iv_data = sim.get_iv_data()
        >>> iv_data.plot_output(drain_electrode=2)
        """
        from .visualization import plot_output_characteristic
        vd, id_vals = self.get_output_characteristic(drain_electrode)
        title = title or f"Output Characteristic (Drain={drain_electrode})"
        return plot_output_characteristic(
            vd, id_vals,
            title=title,
            log_scale=log_scale,
            backend=backend,
            show=show,
            **kwargs
        )

    def plot_all_electrodes(
        self,
        title: str = "I-V Characteristics - All Electrodes",
        log_scale: bool = False,
        backend: Optional[str] = None,
        show: bool = True,
        **kwargs
    ):
        """
        Plot I-V data for all electrodes.

        Parameters
        ----------
        title : str
            Plot title
        log_scale : bool
            Use log scale for current
        backend : str, optional
            'matplotlib' or 'plotly'
        show : bool
            Display plot immediately
        **kwargs
            Additional plotting arguments

        Returns
        -------
        Any
            Plot object (matplotlib Axes or plotly Figure)

        Example
        -------
        >>> iv_data = sim.get_iv_data()
        >>> iv_data.plot_all_electrodes(log_scale=True)
        """
        from .visualization import (
            _get_default_backend,
            _plot_multi_iv_matplotlib,
            _plot_multi_iv_plotly
        )

        if backend is None:
            backend = _get_default_backend()

        data_series = []
        for elec in range(1, self.num_electrodes + 1):
            voltages, currents = self.get_iv_data(elec)
            data_series.append((voltages, currents, f"Electrode {elec}"))

        if backend == 'matplotlib':
            return _plot_multi_iv_matplotlib(
                data_series,
                title=title,
                log_scale=log_scale,
                show=show,
                **kwargs
            )
        else:
            return _plot_multi_iv_plotly(
                data_series,
                title=title,
                log_scale=log_scale,
                show=show,
                **kwargs
            )

    def plot_gummel(
        self,
        base_electrode: int = 2,
        collector_electrode: int = 3,
        emitter_electrode: int = 1,
        title: str = "Gummel Plot",
        log_scale: bool = True,
        backend: Optional[str] = None,
        show: bool = True,
        **kwargs
    ) -> Any:
        """
        Plot Gummel characteristics (Ic, Ib vs Vbe).

        Parameters
        ----------
        base_electrode : int
            Base electrode number (default: 2)
        collector_electrode : int
            Collector electrode number (default: 3)
        emitter_electrode : int
            Emitter electrode number (default: 1)
        title : str
            Plot title
        log_scale : bool
            Use log scale for current (default True)
        backend : str, optional
            'matplotlib' or 'plotly'
        show : bool
            Display plot immediately
        **kwargs
            Additional plotting arguments

        Returns
        -------
        Any
            Plot object
        """
        from .visualization import (
            _get_default_backend,
            _plot_multi_iv_matplotlib,
            _plot_multi_iv_plotly
        )
        
        # Get raw data using the new get_gummel_data method
        vbe, ic = self.get_gummel_data(base_electrode, collector_electrode)
        _, ib = self.get_gummel_data(base_electrode, base_electrode)

        if backend is None:
            backend = _get_default_backend()

        data_series = [
            (vbe, ic, f"Ic (Collector)"),
            (vbe, ib, f"Ib (Base)")
        ]

        if backend == 'matplotlib':
            return _plot_multi_iv_matplotlib(
                data_series,
                xlabel="Base-Emitter Voltage Vbe (V)",
                ylabel="Current (A/µm)",
                title=title,
                log_scale=log_scale,
                show=show,
                **kwargs
            )
        else:
            return _plot_multi_iv_plotly(
                data_series,
                xlabel="Base-Emitter Voltage Vbe (V)",
                ylabel="Current (A/µm)",
                title=title,
                log_scale=log_scale,
                show=show,
                **kwargs
            )


class IVFileParser:
    """
    Parser for PADRE log files (ivfile format).

    PADRE log files contain I-V data in Q-records that store electrode
    voltages and currents for each bias point.  Two header variants exist:

    Old format:
      - Line 1: plain-text header (e.g. "# PADRE2.4E 10/24/94")
      - Line 2: Fortran-formatted integers  N  N  0
      - Q-records follow immediately; each has 4 data lines (20 values).

    New format (Fortran binary-style):
      - Lines 1–N: ASCII character codes (integers 32–126) encoding the
        header string and a date stamp.
      - A mesh-info line: nodes  elements  num_electrodes_defined
      - An IV-setup line:  num_iv_slots  start_voltage  temperature
      - Short preamble; first Q may be embedded mid-line.
      - Q-records have 3 data lines (15 values).

    Value layout (4 IV-slots, 20 values / old format):
      [0-3]   V1 V2 V3 V4
      [4-7]   padding / repeats
      [8-11]  I1_e I2_e I3_e I4_e   (electron currents)
      [12-15] I1_h I2_h I3_h I4_h   (hole currents)
      [16-19] charges

    Value layout (4 IV-slots, 15 values / new format):
      [0-3]   V1 V2 V3 V4
      [4-5]   V_repeats
      [6-9]   I1 I2 I3 I4            (total currents, single carrier)
      [10-11] small residuals
      [12-14] V_repeats
    """

    def __init__(self):
        self.data = IVData()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_ascii_code_line(line: str) -> bool:
        """Return True if every token is an integer in printable-ASCII range
        or a Fortran-style float (contains 'E' / '.') mixed with ASCII codes.
        Used to detect the character-code preamble in the new format."""
        tokens = line.split()
        if not tokens:
            return False
        for tok in tokens:
            # Pure integer in printable-ASCII range (32-126)?
            try:
                v = int(tok)
                if 32 <= v <= 126:
                    continue
                # Large integers (mesh nodes/elements) are NOT ascii-code lines
                return False
            except ValueError:
                pass
            # Fortran float like "0.000000000000000E+000" — appears on
            # line 8 of the new header alongside an ASCII code "35".
            try:
                float(tok)
                continue
            except ValueError:
                return False
        return True

    @staticmethod
    def _split_on_q(line: str):
        """If 'Q' appears mid-line (e.g. '645.9Q0.00000E+00 0.00000E+00'),
        return the portion from Q onward; otherwise return None."""
        idx = line.find('Q')
        if idx > 0:  # Q not at position 0, so it's embedded
            return line[idx:]
        return None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def parse(self, content: str) -> IVData:
        """
        Parse PADRE log file content (both old and new formats).

        Parameters
        ----------
        content : str
            Content of the PADRE log file

        Returns
        -------
        IVData
            Parsed I-V data
        """
        self.data = IVData()
        lines = content.strip().split('\n')
        if len(lines) < 3:
            return self.data

        # ----------------------------------------------------------
        # 1. Detect format and extract num_electrodes
        # ----------------------------------------------------------
        first_line = lines[0].strip()
        if first_line.startswith('#'):
            # --- Old format: plain-text header ---
            # Line 2 has electrode counts as Fortran-formatted integers
            header_parts = lines[1].strip().split()
            if header_parts:
                try:
                    self.data.num_electrodes = int(header_parts[0])
                except ValueError:
                    pass
            scan_start = 2          # Q-records start at line index 2
        else:
            # --- New format: character-code preamble ---
            # Skip lines that look like ASCII-code lines.  Then expect:
            #   mesh-info:  nodes  elements  num_electrodes_defined
            #   iv-setup:   num_iv_slots  start_voltage  temperature
            scan_start = 0
            for i, raw in enumerate(lines):
                stripped = raw.strip()
                if not stripped:
                    continue
                if self._is_ascii_code_line(stripped):
                    continue
                # First non-ascii-code line: mesh info (3 large integers)
                parts = stripped.split()
                if len(parts) == 3:
                    try:
                        int(parts[0]); int(parts[1]); int(parts[2])
                        # Next line is the IV-setup line
                        iv_parts = lines[i + 1].strip().split()
                        self.data.num_electrodes = int(iv_parts[0])
                        scan_start = i + 2   # skip past IV-setup line
                        break
                    except (ValueError, IndexError):
                        pass
                # Fallback: if we can't parse mesh info, just start scanning
                scan_start = i
                break

        # ----------------------------------------------------------
        # 2. Collect Q-record text blocks
        # ----------------------------------------------------------
        # A Q-record starts at a line beginning with 'Q' or with 'Q'
        # embedded after non-Q text (first record in new format).
        # Each block = the Q-header text + subsequent data lines until
        # the next Q or end of file.
        q_blocks: List[str] = []   # each entry is the full text of one Q-record

        current_block: Optional[List[str]] = None

        for i in range(scan_start, len(lines)):
            line = lines[i].strip()
            if not line:
                continue

            if line.startswith('Q'):
                # Flush previous block
                if current_block is not None:
                    q_blocks.append('\n'.join(current_block))
                current_block = [line]
            else:
                # Check for embedded Q (e.g. "645.9Q0.00000E+00 ...")
                embedded = self._split_on_q(line)
                if embedded is not None:
                    if current_block is not None:
                        q_blocks.append('\n'.join(current_block))
                    current_block = [embedded]
                elif current_block is not None:
                    # Continuation data line — at most 4 data lines per record
                    # (Q-line is index 0; indices 1–4 are data)
                    if len(current_block) < 5:
                        current_block.append(line)
                    else:
                        # Block is full; flush and stop collecting until next Q
                        q_blocks.append('\n'.join(current_block))
                        current_block = None
                # else: pre-Q preamble data — ignore

        # Don't forget the last block
        if current_block is not None:
            q_blocks.append('\n'.join(current_block))

        # ----------------------------------------------------------
        # 3. Parse each Q-block into a bias point
        # ----------------------------------------------------------
        for block in q_blocks:
            bias_point = self._parse_q_block(block)
            if bias_point:
                self.data.bias_points.append(bias_point)

        return self.data

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------
    def _parse_q_block(self, block: str) -> Optional[Dict]:
        """Parse a single Q-record block (Q-line + data lines) into a
        bias-point dict with 'voltages' and 'currents'.

        The Q-line itself ("Q0.00000E+00 0.00000E+00") carries bias-step
        metadata and is intentionally excluded.  Electrode data lives
        exclusively in the subsequent data lines."""
        try:
            block_lines = block.strip().split('\n')

            # Collect numeric values from data lines only (skip Q-line)
            values: List[float] = []
            for dl in block_lines[1:]:
                for token in dl.strip().split():
                    try:
                        values.append(float(token))
                    except ValueError:
                        pass

            num_elec = self.data.num_electrodes
            if num_elec == 0:
                num_elec = 4 if len(values) >= 15 else 2

            if len(values) < num_elec + 2:   # need at least voltages + some currents
                return None

            bias_point: Dict = {'voltages': {}, 'currents': {}}

            # Voltages are always the first num_elec values
            for elec in range(1, num_elec + 1):
                bias_point['voltages'][elec] = values[elec - 1]

            # Determine current base index from total value count:
            #   20 values (old, 4 data lines): electron base = 8, hole base = 12
            #   15 values (new, 3 data lines): single-carrier base = 6, no separate hole
            #   10 values (2-electrode):       total current base = 4, no hole
            n_vals = len(values)
            if n_vals >= 20:
                e_base = 8
                h_base = 12
                has_hole = True
            elif n_vals >= 15:
                e_base = 6
                h_base = None
                has_hole = False
            else:
                # 2-electrode / 10-value layout: [0,1]=V, [2,3]=V repeat,
                # [4,5]=I_total, [6,7]=residuals, [8,9]=V repeat
                e_base = 4
                h_base = None
                has_hole = False

            for elec in range(1, num_elec + 1):
                ec: Dict = {}
                e_idx = e_base + (elec - 1)
                if e_idx < n_vals:
                    ec['electron'] = values[e_idx]

                if has_hole and h_base is not None:
                    h_idx = h_base + (elec - 1)
                    if h_idx < n_vals:
                        ec['hole'] = values[h_idx]

                # Total = electron + hole (or whichever is available)
                if 'electron' in ec and 'hole' in ec:
                    ec['total'] = ec['electron'] + ec['hole']
                elif 'electron' in ec:
                    ec['total'] = ec['electron']
                elif 'hole' in ec:
                    ec['total'] = ec['hole']

                bias_point['currents'][elec] = ec

            return bias_point

        except (IndexError, ValueError):
            return None


def parse_iv_file(filename: str) -> IVData:
    """
    Parse a PADRE log file (ivfile).

    Parameters
    ----------
    filename : str
        Path to the PADRE log file

    Returns
    -------
    IVData
        Parsed I-V data with voltages and currents for each electrode

    Example
    -------
    >>> iv_data = parse_iv_file("idvg")
    >>> vg, id = iv_data.get_transfer_characteristic(gate_electrode=3, drain_electrode=2)
    >>> print(f"Vg range: {min(vg):.2f} to {max(vg):.2f} V")
    """
    with open(filename, 'r') as f:
        content = f.read()
    parser = IVFileParser()
    return parser.parse(content)


def parse_iv_content(content: str) -> IVData:
    """
    Parse PADRE log file content directly.

    Parameters
    ----------
    content : str
        Content of the PADRE log file

    Returns
    -------
    IVData
        Parsed I-V data
    """
    parser = IVFileParser()
    return parser.parse(content)


@dataclass
class ACData:
    """
    Parsed AC analysis data from PADRE acfile.

    Units: PADRE solves a 2D cross-section, so capacitances are in farads
    **per micron of device depth** (times the mesh ``width=`` parameter,
    default 1 µm) and conductances in siemens per micron of depth.
    Multiply by the physical device depth in microns for absolute values.
    """
    frequency: np.ndarray = field(default_factory=lambda: np.array([]))
    voltages: Dict[int, np.ndarray] = field(default_factory=dict)
    capacitance: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    conductance: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)

    def get_cv_data(self, gate_electrode: int = 1, bulk_electrode: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract C-V characteristic (Cgg vs Vg).
        
        Assumes data is for a single frequency.
        """
        if gate_electrode in self.voltages:
            vg = self.voltages[gate_electrode]
        elif len(self.voltages) > 0:
            vg = list(self.voltages.values())[0]
        else:
            return np.array([]), np.array([])
            
        # Try to get C(gate, gate) - typically input capacitance
        key = (gate_electrode, gate_electrode)
        if key in self.capacitance:
            cgg = self.capacitance[key]
        elif len(self.capacitance) > 0:
            # First available capacitance if keys fail
            cgg = list(self.capacitance.values())[0]
        else:
            cgg = np.zeros_like(vg)

        return vg, cgg


class ACFileParser:
    """Parser for PADRE AC log files."""
    
    def parse(self, content: str) -> ACData:
        """
        Parse AC file content.
        
        PADRE AC format:
        # PADRE2.4E 10/24/94
        Q  electrode  capacitance  frequency  v1  v2  v3 ...
           continuation_line_with_more_data
           continuation_line_with_more_data
           electrode_num  val1 val2 val3 capacitance val4 ...
           electrode_num  val1 val2 val3 capacitance val4 ...
        Q  electrode  capacitance  frequency  v1  v2  v3 ...  (next bias point)
        
        The varying capacitance is in the numbered electrode lines (5 values),
        specifically the 4th value (index 3).
        """
        import re
        
        # Fix scientific notation format (e.g., 1.23-04 -> 1.23E-04)
        content = re.sub(r'([0-9])([-+][0-9])', r'\g<1>E\g<2>', content)

        # Some PADRE versions embed the first Q record mid-line after garbage bytes.
        # Ensure every Q record starts on its own line.
        content = re.sub(r'(?<!\n)(Q\s+\d)', r'\n\1', content)

        lines = [l.rstrip() for l in content.split('\n')]
        if not lines:
            return ACData()
            
        result = ACData()
        frequencies = []
        voltage_data = {}  # electrode -> list of voltages
        capacitance_data = {}  # (electrode, electrode) -> list of capacitances
        
        current_voltage = None  # Store voltage from Q record
        
        try:
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and header
                if not line or line.startswith('#'):
                    continue
                
                # Extract all numeric values from the line
                regex = re.findall(r'-?\s*[0-9]+\.?[0-9]*(?:[Ee]\s*[-+]?\s*[0-9]+)?', line)
                
                # Q records (start with 'Q')
                if line.startswith('Q'):
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                        
                    # Format: Q electrode capacitance frequency v1 v2 v3 ...
                    electrode = int(parts[1])
                    frequency = float(parts[3])
                    
                    # Voltages start at index 4
                    voltages = []
                    for j in range(4, len(parts)):
                        try:
                            voltages.append(float(parts[j]))
                        except ValueError:
                            break
                    
                    # Store the gate voltage (first voltage value)
                    if len(voltages) > 0:
                        current_voltage = voltages[0]
                    
                    # Store frequency
                    if frequency not in frequencies:
                        frequencies.append(frequency)
                
                # Lines with 5 numeric values contain electrode capacitance data
                elif len(regex) == 5:
                    try:
                        elec_num = int(float(regex[0].strip()))
                        # The 4th value (index 3) is the capacitance
                        cap_value = abs(float(regex[3].strip())) * 1e8  # Scale factor from your code
                        
                        # Store voltage for this electrode
                        if current_voltage is not None:
                            if elec_num not in voltage_data:
                                voltage_data[elec_num] = []
                            # Only add if it's a new voltage value
                            if len(voltage_data[elec_num]) == 0 or voltage_data[elec_num][-1] != current_voltage:
                                voltage_data[elec_num].append(current_voltage)
                        
                        # Store capacitance for this electrode
                        key = (elec_num, elec_num)
                        if key not in capacitance_data:
                            capacitance_data[key] = []
                        capacitance_data[key].append(cap_value)
                        
                    except (ValueError, IndexError):
                        continue
            
            # Convert to numpy arrays
            result.frequency = np.array(frequencies) if frequencies else np.array([])
            
            for electrode, voltages in voltage_data.items():
                result.voltages[electrode] = np.array(voltages)
            
            for key, capacitances in capacitance_data.items():
                result.capacitance[key] = np.array(capacitances)
                
            return result
            
        except Exception as e:
            return ACData()


def parse_ac_file(filename: str) -> ACData:
    """Parse a PADRE AC log file."""
    with open(filename, 'r') as f:
        content = f.read()
    parser = ACFileParser()
    return parser.parse(content)
