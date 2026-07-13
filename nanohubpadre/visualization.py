"""
Visualization module for PADRE simulation results.

Provides plotting functions for I-V characteristics using matplotlib and plotly
as rendering backends. Supports transfer characteristics (Id-Vg), output
characteristics (Id-Vd), general I-V curves, and 2D contour maps of device
quantities (potential, doping, carrier concentrations, etc.).

Example
-------
>>> from nanohubpadre import create_mosfet, Solve, Log
>>> sim = create_mosfet()
>>> sim.add_log(Log(ivfile="idvg"))
>>> sim.add_solve(Solve(v3=0, vstep=0.1, nsteps=15, electrode=3))
>>> result = sim.run()
>>> # Plot using matplotlib
>>> sim.plot_transfer(gate_electrode=3, drain_electrode=2)
>>> # Plot using plotly
>>> sim.plot_transfer(gate_electrode=3, drain_electrode=2, backend="plotly")
"""

from typing import List, Optional, Tuple, Union, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .parser import IVData

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _check_matplotlib() -> bool:
    """Check if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        return True
    except ImportError:
        return False


def _check_plotly() -> bool:
    """Check if plotly is available."""
    try:
        import plotly.graph_objects as go  # noqa: F401
        return True
    except ImportError:
        return False


def get_available_backends() -> List[str]:
    """
    Get list of available plotting backends.

    Returns
    -------
    List[str]
        List of available backend names ('matplotlib', 'plotly')
    """
    backends = []
    if _check_matplotlib():
        backends.append('matplotlib')
    if _check_plotly():
        backends.append('plotly')
    return backends


def _get_default_backend() -> str:
    """Get the default plotting backend."""
    if _check_matplotlib():
        return 'matplotlib'
    elif _check_plotly():
        return 'plotly'
    else:
        raise ImportError(
            "No plotting backend available. Install matplotlib or plotly:\n"
            "  pip install matplotlib\n"
            "  pip install plotly"
        )


# ---------------------------------------------------------------------------
# Matplotlib plotting functions
# ---------------------------------------------------------------------------

def _plot_iv_matplotlib(
    voltages: List[float],
    currents: List[float],
    xlabel: str = "Voltage (V)",
    ylabel: str = "Current (A/µm)",
    title: str = "I-V Characteristic",
    log_scale: bool = False,
    abs_current: bool = True,
    marker: str = 'o-',
    color: Optional[str] = None,
    label: Optional[str] = None,
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (8, 6),
    grid: bool = True,
    show: bool = True,
    **kwargs
) -> Any:
    """
    Plot I-V data using matplotlib.

    Parameters
    ----------
    voltages : List[float]
        Voltage values
    currents : List[float]
        Current values
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str
        Plot title
    log_scale : bool
        Use logarithmic scale for current (y-axis)
    abs_current : bool
        Plot absolute value of current
    marker : str
        Matplotlib marker style
    color : str, optional
        Line color
    label : str, optional
        Legend label
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on
    figsize : tuple
        Figure size (width, height) in inches
    grid : bool
        Show grid
    show : bool
        Call plt.show() after plotting

    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Process currents
    y_data = [abs(i) for i in currents] if abs_current else currents

    # Plot
    plot_kwargs = {'label': label}
    if color:
        plot_kwargs['color'] = color
    plot_kwargs.update(kwargs)

    if log_scale:
        ax.semilogy(voltages, y_data, marker, **plot_kwargs)
    else:
        ax.plot(voltages, y_data, marker, **plot_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if grid:
        ax.grid(True, alpha=0.3)

    if label:
        ax.legend()

    if show:
        plt.show()

    return ax


def _plot_multi_iv_matplotlib(
    data_series: List[Tuple[List[float], List[float], str]],
    xlabel: str = "Voltage (V)",
    ylabel: str = "Current (A/µm)",
    title: str = "I-V Characteristics",
    log_scale: bool = False,
    abs_current: bool = True,
    figsize: Tuple[float, float] = (8, 6),
    grid: bool = True,
    show: bool = True,
    **kwargs
) -> Any:
    """
    Plot multiple I-V curves using matplotlib.

    Parameters
    ----------
    data_series : List[Tuple[List[float], List[float], str]]
        List of (voltages, currents, label) tuples
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str
        Plot title
    log_scale : bool
        Use logarithmic scale for current
    abs_current : bool
        Plot absolute value of current
    figsize : tuple
        Figure size
    grid : bool
        Show grid
    show : bool
        Call plt.show()

    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    for voltages, currents, label in data_series:
        y_data = [abs(i) for i in currents] if abs_current else currents

        if log_scale:
            ax.semilogy(voltages, y_data, 'o-', label=label, **kwargs)
        else:
            ax.plot(voltages, y_data, 'o-', label=label, **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if grid:
        ax.grid(True, alpha=0.3)

    ax.legend()

    if show:
        plt.show()

    return ax


# ---------------------------------------------------------------------------
# Plotly plotting functions
# ---------------------------------------------------------------------------

def _plot_iv_plotly(
    voltages: List[float],
    currents: List[float],
    xlabel: str = "Voltage (V)",
    ylabel: str = "Current (A/µm)",
    title: str = "I-V Characteristic",
    log_scale: bool = False,
    abs_current: bool = True,
    color: Optional[str] = None,
    label: Optional[str] = None,
    fig: Optional[Any] = None,
    width: int = 800,
    height: int = 600,
    show: bool = True,
    **kwargs
) -> Any:
    """
    Plot I-V data using plotly.

    Parameters
    ----------
    voltages : List[float]
        Voltage values
    currents : List[float]
        Current values
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str
        Plot title
    log_scale : bool
        Use logarithmic scale for current
    abs_current : bool
        Plot absolute value of current
    color : str, optional
        Line color
    label : str, optional
        Legend label
    fig : plotly.graph_objects.Figure, optional
        Existing figure to add trace to
    width : int
        Figure width in pixels
    height : int
        Figure height in pixels
    show : bool
        Call fig.show()

    Returns
    -------
    plotly.graph_objects.Figure
        The figure object
    """
    import plotly.graph_objects as go

    # Process currents
    y_data = [abs(i) for i in currents] if abs_current else currents

    # Create or update figure
    if fig is None:
        fig = go.Figure()

    # Add trace
    trace_kwargs = {
        'x': voltages,
        'y': y_data,
        'mode': 'lines+markers',
        'name': label or 'I-V',
    }
    if color:
        trace_kwargs['line'] = {'color': color}
    trace_kwargs.update(kwargs)

    fig.add_trace(go.Scatter(**trace_kwargs))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=width,
        height=height,
        template='plotly_white',
    )

    if log_scale:
        fig.update_yaxes(type='log')

    if show:
        fig.show()

    return fig


def _plot_multi_iv_plotly(
    data_series: List[Tuple[List[float], List[float], str]],
    xlabel: str = "Voltage (V)",
    ylabel: str = "Current (A/µm)",
    title: str = "I-V Characteristics",
    log_scale: bool = False,
    abs_current: bool = True,
    width: int = 800,
    height: int = 600,
    show: bool = True,
    **kwargs
) -> Any:
    """
    Plot multiple I-V curves using plotly.

    Parameters
    ----------
    data_series : List[Tuple[List[float], List[float], str]]
        List of (voltages, currents, label) tuples
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str
        Plot title
    log_scale : bool
        Use logarithmic scale for current
    abs_current : bool
        Plot absolute value of current
    width : int
        Figure width
    height : int
        Figure height
    show : bool
        Call fig.show()

    Returns
    -------
    plotly.graph_objects.Figure
        The figure object
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    for voltages, currents, label in data_series:
        y_data = [abs(i) for i in currents] if abs_current else currents

        fig.add_trace(go.Scatter(
            x=voltages,
            y=y_data,
            mode='lines+markers',
            name=label,
            **kwargs
        ))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=width,
        height=height,
        template='plotly_white',
    )

    if log_scale:
        fig.update_yaxes(type='log')

    if show:
        fig.show()

    return fig


# ---------------------------------------------------------------------------
# Public API - Generic plotting functions
# ---------------------------------------------------------------------------

def plot_iv(
    voltages: List[float],
    currents: List[float],
    xlabel: str = "Voltage (V)",
    ylabel: str = "Current (A/µm)",
    title: str = "I-V Characteristic",
    log_scale: bool = False,
    abs_current: bool = True,
    backend: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> Any:
    """
    Plot I-V characteristic curve.

    Parameters
    ----------
    voltages : List[float]
        Voltage values
    currents : List[float]
        Current values
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str
        Plot title
    log_scale : bool
        Use logarithmic scale for current (y-axis)
    abs_current : bool
        Plot absolute value of current (default True)
    backend : str, optional
        Plotting backend: 'matplotlib' or 'plotly'. If None, uses first available.
    show : bool
        Display the plot immediately
    **kwargs
        Additional arguments passed to the backend plotting function

    Returns
    -------
    Any
        matplotlib.axes.Axes or plotly.graph_objects.Figure depending on backend

    Example
    -------
    >>> from nanohubpadre.visualization import plot_iv
    >>> voltages = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    >>> currents = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
    >>> plot_iv(voltages, currents, log_scale=True)
    """
    if backend is None:
        backend = _get_default_backend()

    if backend == 'matplotlib':
        return _plot_iv_matplotlib(
            voltages, currents, xlabel, ylabel, title,
            log_scale, abs_current, show=show, **kwargs
        )
    elif backend == 'plotly':
        return _plot_iv_plotly(
            voltages, currents, xlabel, ylabel, title,
            log_scale, abs_current, show=show, **kwargs
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'matplotlib' or 'plotly'.")


def plot_transfer_characteristic(
    gate_voltages: List[float],
    drain_currents: List[float],
    title: str = "Transfer Characteristic (Id-Vg)",
    log_scale: bool = True,
    backend: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> Any:
    """
    Plot MOSFET transfer characteristic (Id vs Vg).

    Parameters
    ----------
    gate_voltages : List[float]
        Gate voltage values (Vg)
    drain_currents : List[float]
        Drain current values (Id)
    title : str
        Plot title
    log_scale : bool
        Use logarithmic scale for drain current (default True for subthreshold)
    backend : str, optional
        Plotting backend: 'matplotlib' or 'plotly'
    show : bool
        Display the plot immediately
    **kwargs
        Additional arguments passed to the backend

    Returns
    -------
    Any
        Plot object (axes or figure)

    Example
    -------
    >>> vg, id = sim.get_transfer_characteristic(gate_electrode=3, drain_electrode=2)
    >>> plot_transfer_characteristic(vg, id)
    """
    return plot_iv(
        gate_voltages, drain_currents,
        xlabel="Gate Voltage Vg (V)",
        ylabel="Drain Current |Id| (A)",
        title=title,
        log_scale=log_scale,
        abs_current=True,
        backend=backend,
        show=show,
        **kwargs
    )


def plot_output_characteristic(
    drain_voltages: List[float],
    drain_currents: List[float],
    title: str = "Output Characteristic (Id-Vd)",
    log_scale: bool = False,
    backend: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> Any:
    """
    Plot MOSFET output characteristic (Id vs Vd).

    Parameters
    ----------
    drain_voltages : List[float]
        Drain voltage values (Vd)
    drain_currents : List[float]
        Drain current values (Id)
    title : str
        Plot title
    log_scale : bool
        Use logarithmic scale for drain current (default False for output)
    backend : str, optional
        Plotting backend: 'matplotlib' or 'plotly'
    show : bool
        Display the plot immediately
    **kwargs
        Additional arguments passed to the backend

    Returns
    -------
    Any
        Plot object (axes or figure)

    Example
    -------
    >>> vd, id = sim.get_output_characteristic(drain_electrode=2)
    >>> plot_output_characteristic(vd, id)
    """
    return plot_iv(
        drain_voltages, drain_currents,
        xlabel="Drain Voltage Vd (V)",
        ylabel="Drain Current |Id| (A)",
        title=title,
        log_scale=log_scale,
        abs_current=True,
        backend=backend,
        show=show,
        **kwargs
    )


def plot_diode_iv(
    voltages: List[float],
    currents: List[float],
    title: str = "Diode I-V Characteristic",
    log_scale: bool = True,
    backend: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> Any:
    """
    Plot diode I-V characteristic.

    Parameters
    ----------
    voltages : List[float]
        Applied voltage values
    currents : List[float]
        Current values
    title : str
        Plot title
    log_scale : bool
        Use logarithmic scale for current (default True)
    backend : str, optional
        Plotting backend: 'matplotlib' or 'plotly'
    show : bool
        Display the plot immediately
    **kwargs
        Additional arguments passed to the backend

    Returns
    -------
    Any
        Plot object (axes or figure)
    """
    return plot_iv(
        voltages, currents,
        xlabel="Voltage (V)",
        ylabel="|Current| (A)",
        title=title,
        log_scale=log_scale,
        abs_current=True,
        backend=backend,
        show=show,
        **kwargs
    )


# ---------------------------------------------------------------------------
# 2D contour map for Plot3D data
# ---------------------------------------------------------------------------

def plot_2d_map(fig, data, col, colorscale="RdBu_r", log_scale=False,
                cbar_title="V", n_grid=100):
    """
    Interpolate Plot3D scatter data onto a regular grid and add a heatmap
    with contour overlay to a plotly subplot.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        A plotly Figure (typically created with ``make_subplots``)
    data : Plot3DData
        Parsed Plot3D scatter data (must have a ``to_grid`` method)
    col : int
        Subplot column index (1-based)
    colorscale : str
        Plotly colorscale name (default: "RdBu_r")
    log_scale : bool
        If True, plot log10 of absolute values (default: False)
    cbar_title : str
        Colorbar title (default: "V")
    n_grid : int
        Number of grid points for interpolation (default: 100)

    Returns
    -------
    np.ndarray
        The 2D interpolated (and optionally log-transformed) grid values
    """
    import plotly.graph_objects as go

    xi, yi, zi = data.to_grid(n_grid=n_grid, method='linear')

    if log_scale:
        zi = np.log10(np.abs(zi) + 1e-30)
        cbar_title = f"log\u2081\u2080|{cbar_title}|"

    fig.add_trace(go.Heatmap(
        x=xi, y=yi, z=zi,
        colorscale=colorscale,
        colorbar=dict(title=cbar_title, x=0.45 if col == 1 else 1.0, len=0.9),
    ), row=1, col=col)

    fig.add_trace(go.Contour(
        x=xi, y=yi, z=zi,
        contours=dict(coloring="none", showlabels=True,
                      labelfont=dict(size=9, color="black")),
        line=dict(color="black", width=1),
        showscale=False, showlegend=False,
    ), row=1, col=col)

    return zi


# ---------------------------------------------------------------------------
# 2D contour map — high-level API with backend support
# ---------------------------------------------------------------------------

def _plotly_cmap_to_matplotlib(name: str) -> str:
    """Convert a Plotly colorscale name to a matplotlib cmap name.

    Plotly uses capitalised names (``"Jet"``, ``"Hot"``, ``"Viridis"``),
    while matplotlib uses lowercase (``"jet"``, ``"hot"``, ``"viridis"``).
    Names that already exist in matplotlib (e.g. ``"RdBu_r"``) are passed
    through unchanged.
    """
    import matplotlib.pyplot as plt
    # Build set of known colormap names (works across matplotlib versions)
    try:
        known = set(plt.colormaps())  # matplotlib < 3.5 returns a list
    except TypeError:
        known = set(plt.colormaps)    # matplotlib >= 3.5 is a registry
    # If the name is already valid, use it directly
    if name in known:
        return name
    # Try lowercase
    lower = name.lower()
    if lower in known:
        return lower
    # Return as-is and let matplotlib raise a clear error
    return name


def _plot_contour_matplotlib(data_list, titles, colorscale, log_scale,
                             cbar_title, n_grid, show, **kwargs):
    """Matplotlib backend for plot_contour."""
    import matplotlib.pyplot as plt

    cmap = _plotly_cmap_to_matplotlib(colorscale)

    ncols = len(data_list)
    fig, axes = plt.subplots(
        1, ncols,
        figsize=kwargs.get('figsize', (6 * ncols, 5)),
    )
    if ncols == 1:
        axes = [axes]

    cb_title = cbar_title
    if log_scale:
        cb_title = f"log\u2081\u2080|{cbar_title}|"

    for ax, data, sub_title in zip(axes, data_list, titles):
        xi, yi, zi = data.to_grid(n_grid=n_grid, method='linear')
        if log_scale:
            zi = np.log10(np.abs(zi) + 1e-30)

        pcm = ax.pcolormesh(xi, yi, zi, cmap=cmap, shading='auto')
        cs = ax.contour(xi, yi, zi, colors='black', linewidths=0.8)
        ax.clabel(cs, fontsize=8, inline=True)
        fig.colorbar(pcm, ax=ax, label=cb_title)
        ax.set_xlabel("X (\u00b5m)")
        ax.set_ylabel("Y (\u00b5m)")
        ax.set_title(sub_title)
        ax.set_aspect('equal')

    plt.tight_layout()
    if show:
        plt.show()

    return fig


def _plot_contour_plotly(data_list, titles, colorscale, log_scale,
                         cbar_title, n_grid, show, **kwargs):
    """Plotly backend for plot_contour."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ncols = len(data_list)
    fig = make_subplots(
        rows=1, cols=ncols,
        subplot_titles=titles,
        horizontal_spacing=0.12 if ncols > 1 else 0.0,
    )

    for col_idx, data in enumerate(data_list, start=1):
        xi, yi, zi = data.to_grid(n_grid=n_grid, method='linear')
        cb = cbar_title
        if log_scale:
            zi = np.log10(np.abs(zi) + 1e-30)
            cb = f"log\u2081\u2080|{cbar_title}|"

        cbar_x = 0.45 if (col_idx == 1 and ncols > 1) else 1.0

        fig.add_trace(go.Heatmap(
            x=xi, y=yi, z=zi,
            colorscale=colorscale,
            colorbar=dict(title=cb, x=cbar_x, len=0.9),
        ), row=1, col=col_idx)

        fig.add_trace(go.Contour(
            x=xi, y=yi, z=zi,
            contours=dict(coloring="none", showlabels=True,
                          labelfont=dict(size=9, color="black")),
            line=dict(color="black", width=1),
            showscale=False, showlegend=False,
        ), row=1, col=col_idx)

    fig.update_xaxes(title_text="X (\u00b5m)")
    fig.update_yaxes(title_text="Y (\u00b5m)")
    fig.update_layout(
        template="plotly_white",
        width=kwargs.get('width', 550 * ncols),
        height=kwargs.get('height', 500),
        showlegend=False,
    )
    for col_idx in range(1, ncols + 1):
        fig.update_yaxes(
            scaleanchor=f"x{'' if col_idx == 1 else col_idx}",
            scaleratio=1, row=1, col=col_idx,
        )

    if show:
        fig.show()

    return fig


def plot_contour(
    data,
    title="2D Contour Map",
    colorscale="RdBu_r",
    log_scale=False,
    cbar_title="V",
    n_grid=100,
    backend=None,
    show=True,
    **kwargs,
):
    """
    Plot 2D contour map(s) from Plot3D scatter data.

    Interpolates scattered mesh-node data onto a regular grid and renders
    a heatmap with contour-line overlay.  Supports side-by-side comparison
    when given multiple datasets (e.g. equilibrium vs. bias).

    Parameters
    ----------
    data : Plot3DData or list of Plot3DData
        Parsed Plot3D data.  A single object produces one panel; a list
        produces side-by-side subplots.
    title : str or list of str
        Plot title.  When *data* is a list, *title* can be a list of
        per-panel titles (e.g. ``["Equilibrium", "Vgs=1V"]``).
    colorscale : str
        Colorscale / colormap name (default ``"RdBu_r"``).
        Plotly and matplotlib names both work.
    log_scale : bool
        If True, plot ``log10(|values|)`` (default False).
    cbar_title : str
        Colorbar label (default ``"V"``).
    n_grid : int
        Interpolation grid resolution (default 100).
    backend : str, optional
        ``"matplotlib"`` or ``"plotly"``.  If None, uses first available.
    show : bool
        Display the plot immediately (default True).
    **kwargs
        Extra keyword arguments forwarded to the backend
        (e.g. ``figsize``, ``width``, ``height``).

    Returns
    -------
    Any
        matplotlib Figure or plotly Figure.

    Example
    -------
    >>> from nanohubpadre import parse_plot3d_file, plot_contour
    >>> data = parse_plot3d_file("pot_eq")
    >>> plot_contour(data, title="Potential", cbar_title="V")
    >>>
    >>> # Side-by-side comparison
    >>> d1 = parse_plot3d_file("pot_eq")
    >>> d2 = parse_plot3d_file("pot_bias")
    >>> plot_contour([d1, d2], title=["Equilibrium", "Bias"], cbar_title="V")
    """
    if backend is None:
        backend = _get_default_backend()

    # Normalise to list
    if not isinstance(data, list):
        data = [data]

    # Normalise titles
    if isinstance(title, str):
        if len(data) == 1:
            titles = [title]
        else:
            titles = [title] + [""] * (len(data) - 1)
    else:
        titles = list(title)

    if backend == 'matplotlib':
        return _plot_contour_matplotlib(
            data, titles, colorscale, log_scale,
            cbar_title, n_grid, show, **kwargs,
        )
    elif backend == 'plotly':
        return _plot_contour_plotly(
            data, titles, colorscale, log_scale,
            cbar_title, n_grid, show, **kwargs,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'matplotlib' or 'plotly'.")


# ---------------------------------------------------------------------------
# IVData plotting mixin
# ---------------------------------------------------------------------------

class IVDataPlotMixin:
    """
    Mixin class to add plotting methods to IVData.

    This mixin provides convenient plotting methods that can be added to the
    IVData class for direct visualization of simulation results.
    """

    def plot(
        self,
        electrode: int,
        title: Optional[str] = None,
        log_scale: bool = False,
        backend: Optional[str] = None,
        show: bool = True,
        **kwargs
    ) -> Any:
        """
        Plot I-V data for a specific electrode.

        Parameters
        ----------
        electrode : int
            Electrode number
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
            Plot object
        """
        voltages, currents = self.get_iv_data(electrode)
        title = title or f"I-V Characteristic - Electrode {electrode}"
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
    ) -> Any:
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
            Plot object
        """
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
    ) -> Any:
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
            Plot object
        """
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
    ) -> Any:
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
            Plot object
        """
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
