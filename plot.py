from math import ceil
from typing import Optional

import matplotlib as mpl
from matplotlib.backends.backend_template import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np

from .protocol import evaluate
from .units import Quantity, format_unit


COLOR_DEEMPH = "dimgray"
mpl.rcParams["xtick.color"] = COLOR_DEEMPH
mpl.rcParams["ytick.color"] = COLOR_DEEMPH
mpl.rcParams["xtick.labelcolor"] = COLOR_DEEMPH
mpl.rcParams["ytick.labelcolor"] = COLOR_DEEMPH
mpl.rcParams["figure.autolayout"] = True
DEFAULT_PTS_PER_SEGMENT = 100


def hide_spines(ax):
    for s in ("left", "right", "top", "bottom"):
        ax.spines[s].set_visible(False)


def plot_protocol(protocol, abscissa="t", n=DEFAULT_PTS_PER_SEGMENT):
    """Return plot of entire test protocol"""
    if isinstance(abscissa, str):
        xvar = protocol.variables[abscissa]
    else:
        xvar = abscissa
    ny = len(protocol.variables) - 1
    fig = Figure(figsize=(11, 1 + ny * 1.5))
    FigureCanvas(fig)
    gs = GridSpec(ny, 1, figure=fig)
    # Plot protocol
    state = evaluate(protocol.initial_state, protocol.parameters)
    times = [state[xvar]]
    for seg in protocol.segments:
        target = seg.target_state(state, protocol.parameters)
        t0 = state[xvar]
        t1 = target[xvar]
        t = np.linspace(t0, t1, n)
        times.append(t[1:])
        state = target
    times = np.hstack(times)
    states = protocol.eval_states(xvar, times, protocol.parameters)
    i = 0
    axarr = []
    axes = {}
    for nm in protocol.variables:
        var = protocol.variables[nm]
        if var == xvar:
            continue
        if axarr:
            ax = fig.add_subplot(gs[i, 0], sharex=axarr[0])
        else:
            ax = fig.add_subplot(gs[i, 0])
        y = np.array(
            [s[var].to(var.units).m if s[var] is not None else np.nan for s in states]
        )
        t = times.to(xvar.units).m
        ax.plot(t, y, "-k", label="Protocol")
        # Plot a marker at every constrained ↔ free transition
        m = np.zeros(y.shape, dtype=bool)
        d = np.diff(np.isnan(y).astype(int))
        # Constrained → free
        m[:-1][d == 1] = True
        # Free → constrained
        m[1:][d == -1] = True
        ax.plot(t[m], y[m], "k", marker=".", linestyle="none")
        # Format plot
        ax.set_ylabel(f"{nm} [{format_unit(var.units)}]")
        hide_spines(ax)
        if i != (ny - 1):
            ax.xaxis.set_visible(False)
        axarr.append(ax)
        i += 1
        axes[nm] = ax
    axarr[-1].set_xlabel(f"{abscissa} [{format_unit(xvar.units)}]")
    return fig, axes


def translate_data(protocol, data, varmap={}):
    """Return renamed data variables with correct units"""

    def split_colname(col):
        """Split 'name [unit]' column name into (name, unit)"""
        col = col.strip()
        if col.endswith("]"):
            i = col.rfind("[")
            if i != -1:
                nm = col[:i].rstrip()
                unit = col[i + 1 : -1]
                return nm, unit
        return col, None

    varmapr = {v: k for k, v in varmap.items()}
    tdata = {}
    for col in data.columns:
        dvar, dunit = split_colname(col)
        nm = varmapr.get(dvar, dvar)
        tdata[nm] = Quantity(data[col].values, dunit)
    out = {}
    for nm, var in protocol.variables.items():
        out[nm] = tdata.get(nm, Quantity(np.full(len(data), np.nan), var.units))
    return out


def check_data_all(protocol, data, abscissa="t", varmap={}, n=DEFAULT_PTS_PER_SEGMENT):
    """Return plot comparing entire test protocol with data"""
    fig, axes = plot_protocol(protocol, abscissa=abscissa, n=n)
    data = translate_data(protocol, data, varmap)
    # Handle time / abscissa
    t = data[abscissa].to(protocol.variables[abscissa].units).m
    for var, ax in axes.items():
        y = data[var].to(protocol.variables[var].units).m
        ax.plot(t, y, ":", color="firebrick", label="Data")
        ax.legend()
    return fig, axes


def check_data_by_transition(
    protocol,
    data,
    abscissa="t",
    varmap={},
    n=DEFAULT_PTS_PER_SEGMENT,
    t_span: Optional[Quantity] = None,
):
    """Return plot comparing test protocol with data at each segment transition"""
    if isinstance(abscissa, str):
        xvar = protocol.variables[abscissa]
    else:
        xvar = abscissa
    data = translate_data(protocol, data, varmap)
    data_t = data[abscissa].to(protocol.variables[abscissa].units).m
    nvar = len(protocol.variables) - 1
    nseg = len(protocol.segments)
    nplt_w = 4
    nplt_h = ceil(nseg / nplt_w)
    szplt_w = 4.0
    szplt_h = 1 + 1.5 * nvar
    fig = Figure(figsize=(nplt_w * szplt_w, nplt_h * szplt_h))
    gs0 = GridSpec(nplt_h, nplt_w)
    segs = [protocol.segments[0]]
    trans_states = [
        evaluate(protocol.initial_state, protocol.parameters),
        segs[0].target_state(protocol.initial_state, protocol.parameters),
    ]
    for i, seg in enumerate(protocol.segments[1:]):
        segs.append(seg)
        trans_states.append(segs[1].target_state(trans_states[1], protocol.parameters))
        t_a = trans_states[0][xvar]
        t_b = trans_states[1][xvar]
        t_c = trans_states[2][xvar]
        if t_span is None:
            t_span = 0.5 * min(t_b - t_a, t_c - t_b)
        t_left = np.linspace(t_b - t_span, t_b, n // 2 + n % 2)
        t_right = np.linspace(t_b, t_b + t_span, n // 2)[1:]
        states = [
            segs[0].eval_state(xvar, t, trans_states[0], protocol.parameters)
            for t in t_left
        ] + [
            segs[1].eval_state(xvar, t, trans_states[1], protocol.parameters)
            for t in t_right
        ]
        t = np.hstack([t_left, t_right]).to(xvar.units).m
        gs = GridSpecFromSubplotSpec(
            nvar, 1, subplot_spec=gs0[np.unravel_index(i, (nplt_h, nplt_w))]
        )
        j = 0
        for nm, var in protocol.variables.items():
            if var == xvar:
                continue
            # Plot protocol
            y = np.array(
                [
                    s[var].to(var.units).m if s[var] is not None else np.nan
                    for s in states
                ]
            )
            ax = fig.add_subplot(gs[j, 0])
            ax.plot(t, y, "-k", label="Protocol")
            # Plot data
            m = (t[0] <= data_t) & (data_t <= t[-1])
            m = np.hstack([[False], m, [False]])
            m = (m | np.roll(m, -1) | np.roll(m, 1))[1:-1]
            data_y = data[var.name].to(var.units).m
            ax.plot(
                data_t[m], data_y[m], ":", marker=".", color="firebrick", label="Data"
            )
            # Format plot
            if j == 0:
                ax.legend()
                ax.set_title(f"Segment {i} → {i + 1}")
            ax.set_ylabel(f"{nm} [{format_unit(var.units)}]")
            hide_spines(ax)
            if j == nvar - 1:
                ax.set_xlabel(f"{abscissa} [{format_unit(xvar.units)}]")
            else:
                ax.xaxis.set_visible(False)
            j += 1
        del trans_states[0]
        del segs[0]
    return fig
