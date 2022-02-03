import matplotlib as mpl
from matplotlib.backends.backend_template import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import numpy as np

from .protocol import evaluate
from .units import Quantity, format_unit


COLOR_DEEMPH = "dimgray"
mpl.rcParams["xtick.color"] = COLOR_DEEMPH
mpl.rcParams["ytick.color"] = COLOR_DEEMPH
mpl.rcParams["xtick.labelcolor"] = COLOR_DEEMPH
mpl.rcParams["ytick.labelcolor"] = COLOR_DEEMPH
mpl.rcParams["figure.autolayout"] = True


def hide_spines(ax):
    for s in ("left", "right", "top", "bottom"):
        ax.spines[s].set_visible(False)


def plot_protocol(protocol, abscissa="t", n=20):
    """Return plot of test protocol compared with data"""
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


def compare_protocol(protocol, data, abscissa="t", varmap={}, n=20):
    """Return plot of test protocol compared with data"""
    fig, axes = plot_protocol(protocol, abscissa=abscissa, n=n)

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

    datavars = {}
    for col in data.columns:
        dvar, dunit = split_colname(col)
        datavars[dvar] = (col, dvar, dunit)
    # Handle time / abscissa
    dvar = varmap[abscissa]
    col, dvar, dunit = datavars[dvar]
    t = Quantity(data[col].values, dunit).to(protocol.variables[abscissa].units).m
    for var, ax in axes.items():
        if var not in varmap:
            continue
        col, dvar, dunit = datavars[varmap[var]]
        y = Quantity(data[col].values, dunit).to(protocol.variables[var].units).m
        ax.plot(t, y, ":", color="firebrick", label="Data")
        ax.legend()
    return fig, axes
