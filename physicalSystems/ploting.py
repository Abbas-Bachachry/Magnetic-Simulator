import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _plot_bounds(x):
    fine_x_index = np.where(np.isfinite(x))[0]
    fine_x = x[fine_x_index]
    if fine_x.min() < 0:
        lower = 1.2 * fine_x.min()
    elif fine_x.min() > 0:
        lower = .8 * fine_x.min()
    else:
        lower = 0
    if fine_x.max() > 0:
        upper = 1.2 * fine_x.max()
    elif fine_x.max() < 0:
        upper = .8 * fine_x.max()
    else:
        upper = 0
    if lower == 0 and upper == 0:
        lower, upper = -.2, .2
    elif lower == 0:
        lower = -.2 * upper
    elif upper == 0:
        upper = -.2 * lower

    return lower, upper


def _adjust_plot_bounds(bound: tuple[int | float, int | float], axs_bound):
    lower = bound[0] if bound[0] < axs_bound[0] else axs_bound[0]
    upper = bound[1] if bound[1] > axs_bound[1] else axs_bound[1]
    return lower, upper


def plot_data(*args, **kwargs):
    if len(args) == 2:
        return plot_xy(*args, **kwargs)
    elif len(args) == 3:
        return plot_contours_xyz(*args, **kwargs)
    elif len(args[0].shape) == 3:
        matrix = args[0]
        # get keyword arguments
        type = kwargs.get('type', 'Curves')
        type = type.capitalize()
        if type == 'Curves':
            return plot_curves(matrix, **kwargs)
        elif type == 'Contours':
            return plot_contours(matrix, **kwargs)
    elif len(args[0].shape) == 2:
        data = args[0]
        # get x and y from the data
        x = data[:, 0]
        y = data[:, 1]
        return plot_xy(x, y, **kwargs)


def plot_xy(x, y, **kwargs):
    # get keyword arguments
    label = kwargs.get('label', '')
    title = kwargs.get('title', 'hysteresis loop')
    show = kwargs.get('show', True)
    xlabel = f"{kwargs.get('x_name', 'H')} ({kwargs.get('x_unit', 'Oe')})"
    ylabel = f"{kwargs.get('y_name', 'M')} ({kwargs.get('y_unit', 'emu/g')})"
    mc = kwargs.get('mc', 'k')
    fig_axs = kwargs.get('fig_axs')
    if fig_axs is None:
        fig, axs = plt.subplots()
    else:
        fig, axs = fig_axs
    axs: Axes
    fig: Figure

    # get the upper and lower rage of axes
    # X-axis
    lower_x, upper_x = _plot_bounds(x)
    # Y-axis
    lower_y, upper_y = _plot_bounds(y)
    # _adjust axes limit
    if fig_axs is not None:
        lower_x, upper_x = _adjust_plot_bounds((lower_x, upper_x), axs.get_xlim())
        lower_y, upper_y = _adjust_plot_bounds((lower_y, upper_y), axs.get_ylim())

    # plot the data
    axs.plot(x, y, mc, label=label)
    # **** adjust axes ****
    # adjust X-axis
    axs.set_xlim(lower_x, upper_x)
    axs.hlines(0, lower_x, upper_x, 'k')
    # adjust Y-axis
    axs.set_ylim(lower_y, upper_y)
    axs.vlines(0, lower_y, upper_y, 'k')
    # **** adjust the graph ****
    axs.set_title(title)
    if label:
        axs.legend()
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    # **** show the graph ****
    if show:
        fig.show()
    return fig, axs


def plot_curves(matrix, **kwargs):
    # get x and y from the data
    x = matrix[0, :, 0]  # get filed
    y = matrix[:, :, 1]

    # get keyword arguments
    label = kwargs.get('label', '')
    title = kwargs.get('title', 'First-Order Reversal Curves')
    show = kwargs.get('show', True)
    xlabel = f"{kwargs.get('x_name', 'H')} ({kwargs.get('x_unit', 'Oe')})"
    ylabel = f"{kwargs.get('y_name', 'M')} ({kwargs.get('y_unit', 'a.u.')})"

    # get the upper and lower rage of axes
    # X-axis
    lower_x = 1.2 * x.min()
    upper_x = 1.2 * x.max()
    # Y-axis
    finite_index = np.where(np.isfinite(y))
    lower_y = 1.2 * y[finite_index].min()
    upper_y = 1.2 * y[finite_index].max()

    # plot the data
    fig, axs = plt.subplots()
    axs: Axes
    fig: Figure
    for curve in y:
        line = axs.plot(x, curve, 'b')
    # **** adjust axes ****
    # adjust X-axis
    axs.set_xlim(lower_x, upper_x)
    axs.hlines(0, lower_x, upper_x, 'k')
    # adjust Y-axis
    axs.set_ylim(lower_y, upper_y)
    axs.vlines(0, lower_y, upper_y, 'k')
    # **** adjust the graph ****
    axs.set_title(title)
    if label:
        line[0].set_label(label)
        axs.legend()
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    # **** show the graph ****
    if show:
        fig.show()
    return fig, axs


def plot_contours(matrix, **kwargs):
    # get x and y from the data
    x = matrix[0, :, 0]
    y = matrix[:, :, 1]

    # get keyword arguments
    title = kwargs.get('title', 'First-Order Reversal Curves')
    show = kwargs.get('show', True)
    xlabel = f"{kwargs.get('x_name', 'H')} ({kwargs.get('x_unit', 'Oe')})"
    ylabel = f"{kwargs.get('y_name', 'H')} ({kwargs.get('y_unit', 'Oe')})"

    # get the upper and lower rage of axes
    # X-axis
    lower = x.min()
    upper = x.max()

    # plot the data
    fig, axs = plt.subplots()
    axs: Axes
    fig: Figure
    axs.contourf(x, x, y)
    # **** adjust axes ****
    # adjust X-axis
    axs.set_xlim(lower, upper)
    axs.hlines(0, lower, upper, 'k')
    axs.plot(x[x.size // 2:], -x[x.size // 2:], 'k')
    # adjust Y-axis
    axs.set_ylim(lower, upper)
    axs.vlines(0, lower, upper, 'k')
    axs.axline((0, 0), (1, 1), color='k')
    # **** adjust the graph ****
    axs.minorticks_off()
    axs.set_title(title)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    # **** show the graph ****
    if show:
        plt.show()
    return fig, axs


def plot_contours_xyz(x, y, z, **kwargs):
    # get keyword arguments
    title = kwargs.get('title', 'First-Order Reversal Curves')
    show = kwargs.get('show', True)
    xlabel = f"{kwargs.get('x_name', 'H')} ({kwargs.get('x_unit', 'Oe')})"
    ylabel = f"{kwargs.get('y_name', 'H')} ({kwargs.get('y_unit', 'Oe')})"
    # get the upper and lower rage of axes
    # X-axis
    xlim = x.min(), x.max()
    # Y-axis
    ylim = y.min(), y.max()

    # plot the data
    fig, axs = plt.subplots()
    axs: Axes
    fig: Figure
    axs.contourf(x, y, z)
    # **** adjust axes ****
    # adjust X-axis
    axs.set_xlim(*xlim)
    axs.hlines(0, *xlim, 'k').set_linewidth(.5)
    # adjust Y-axis
    axs.set_ylim(*ylim)
    axs.vlines(0, *ylim, 'k').set_linewidth(.5)
    # **** adjust the graph ****
    axs.minorticks_off()
    axs.set_title(title)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    # **** show the graph ****
    if show:
        plt.show()
    return fig, axs


def plot_more(data: list[dict] | dict, fig=None, **kwargs):
    show = kwargs.get('show', False)
    axs_lim: list[tuple | None] = kwargs.get('axs_lim', [None, None])
    if isinstance(data, dict):
        data = [data]
    # get x and y from the data
    for kwargd in data:
        x = kwargd.get('x')
        assert x is not None, 'missing x data'
        y = kwargd.get('y')
        assert y is not None, 'missing y data'
        axs = kwargd.get('axs')
        label = kwargd.get('label', '')
        mc = kwargd.get('mc', '-')
        # plot the data
        axs.plot(x, y, mc, label=label)
        if label:
            axs.legend()

    # adjust X-axis
    if axs_lim[0] is not None:
        plt.xlim(*axs_lim[0])
        plt.hlines(0, *axs_lim[0], 'k')
    # adjust Y-axis
    if axs_lim[1] is not None:
        plt.ylim(*axs_lim[1])
        plt.vlines(0, *axs_lim[1], 'k')
    # **** show the graph ****
    if show:
        if fig:
            fig.show()
        else:
            plt.show()
