import matplotlib.pyplot as plt
import numpy as np


def timeseries(reservoir, balance, **kwargs):
    """
    Plots daily storage over time. Arguments:
        reservoir: an instance of the Reservoir class whose storage is being plotted
        balance: a Pandas DataFrame containing the time series of storage
        optional argument `first_date`: a datetime date to specify when to start plotting
        optional argument `last_date`: a datetime date to specify when to stop plotting
    Returns the matplotlib figure created, for plotting / saving, etc.
    """

    # Optional arguments
    first_date = kwargs.pop("first_date", balance.index[0])
    last_date = kwargs.pop('last_date', balance.index[-1])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    s, = ax.plot(balance.index, balance['Storage (m3)'], c='b', linewidth=2, label='Storage')
    smin, = ax.plot(balance.index, reservoir.dead_storage * np.ones(len(balance)), c='black', linestyle='--',
                    linewidth=2, label='Dead storage')
    legend = ax.legend(handles=[s, smin], loc=4)
    ax.set_xlabel('Date', size=14)
    ax.set_ylabel('Storage (m3)', size=14)
    ax.set_xlim(first_date, last_date)

    return fig
