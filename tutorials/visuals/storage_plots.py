import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    s_min, = ax.plot(balance.index, reservoir.dead_storage * np.ones(len(balance)), c='black', linestyle='--',
                     linewidth=2, label='Dead storage')
    ax.legend(handles=[s, s_min], loc=4)
    ax.set_xlabel('Date', size=14)
    ax.set_ylabel('Storage (m3)', size=14)
    ax.set_xlim(first_date, last_date)

    return fig


def compare_timeseries(reservoir, storage_1, storage_2, labels, **kwargs):
    """
    Plots daily storage over time. Arguments:
        reservoir: an instance of the Reservoir class whose storage is being plotted
        storage_1: a Pandas Series containing a first time series of storage
        storage_2: a Pandas Series containing a second time series of storage
        labels: a list of Strings for legend labels, one for each of the time series above
        optional argument `first_date`: a datetime date to specify when to start plotting
        optional argument `last_date`: a datetime date to specify when to stop plotting
        optional argument `storage_3`: a Pandas Series containing a third time series of storage
    Returns the matplotlib figure created, for plotting / saving, etc.
    """

    # Optional arguments
    first_date = kwargs.pop("first_date", storage_1.index[0])
    last_date = kwargs.pop('last_date', storage_1.index[-1])
    # If there are only two time series to compare
    dummy_array = np.empty(5)
    dummy_array[:] = np.nan
    storage_3 = kwargs.pop('storage_3', pd.Series(dummy_array))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(storage_1.index, storage_1, c='b', linewidth=2, label=labels[0])
    ax.plot(storage_1.index, storage_2, c='r', linewidth=2, label=labels[1])
    if storage_3.hasnans is False:  # There is a third time series
        ax.plot(storage_1.index, storage_3, c='k', linewidth=2, label=labels[2])
    ax.plot(storage_1.index, reservoir.dead_storage * np.ones(len(storage_1)), c='black', linestyle='--',
            linewidth=2, label='Dead storage')
    legend = ax.legend(loc=4)
    ax.set_xlabel('Date', size=14)
    ax.set_ylabel('Storage (m3)', size=14)
    ax.set_xlim(first_date, last_date)

    return fig

