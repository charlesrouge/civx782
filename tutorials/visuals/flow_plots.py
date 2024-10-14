import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def timeseries(balance, flux_name, **kwargs):
    """
    Plots daily timeseries of a water balance flow component over time. Arguments:
        balance: a Pandas DataFrame containing the time series of the water flux to plot
        flux_name: a string with the name of the flow component to plot
        optional argument `first_date`: a datetime date to specify when to start plotting
        optional argument `last_date`: a datetime date to specify when to stop plotting
    Returns the matplotlib figure created, for plotting / saving, etc.
    """

    # Optional arguments
    first_date = kwargs.pop("first_date", balance.index[0])
    last_date = kwargs.pop('last_date', balance.index[-1])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(pd.date_range(start=first_date, end=last_date, freq='D'),
            balance.loc[first_date:last_date, flux_name + ' (m3/s)'], c='b', linewidth=2)
    ax.set_xlabel('Date', size=14)
    ax.set_ylabel(flux_name + ' (m3/s)', size=14)
    ax.set_xlim(first_date, last_date)
    ax.set_ylim(0,  balance.loc[first_date:last_date, flux_name + ' (m3/s)'].max()*1.1)

    return fig


def compare_timeseries(reference, alternative, labels, **kwargs):
    """
    Plots daily timeseries of a water balance flow component over time. Arguments:
        reference: a Pandas Series containing the first time series to plot
        alternative: a Pandas Series containing the second time series to plot
        labels: a list of two Strings for legend labels for the two time series above.
        optional argument `first_date`: a datetime date to specify when to start plotting
        optional argument `last_date`: a datetime date to specify when to stop plotting
        optional argument `alternative_2`: a Pandas Series containing a third time series to plot
        Returns the matplotlib figure created, for plotting / saving, etc.
    """

    # Optional arguments
    first_date = kwargs.pop("first_date", reference.index[0])
    last_date = kwargs.pop('last_date', reference.index[-1])
    # If there are only two time series to compare
    dummy_array = np.empty(5)
    dummy_array[:] = np.nan
    alternative_2 = kwargs.pop('alternative_2', pd.Series(dummy_array))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(reference.index, reference, c='b', linewidth=2, label=labels[0])
    ax.plot(reference.index, alternative, c='r', linewidth=2, label=labels[1])
    if alternative_2.hasnans is False:  # There is a third time series
        ax.plot(reference.index, alternative_2, c='k', linewidth=2, label=labels[2])
    ax.set_xlabel('Date', size=14)
    ax.set_ylabel(alternative.name, size=14)
    ax.set_xlim(first_date, last_date)
    ax.set_ylim(0, reference.loc[first_date:last_date].max()*1.1)
    ax.legend()

    return fig


def compute_monthly_average(flows):
    """
    Computes monthly average inflows from a `flows` pandas Series.
    Output:
    averages: a Numpy vector of size 12 for the 12 average monthly values
    """

    # Initialise output
    averages = np.zeros(12)

    # Main loop to compute all 12 monthly averages
    for month in np.arange(1, 13, 1):
        monthly_mask = flows.index.month == month  # Select only values for the right month
        averages[month - 1] = flows.loc[monthly_mask].mean()  # Apply average operator

    return averages


def monthly_averages(flows, **kwargs):
    """
    Plot monthly average inflows from `flows` pandas Series.
    """

    # Optional argument
    yaxis_label = kwargs.pop('yaxis_label', 'Average inflows (m3/s)')

    # Get monthly average inflows
    monthly_average = compute_monthly_average(flows)

    # Plot figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(1, 13, 1), monthly_average, c='b')
    plt.xticks(ticks=np.arange(1, 13, 1), labels=['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax.set_xlabel('Month', size=14)
    ax.set_ylabel(yaxis_label, size=14)
    ax.set_xlim(1, 12)

    return fig


def compare_monthly_averages(reference, alternative, labels, **kwargs):
    """
    Plot a comparison of monthly average inflows from two time series. Arguments:
        reference: pandas DataFrame containing a column with reference inflows
        alternative: pandas DataFrame containing a column with alternative inflows
        labels: list of two Strings, the labels to insert in the figure's legend
    """

    # Optional arguments: third time series to compare
    dummy_array = np.empty(5)
    dummy_array[:] = np.nan
    alternative_2 = kwargs.pop('alternative_2', pd.Series(dummy_array))

    # First, compute monthly averages for both DataFrames
    average_1 = compute_monthly_average(reference)
    average_2 = compute_monthly_average(alternative)
    if alternative_2.hasnans is False:  # There is a third time series
        average_3 = compute_monthly_average(alternative_2)

    # Plot figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(1, 13, 1), average_1, c='b', label=labels[0])
    ax.plot(np.arange(1, 13, 1), average_2, c='r', label=labels[1])
    if alternative_2.hasnans is False:  # There is a third time series
        ax.plot(np.arange(1, 13, 1), average_3, c='k', label=labels[2])
    plt.xticks(ticks=np.arange(1, 13, 1), labels=['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax.set_xlabel('Month', size=14)
    ax.set_ylabel('Average inflows (m3/s)', size=14)
    ax.set_xlim(1, 12)
    ax.legend()

    return fig
