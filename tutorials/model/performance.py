import pandas as pd
import numpy as np


def rrv_indicators(time_series, dynamic_threshold, above_desirable, name, **kwargs):
    """
    Compute the RRV indicators for a time series vs. a threshold. Arguments:
        time_series: numpy vector
        dynamic_threshold: numpy vectors of same length as `time_series`
        above_desirable: boolean. If True we value staying at or above a threshold.
        name: String, the name of the site
        optional argument `vul_unit`: String, default as a percentage, to specify how vulnerability is evaluated
    Returns a pandas DataFrame with several perf_metrics metrics.
    """

    # Optional argument
    vul_unit = kwargs.pop("vul_unit", '%')

    # Local variables
    n_steps = len(time_series)
    tolerance = 1E-6  # for rounding errors

    # If above_desirable is false we need to change sign of all data now, so we compare a and b
    a = (2 * above_desirable - 1) * time_series
    b = (2 * above_desirable - 1) * dynamic_threshold
    b = b - tolerance

    # Initialise output
    indicators = pd.DataFrame(columns=['Name', 'Reliability (0-1)', 'Resilience (-)', 'Vulnerability', 'Failure count'])
    indicators.loc[0, 'Name'] = name

    # Reliability
    indicators.loc[0, 'Reliability (0-1)'] = 1 - np.sum(a < b) / n_steps

    # We need to count failure events to compute resilience and vulnerability
    event_count = 0
    # We also need to have the maximal amplitude or magnitude of failure
    magnitude = []
    # We use a while loop to count events and their magnitude
    t = 0
    while t < n_steps:

        if a[t] < b[t]:
            # New event! we need to update the count of failure events
            event_count = event_count + 1
            # We also need to keep track of the maximum amplitude of failure
            # By default failure is expressed in relative terms
            if vul_unit == '%':
                magnitude.append((b[t] - a[t]) / abs(b[t]))
            else:
                magnitude.append(b[t] - a[t])
            # Now while event lasts
            while a[t] < b[t]:
                t = t+1
                if t == n_steps:
                    break
                if vul_unit == '%':
                    magnitude[-1] = max(magnitude[-1], (b[t] - a[t]) / abs(b[t]))
                else:
                    magnitude[-1] = max(magnitude[-1], b[t] - a[t])

        # Time increment so while loop concludes
        t = t+1

    # Exporting the failure count
    indicators.loc[0, 'Failure count'] = event_count

    if event_count > 0:  # there are failures

        # Resilience
        indicators.loc[0, 'Resilience (-)'] = event_count / (n_steps * (1 - indicators.loc[0, 'Reliability (0-1)']))

        # Vulnerability (as a percentage)
        if vul_unit == '%':
            indicators.loc[0, 'Vulnerability'] = "{:.0f}".format(np.mean(magnitude) * 100) + '%'
        else:
            indicators.loc[0, 'Vulnerability'] = "{:.2f}".format(np.mean(magnitude)) + vul_unit

    else:

        # No failure: empirical resilience is infinite and vulnerability is 0
        indicators.loc[0, 'Resilience (-)'] = np.inf
        indicators.loc[0, 'Vulnerability'] = 0

    return indicators


def reliability(time_series, dynamic_threshold, above_desirable):
    """
    Compute reliability for a time series vs. a threshold. Arguments:
        time_series: numpy vector
        dynamic_threshold: numpy vectors of same length as `time_series`
        above_desirable: boolean. If True we value staying at or above a threshold.
    Returns a float between 0 and 1
    """

    # If above_desirable is false we need to change sign of all data now, so we compare a and b
    a = (2 * above_desirable - 1) * time_series
    b = (2 * above_desirable - 1) * dynamic_threshold
    b = b - 1E-6  # for rounding errors

    return 1 - np.sum(a < b) / len(time_series)


def all_metrics(reservoir, water_balance):

    metrics = pd.concat([rrv_indicators(water_balance['Withdrawals Baltimore (m3/s)'].to_numpy(),
                                        water_balance['Baltimore demand (m3/s)'].to_numpy(), True, 'Baltimore'),
                         rrv_indicators(water_balance['Withdrawals Chester (m3/s)'].to_numpy(),
                                        water_balance['Chester demand (m3/s)'].to_numpy(), True, 'Chester'),
                         rrv_indicators(water_balance['Withdrawals Nuclear plant (m3/s)'].to_numpy(),
                                        water_balance['Nuclear plant demand (m3/s)'].to_numpy(), True, 'Nuclear'),
                         rrv_indicators(water_balance['Release (m3/s)'].to_numpy(),
                                        water_balance['Environmental demand (m3/s)'].to_numpy(), True, 'Env. flows'),
                         rrv_indicators(water_balance['Release (m3/s)'].to_numpy(),
                                        15000 * np.ones(len(water_balance)), False, 'Flooding')],
                        axis=0, ignore_index=True)

    # Summer recreation (lake levels need to stay above a certain level in June, July and August)

    # We need time series of level objectives. We initialise at 0 requirement.
    level_objective = pd.Series(index=water_balance.index, data=np.zeros(len(water_balance)))

    # We set a level during summer months, to be compared with lake level (which coincide with hydraulic head)
    summer_requirement = 106.5 * 0.3048
    for month in np.arange(6, 9, 1):
        level_objective[level_objective.index.month == month] = summer_requirement

    # Get hydraulic head time series, assuming linear relationship between depth and lake area
    hydraulic_head = np.zeros(len(water_balance))
    for t in range(len(water_balance)):
        depth = reservoir.get_height(water_balance.iloc[t, -1])
        hydraulic_head[t] = reservoir.hydropower_plant.nominal_head - reservoir.total_lake_depth + depth

    # Get the indicators
    recreation_metrics = rrv_indicators(hydraulic_head, level_objective.to_numpy(), True, 'Recreation', vul_unit='m')

    # We need to account for the fact that this requirement is for three months only, which impacts reliability
    # Failure happens more often if measured in the shorter time window
    recreation_metrics.iloc[0, 1] = 1 - (1 - recreation_metrics.iloc[0, 1]) * len(level_objective) / (
                70 * (30 + 31 + 31))

    metrics = pd.concat([metrics, recreation_metrics], axis=0, ignore_index=True)

    # Add a new column, volumetric reliability
    metrics.insert(5, 'Volumetric reliability', [0, 0, 0, 0, 'N/A', 'N/A'])

    # Volumetric reliability is only defined for the demands, and it relies on the grand total supply / demand
    totals = water_balance.sum(axis=0)

    metrics.loc[0, 'Volumetric reliability'] = totals['Withdrawals Baltimore (m3/s)'] / totals[
        'Baltimore demand (m3/s)']
    metrics.loc[1, 'Volumetric reliability'] = totals['Withdrawals Chester (m3/s)'] / totals['Chester demand (m3/s)']
    metrics.loc[2, 'Volumetric reliability'] = totals['Withdrawals Nuclear plant (m3/s)'] / totals[
        'Nuclear plant demand (m3/s)']
    metrics.loc[3, 'Volumetric reliability'] = np.sum(
        np.minimum(water_balance['Environmental demand (m3/s)'], water_balance['Release (m3/s)'])) / totals[
                                                   'Environmental demand (m3/s)']

    return metrics


def regret(x, ref_value, more_desired):
    """
    x: vector of performance, numpy array
    ref_value: float, reference value from which to compute regret
    more_desired: boolean, if True, lower values lead to more regret
    """

    if more_desired is True:
        return (ref_value - x) / ref_value
    else:
        return (x - ref_value) / ref_value
