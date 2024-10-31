from .balance_calcs import local_demand_init, downstream_demand_init, sop_single_step
import pandas as pd
import numpy as np
import datetime


def backward_hp_max(reservoir, water_flows, first_year, nb_states, nb_decisions, **kwargs):
    """
    Backward dynamic programming, for a 2-year period by default, and on a daily time step.
    The dynamic programming aims to maximise hydropower production.
    Note that here is no restriction on still meeting all demands as long as there's water.
    :param reservoir: object of Reservoir class
    :param water_flows: pandas DataFrame, with the input water balance components (inflows, demands)
    :param first_year: int, year programme starts
    :param nb_states: int, number of states in discretisation of state variable
    :param nb_decisions: int, number of decisions considered in decision range
    :param kwargs: "threshold_volume" a storage volume below which policy reverts to SOP to protect demands
                   "nb_years" duration of dynamic programming, default 2
    :return: the value and release tables, as numpy arrays
    """

    # Optional arguments
    nb_years = kwargs.pop("nb_years", 2)
    threshold_volume = kwargs.pop("threshold_volume", reservoir.dead_storage)


    # Parameters
    n_sec = 86400  # Number of seconds in a day (the time step)

    # Discretisation on whole storage range
    storage_mesh = pd.Series(reservoir.dead_storage + np.arange(0, nb_states) / (nb_states - 1) *
                             (reservoir.full_lake_volume - reservoir.dead_storage))

    # Initialise inflows and demands
    time_mask = (water_flows.index.year >= first_year) & (water_flows.index.year < first_year + nb_years)
    h2o_balance = water_flows.iloc[time_mask, :].copy()
    total_local_demands = local_demand_init(reservoir, h2o_balance, n_sec)
    total_downstream_demands = downstream_demand_init(reservoir, h2o_balance, n_sec)

    # Value table
    power_table = np.zeros((len(h2o_balance) + 1, nb_states))

    # Release table
    release_table = np.zeros((len(h2o_balance), nb_states))

    # Backward loop
    for t in np.arange(len(h2o_balance ) -1, -1, -1):

        # At each time step, loop on states
        for s in range(nb_states):

            # Start with SOP, if there is spillage, no need to find a solution
            wb_sop = sop_single_step(reservoir, storage_mesh[s],
                                     h2o_balance.loc[h2o_balance.index[t], 'Total inflows (m3/s)' ] *n_sec,
                                     total_local_demands[t], total_downstream_demands[t])
            # result of SOP is the minimum release decision, leading to maximum storage among all decisions
            st_max = wb_sop[0]
            release_min = wb_sop[1]

            # If there is no spillage, see if increasing release can produce more hydropower
            if release_min / n_sec < reservoir.hydropower_plant.max_release:

                # Minimal end-of-step storage if release is maximal useful value for hydropower
                # (lower bound threshold_volume)
                st_min = max(st_max + release_min - reservoir.hydropower_plant.max_release * n_sec, threshold_volume)

                # Corresponding maximal considered release
                release_max = release_min + st_max - st_min

                # Vector of current values for each decision
                current_value = np.zeros(nb_decisions)
                for dec in range(nb_decisions):
                    current_release = release_max - (release_max - release_min) * dec / (nb_decisions - 1)
                    storage = st_max + release_min - current_release
                    hydraulic_head = reservoir.hydropower_plant.nominal_head - reservoir.total_lake_depth + \
                                     reservoir.get_height(storage)
                    current_value[dec] = 1000 * 9.81 * reservoir.hydropower_plant.efficiency * hydraulic_head * \
                                         current_release / n_sec * 24 / 1E6  # in MWh

                # Find best decision (storage, corresponding release, and score in MWh)
                x = decision_taking(current_value, power_table[t + 1, :], st_min, st_max, storage_mesh, False,
                                    nb_decisions)
                storage_decision = st_min + (x[0] / (nb_decisions -1)) * (st_max -st_min)
                release_table[t, s] = release_min + st_max - storage_decision  # Release decision
                power_table[t, s] = x[1]
            else:
                release_table[t, s] = release_min
                hydraulic_head = reservoir.hydropower_plant.nominal_head - reservoir.total_lake_depth + \
                                 reservoir.get_height(st_max)
                current_value = 1000 * 9.81 * reservoir.hydropower_plant.efficiency * hydraulic_head * \
                                reservoir.hydropower_plant.max_release * 24 / 1E6  # in MWh
                power_table[t, s] = value_update(immediate_value=current_value,
                                                 future_state=st_max,
                                                 future_value=power_table[t + 1],
                                                 mesh=storage_mesh)

    return release_table, power_table


def decision_taking(current_value, future_value, state_min, state_max, mesh, we_minimise, nb_decisions):
    """
    Finds the best decision for dynamic programming, from a discrete set of decisions to evaluate
    :param current_value: vector, current value of each decision
    :param future_value: vector of future values for discretised state values
    :param state_min: float, minimal possible future state resulting from decision
    :param state_max: float, maximal possible future state resulting from decision
    :param mesh: vector of corresponding discretised state values
    :param we_minimise: Boolean, True if minimization problem, False if maximization
    :param nb_decisions: int, number of decisions considered
    :return:
    """

    # Initialise vector of values for each possible decision
    value = np.zeros(nb_decisions)

    # Loop on discretised decisions
    for n in range(nb_decisions):
        # Find future state resulting from decision
        state = state_min + n / (nb_decisions - 1) * (state_max - state_min)
        # Find corresponding value
        value[n] = value_update(current_value[n], state, future_value, mesh)

    if we_minimise is True:
        # Best decision minimises value
        best_value = np.min(value)
        decision = np.argmin(value)
    else:
        # Best decision maximises value
        best_value = np.max(value)
        decision = np.argmax(value)

    return decision, best_value


def value_update(immediate_value, future_state, future_value, mesh):
    """
    Value calculation for a decision under dynamic programming, general formula
    :param immediate_value: float, the immediate value of the decision
    :param future_state: float, the future state of the system as a consequence of the decision
    :param future_value: vector of future values for discretised state values
    :param mesh: vector of corresponding discretised state values
    :return: total value of the decision
    """

    # Record discrete future states where future values are of interest for our decision
    mesh_step = mesh[1] - mesh[0]
    locator = mesh[mesh > future_state - mesh_step][mesh < future_state + mesh_step].index.to_numpy()

    # The future state is one of the discretised states (almost never)
    if len(locator) == 1:
        value = immediate_value + future_value[locator[0]]

    # The future state is between two discretised states (almost always)
    if len(locator) == 2:
        weight = (future_state - mesh[locator[0]]) / (mesh[locator[1]] - mesh[locator[0]])
        value = immediate_value + weight * future_value[locator[1]] + (1-weight) * future_value[locator[0]]

    # If rounding errors lead to three values
    if len(locator) == 3:
        value = immediate_value + future_value[locator[1]]

    return value


def forward_loop(reservoir, water_flows, year_beg, release_table, **kwargs):
    """
    Forward loop using the release table to determine actual sequence of release decisions for a reservoir.
        :param reservoir: object of Reservoir class
        :param water_flows: pandas DataFrame, with the input water balance components (inflows, demands)
        :param year_beg: int, year programme starts
        :param release_table: numpy array containing the release decision for each discretised state in range
        :param kwargs: "threshold_volume" a storage volume below which policy reverts to SOP to protect demands
                       "nb_years" duration of dynamic programming, default 2. Must match "nb_years" in backward phase.
        :return: the value and release tables, as numpy arrays
    """

    # Parameters
    n_sec = 86400  # Number of seconds in a day (the time step)

    # Optional arguments
    current_storage = kwargs.pop("initial_storage", reservoir.initial_storage)

    # Initialise water balance
    h2o_balance = water_flows.loc[datetime.date(year_beg, 1, 1): datetime.date(year_beg, 1, 1)
                                  + datetime.timedelta(days=len(release_table)), :].copy()
    total_local_demands = local_demand_init(reservoir, water_flows, n_sec)

    # Initialise water balance outputs
    for i in range(total_local_demands.shape[1]):
        h2o_balance['Withdrawals ' + reservoir.demand_on_site[i].name + ' (m3/s)'] = np.zeros(len(h2o_balance))
    h2o_balance['Release (m3/s)'] = np.zeros(len(h2o_balance))
    h2o_balance['Storage (m3)'] = np.zeros(len(h2o_balance))

    # Discretisation on whole storage range
    nb_states = release_table.shape[1]
    storage_mesh = pd.Series(reservoir.dead_storage + np.arange(0, nb_states) / (nb_states - 1) *
                             (reservoir.full_lake_volume - reservoir.dead_storage))
    mesh_step = storage_mesh[1] - storage_mesh[0]

    # Forward loop
    for t in range(len(release_table)):

        locator = (storage_mesh[storage_mesh > current_storage - mesh_step][storage_mesh < current_storage + mesh_step].
                   index.to_numpy())

        if len(locator) == 1:
            release = release_table[t, locator[0]]

        if len(locator) == 2:
            weight = ((current_storage - storage_mesh[locator[0]]) /
                      (storage_mesh[locator[1]] - storage_mesh[locator[0]]))
            release = weight * release_table[t, locator[1]] + (1 - weight) * release_table[t, locator[0]]

        if len(locator) == 3:
            release = release_table[t, locator[1]]

        wb_sop = sop_single_step(reservoir, current_storage,
                                 h2o_balance.loc[h2o_balance.index[t], 'Total inflows (m3/s)'] * n_sec,
                                 total_local_demands[t], release)

        # Record results
        h2o_balance.loc[h2o_balance.index[t], 'Storage (m3)'] = wb_sop[0]
        h2o_balance.loc[h2o_balance.index[t], 'Release (m3/s)'] = wb_sop[1] / n_sec
        for i in range(total_local_demands.shape[1]):
            h2o_balance.loc[h2o_balance.index[t], 'Withdrawals ' + reservoir.demand_on_site[i].name + ' (m3/s)'] = \
                wb_sop[2][i] / n_sec

        # Update storage
        current_storage = h2o_balance.loc[h2o_balance.index[t], 'Storage (m3)']

    return h2o_balance
