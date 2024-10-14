import numpy as np


def sop_full(reservoir, water_flows):

    """
    This function performs the water balance. Arguments are:
        reservoir: an instance of the Reservoir class
        water_flows: a pandas DataFrame that must contain inflows and demands.
    The function returns an updated water_flows DataFrame.
    """

    # Local variable: number of time steps
    t_total = len(water_flows)

    # Local variable: number of seconds in a day
    n_sec = 86400

    # For computing efficiency: convert flows to numpy arrays outside of time loop

    # Inflows (in m3)
    inflows = water_flows['Total inflows (m3/s)'].to_numpy() * n_sec

    # Total downstream demand (in m3)
    downstream_demands = np.zeros(len(water_flows))
    for i in range(len(reservoir.demand_downstream)):
        # Get column with that demand
        demand_col = ([col for col in water_flows.columns if reservoir.demand_downstream[i].name in col])
        # Add this demand to total demand
        downstream_demands = downstream_demands + water_flows.loc[:, demand_col[0]].to_numpy()
    downstream_demands = downstream_demands * n_sec  # conversion to m3

    # Total at-site demands (in m3)
    at_site_demands = np.zeros((len(water_flows), len(reservoir.demand_on_site)))
    for i in range(len(reservoir.demand_on_site)):
        # Get column with that demand
        demand_col = ([col for col in water_flows.columns if reservoir.demand_on_site[i].name in col])
        at_site_demands[:, i] = water_flows.loc[water_flows.index, demand_col[0]]
    at_site_demands = at_site_demands * n_sec  # conversion to m3

    # Initialise outputs
    # Storage needs to account for initial storage
    storage = np.zeros(t_total + 1)
    storage[0] = reservoir.initial_storage
    # Initialise at-site withdrawals and outflows as water balance components
    withdrawals = np.zeros((t_total, len(reservoir.demand_on_site)))
    outflows = np.zeros(t_total)

    # Main loop
    for t in range(t_total):

        wb_out = sop_single_step(reservoir, storage[t], inflows[t], at_site_demands[t, :], downstream_demands[t])
        storage[t+1] = wb_out[0]
        outflows[t] = wb_out[1]
        withdrawals[t, :] = wb_out[2]

    # Insert data into water balance (mind the flow rates conversions back into m3/s)
    for i in range(withdrawals.shape[1]):
        water_flows['Withdrawals ' + reservoir.demand_on_site[i].name + ' (m3/s)'] = withdrawals[:, i] / n_sec
    water_flows['Outflows (m3/s)'] = outflows / n_sec
    water_flows['Storage (m3)'] = storage[1:]

    return water_flows


def sop_single_step(reservoir, storage_beg, inflows, site_demand, downstream_demand):

    """
    Note all in m3.
    :param reservoir: Object of the Reservoir class
    :param storage_beg: Initial storage at the beginning of the time step (m3)
    :param inflows: Inflows over the time step (m3)
    :param site_demand: Demand for withdrawal from reservoir over the time step (m3). Vector with length the number of demands
    :param downstream_demand: Demand for release for downstream use over the time step (m3)
    :return: storage_end (end of time step storage, m3)
    :return: outflows (amount of water released over time step, m3)
    :return: withdrawals (to meet demand over time step at reservoir, m3)
    """

    # Compute water availability, accounting for dead storage (volume units)
    water_available = storage_beg - reservoir.dead_storage + inflows

    # Release for downstream demand (volumetric rate)
    outflows = np.min([water_available, downstream_demand])

    # Update water availability
    water_available = water_available - outflows

    # Height of water available in the reservoir, computed with height=0 when reservoir is empty
    height = reservoir.get_height(water_available + reservoir.dead_storage)

    # Initialise withdrawals FOR EACH DEMAND SOURCE
    withdrawals = np.zeros(len(reservoir.demand_on_site))

    # Compute on-site withdrawals FOR EACH DEMAND SOURCE
    for i in range(len(reservoir.demand_on_site)):

        # Check abstraction is possible
        if height + reservoir.demand_on_site[i].intake_depth > reservoir.total_lake_depth:
            # Withdrawals for downstream demand (volumetric rate)
            withdrawals[i] = np.min([water_available, site_demand[i]])
            # Update water availability
            water_available = water_available - withdrawals[i]

    # Check if reservoir is over full
    if water_available + reservoir.dead_storage > reservoir.full_lake_volume:
        # Lake is full
        storage_end = reservoir.full_lake_volume
        # Excess storage is spilled
        outflows = outflows + (water_available + reservoir.dead_storage - reservoir.full_lake_volume)
    else:
        # Lake is not full so water availability determines new storage
        storage_end = water_available + reservoir.dead_storage

    return storage_end, outflows, withdrawals