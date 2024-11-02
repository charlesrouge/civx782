import numpy as np
import pandas as pd


# Define reservoir class
class Reservoir:

    # Define attributes specific to each Reservoir object
    def __init__(self, name, lake_area, lake_volume, dead_storage=0, hydropower_plant=None):
        self.name = name
        self.full_lake_area = lake_area  # Surface area of the lake at full capacity in square metres (m2)
        self.full_lake_volume = lake_volume  # Volume of water at full capacity in cubic metres (m3)

        # Attributes deduced from others
        # Assuming a simplified linear relationship between water level and surface area
        self.total_lake_depth = self.full_lake_volume / (self.full_lake_area / 2)
        self.initial_storage = 0.9 * self.full_lake_volume  # Initialize storage at 90% full

        # Optional attribute with default value of 0: dead storage volume, the volume below which release is impossible
        # By default dead storage is empty storage
        self.dead_storage = dead_storage

        # Optional attribute , add a hydropower plant
        self.hydropower_plant = hydropower_plant

        # Initialise demands
        self.demand_on_site = []
        self.demand_downstream = []

    # Method to calculate the current surface area based on current volume (in m3)
    def get_surface_area(self,  volume):
        # Make sure volume is bounded
        current_volume = np.minimum(self.full_lake_volume, np.maximum(0, volume))
        return np.sqrt(2 * current_volume * self.full_lake_area / self.total_lake_depth)

    # Method to calculate the current water height based on the current volume (in m3)
    def get_height(self, volume):
        # Make sure volume is bounded
        return 2 * np.divide(np.minimum(self.full_lake_volume, np.maximum(0, volume)), self.get_surface_area(volume))

    def area_from_height(self, height):
        return self.full_lake_area * height / self.total_lake_depth

    def volume_from_height(self, height):
        return height * self.area_from_height(height) / 2

    # Method to add an on-site demand
    def add_on_site_demand(self, demand):
        self.demand_on_site.append(demand)
        return None

    # Method to add a demand downstream of the reservoir
    def add_downstream_demand(self, demand):
        self.demand_downstream.append(demand)
        return None

    # Method to calculate daily hydropower production from time series of daily releases
    def daily_production(self, water_balance):
        """
        param water_balance: pandas DataFrame of the water balance, containing columns 'Release (m3/s)' and
                            'Storage (m3)', and indexed on dates
        return: pandas Series of daily hydropower production in MWh, same index
        """

        n_steps = len(water_balance)

        # Get release time series (capped by max release)
        release = np.minimum(water_balance['Release (m3/s)'].values, np.ones(n_steps) *
                             self.hydropower_plant.max_release)

        # Get hydraulic head time series, assuming linear relationship between depth and lake area
        hydraulic_head = self.hydropower_plant.nominal_head - self.total_lake_depth + \
                         self.get_height(water_balance.loc[:, 'Storage (m3)'])

        # Deduce daily hydropower production time series (in MW)
        hydropower_daily = pd.Series(index=water_balance.index,
                                     data=1000 * 9.81 * self.hydropower_plant.efficiency *
                                     np.multiply(hydraulic_head, release) * 24 / 1E6,
                                     name='Daily hydropower production (MWh)')

        return hydropower_daily


# Define demand class
class Demand:

  # Define attributes specific to each Demand object
  def __init__(self, name, intake_depth=np.inf):
    self.name = name
    # intake depth from full lake level
    self.intake_depth = intake_depth


# Define hydropower plant class
class Hydropower:

    # Define attributes
    def __init__(self, installed_capacity_mw, nominal_head_m, max_release_m3s, firm_power_mw=0):
        self.installed_capacity = installed_capacity_mw
        self.nominal_head = nominal_head_m
        self.max_release = max_release_m3s

        # Optional argument: firm power
        self.firm_power = firm_power_mw

        # Deduce plant efficiency
        self.efficiency = self.installed_capacity*1E6 / (1000*9.81*self.nominal_head*self.max_release)


def define_reservoir(reservoir_name, downstream_demand_names, direct_demand_names):
    """
    Reads reservoir characteristics from file and builds a Reservoir object based on this. Arguments:
        reservoir_name: string that corresponds to the reservoir name on key files
        downstream_demand_names: list of strings for demands situated downstream of the reservoir
        direct_demand_names: list of strings for demands associated to water withdrawals directly from reservoir
    Returns an instance of the Reservoir class.
    """

    # First we download the key data
    key_parameters = pd.read_excel('data/' + reservoir_name + '_data.xlsx', sheet_name='Reservoir characteristics',
                                   dtype={'Key parameter and unit': str, 'Value': float})

    # Define hydropower plant characteristics from file
    nominal_head = key_parameters.iloc[3, key_parameters.columns.get_loc('Value')]
    installed_capacity = key_parameters.iloc[4, key_parameters.columns.get_loc('Value')]
    max_release = key_parameters.iloc[5, key_parameters.columns.get_loc('Value')]
    if key_parameters['Key parameter and unit'].str.contains('Firm').iloc[-1]:  # There's a firm power
        firm_power = key_parameters.iloc[-1, key_parameters.columns.get_loc('Value')]
    else:
        firm_power = 0
    hpp = Hydropower(installed_capacity, nominal_head, max_release, firm_power_mw=firm_power)

    # Extract key reservoir characteristics
    min_storage = key_parameters.iloc[0, key_parameters.columns.get_loc('Value')] * 100 ** 3
    full_lake_volume = key_parameters.iloc[1, key_parameters.columns.get_loc('Value')] * 100 ** 3
    full_lake_area = key_parameters.iloc[2, key_parameters.columns.get_loc('Value')] * 100 ** 2

    # Create a reservoir object with the specified values
    reservoir = Reservoir(reservoir_name, full_lake_area, full_lake_volume, min_storage, hpp)

    # Define and add demands (note this is bespoke)
    for i in range(len(downstream_demand_names)):
        reservoir.add_downstream_demand(Demand(downstream_demand_names[i]))
    for i in range(len(direct_demand_names)):
        reservoir.add_on_site_demand(Demand(direct_demand_names[i], key_parameters.iloc[6 + i, 1]))

    return reservoir


def extract_flows(reservoir):
    """
    This function reads the inflows and demand data from Excel or CSV files to put that into a DataFrame

    Argument:
        reservoir: An instance of the Reservoir class

    Output:
        water_flows: a pandas DataFrame with the inflows and demand data, in this order.
    """

    # Read inflow and demand files
    inflow_data = pd.read_excel('data/' + reservoir.name + '_data.xlsx', sheet_name='Flow data', index_col=0)
    demand_data = pd.read_excel('data/' + reservoir.name + '_data.xlsx', sheet_name='Demands')

    # Initialise water balance with inflows
    water_flows = pd.DataFrame(inflow_data.sum(axis=1) * 0.3048 ** 3, columns=['Total inflows (m3/s)'])

    # Initialise demand time series
    for i in range(len(reservoir.demand_on_site)):
        water_flows[reservoir.demand_on_site[i].name + ' demand (m3/s)'] = np.zeros(len(water_flows))
    for i in range(len(reservoir.demand_downstream)):
        water_flows[reservoir.demand_downstream[i].name + ' demand (m3/s)'] = np.zeros(len(water_flows))

    # Convert monthly demand pattern into daily time series
    months = np.arange(1, 13, 1)
    demand_number = len(reservoir.demand_on_site) + len(reservoir.demand_downstream)
    for month in months:
        # Make a mask to only keep the days that correspond to the current month.
        monthly_mask = water_flows.index.month == month

        # For all days of that month, get the correct data (note this assumes the order of demands is respected)
        water_flows.iloc[monthly_mask, 1:1+demand_number] = demand_data.iloc[month - 1, 1:1+demand_number] * 0.3048 ** 3

    return water_flows
