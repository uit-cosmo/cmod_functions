import MDSplus as mds
import numpy as np
import xarray as xr

"""
Abbreviations:
    MLP: mirror-Langmuir probe
    ASP: A-port scanning probe (scans in the horizonal direction)
    ISP: Ion-sensitive probe
    
    Functions saying: asp_mlp or asp_isp means they are NOT a convential Langmuir probe
    Note: The mirror-Langmuir probe was NOT operational before 2012.
    
    For the mirror-Langmuir probe:
    p0 is the NE pin
    p1 is the SW pin
    p2 is the SE pin
    p3 is the NW pin
    
    This may differ for the convention Langmuir probe. It is advised to read the logbook
    for further details as this may differ between rundays.
"""

# Node names for ASP MLP data. Use this convention.
variables_dictionary_asp_mlp = {
    "ne": "DENSITY_FIT",
    "Is": "ISAT_FIT",
    "Js": "JSAT_FIT",
    "Vp": "PHI_FIT",
    "Te": "TE_FIT",
    "Vf": "VF_FIT",
}

# Node names for ASP data. Use this convention.
# Fast data is highly sampled data. Slow data is downsampled data.
variables_dictionary_asp = {
    "ne_fast": "NE_FAST",
    "Te_fast": "TE_FAST",
    "Is_fast": "I_FAST",
    "Vf_fast": "V_FAST",
    "ne_slow": "NE_SLOW",
    "Te_slow": "TE_SLOW",
    "Is_slow": "I_SLOW",
    "Vf_slow": "V_SLOW",
}

# Node names for ASP ISP data. Use this convention.
variables_dictionary_asp_isp = {
    "Is_fast": "I_FAST",
    "Vf_fast": "V_FAST",
    "Is_slow": "I_SLOW",
    "Vf_slow": "V_SLOW",
}


def get_plunge_depth(shot_number: int):
    """
    Extracts the plunge distance and corresponding time of the probe.

    Args:
        shot_number: Shot number(s) of interest.

    Returns:
        plunge_time: Corresponding time data in seconds.
        plunge: Plunge depth of probe in metres and assumed purely horizontal.
    """

    c = mds.Connection("alcdata")
    c.openTree("edge", shot_number)

    dataname_plunge = "\EDGE::TOP.PROBES.ASP.PLUNGE"

    plunge = c.get(dataname_plunge).data()
    plunge_time = c.get(f"dim_of({dataname_plunge})").data()

    return plunge_time, plunge


def get_probe_origin(shot_number: int):
    """
    Extracts the probe origin.

    Args:
        shot_number: Shot number(s) of interest.

    Returns:
        origin: Probe origin giving (R, Z, R*Phi) of the probe, *EXPLAIN R, Z, Phi*
    """

    dataname_origin = "\EDGE::TOP.PROBES.ASP.G_1.ORIGIN"
    c = mds.Connection("alcdata")
    c.openTree("edge", shot_number)

    origin = c.get(dataname_origin).data()

    return origin


def get_asp_rho(shot: int, probe_pin_number: int):
    """
    Extracts the rho, the distance relative to the last-closed flux surface, of a ASP probe tip.
    Shots before 2012 have ASP data (i.e. conventional Langmuir probe).

    Args:
        shot_number: Shot number(s) of interest.
        probe_pin_number: Particluar probe tip usually from 0 to 3.

    Returns:
        rho_time: Time data for rho
        rho: The probe position relative to the separatrix
    """

    c = mds.Connection("alcdata")
    c.openTree("edge", shot)

    dataname_rho = f"\EDGE::TOP.PROBES.ASP.G_1.P{probe_pin_number}:RHO"

    rho = c.get(dataname_rho)
    rho_time = c.get(f"dim_of({dataname_rho})").data()

    return rho_time, rho


def get_raw_asp_data(
    shot_number: int,
    probe_pin_number: int,
    variable_name: str,
    time_start: float = -np.inf,
    time_end: float = np.inf,
):
    """
    Extracts raw ASP data. Shots before 2012 have ASP data.
    This is just a conventional Langmuir probe.

    Args:
        shot_number: Shot number(s) of interest.
        probe_pin_number: Particluar probe tip usually from 0 to 3.
        variable_name: The variable of interests
            variables_dictionary_asp = {
            "Is_fast": "I_FAST",
            "Vf_fast": "V_FAST",
            "Is_slow": "I_SLOW",
            "Vf_slow": "V_SLOW",}
        time_start: Start time of interest. Set to first frame by default.
        time_end: End time of interest. Set to last frame by default.

    Returns:
        asp_time: Time data for ASP.
        asp_data: Raw ASP data of a particular variable.
    """

    c = mds.Connection("alcdata")
    c.openTree("edge", shot_number)

    dataname = f"\EDGE::TOP.PROBES.ASP.G_1.P{probe_pin_number}:{variables_dictionary_asp[variable_name]}"

    asp_data = c.get(dataname).data()

    asp_time = c.get(f"dim_of({dataname})").data()

    time_interval = (asp_time > time_start) & (asp_time < time_end)
    return asp_time[time_interval], asp_data[time_interval]


def get_asp_isp_rho(shot_number: int, probe_pin_number: int):
    """
    Extracts the rho, the distance relative to the last-closed flux surface, of ISP probe tip.
    Check the logbook whether the shot you're after used the ISP.

    Args:
        shot_number: Shot number(s) of interest.
        probe_pin_number: Particluar probe tip usually from 0 to 3.

    Returns:
        rho_time: Time data for rho
        rho: The probe position relative to the separatrix
    """

    c = mds.Connection("alcdata")
    c.openTree("edge", shot_number)

    dataname_rho = f"\EDGE::TOP.PROBES.ASP.ISP.P{probe_pin_number}:RHO"

    rho = c.get(dataname_rho)
    rho_time = c.get(f"dim_of({dataname_rho})").data()

    return rho_time, rho


def get_raw_asp_isp_data(
    shot_number: int,
    probe_pin_number: int,
    variable_name: str,
    time_start: float = -np.inf,
    time_end: float = np.inf,
):
    """
    Extracts raw ISP data. Check the logbook whether the shot you're after used the ISP.

    Args:
        shot_number: Shot number(s) of interest.
        probe_pin_number: Particluar probe tip usually from 0 to 3.
        variable_name: The variable of interests
            variables_dictionary_asp_isp = {
            "Is_fast": "I_FAST",
            "Vf_fast": "V_FAST",
            "Is_slow": "I_SLOW",
            "Vf_slow": "V_SLOW",}
        time_start: Start time of interest. Set to first frame by default.
        time_end: End time of interest. Set to last frame by default.

    Returns:
        asp_isp_time: Time data for ISP.
        asp__ispdata: Raw ISP data of a particular variable.
    """

    c = mds.Connection("alcdata")
    c.openTree("edge", shot_number)

    dataname = f"\EDGE::TOP.PROBES.ASP.ISP.P{probe_pin_number}:{variables_dictionary_asp_isp[variable_name]}"

    asp_isp_data = c.get(dataname).data()

    asp_isp_time = c.get(f"dim_of({dataname})").data()

    time_interval = (asp_isp_time > time_start) & (asp_isp_time < time_end)
    return asp_isp_time[time_interval], asp_isp_data[time_interval]


def get_asp_mlp_rho(shot_number: int, probe_pin_number: int):
    """
    Extracts the rho, the distance relative to the last-closed flux surface, of an MLP probe tip.
    Shots from 2012 onwards have MLP data.

    Args:
        shot_number: Shot number(s) of interest.
        probe_pin_number: Particluar probe tip usually from 0 to 3.

    Returns:
        rho_time: Time data for rho
        rho: The probe position relative to the separatrix
    """

    c = mds.Connection("alcdata")
    c.openTree("edge", shot_number)

    dataname_rho = f"\EDGE::TOP.PROBES.ASP.MLP.P{probe_pin_number}:RHO"

    rho = c.get(dataname_rho)
    rho_time = c.get(f"dim_of({dataname_rho})").data()

    return rho_time, rho


def get_raw_asp_mlp_data(
    shot_number: int,
    probe_pin_number: int,
    variable_name: str,
    time_start: float = -np.inf,
    time_end: float = np.inf,
):
    """
    Extracts raw mirror-Langmuir probe (MLP) data. Shots from 2012 onwards have MLP data. Please interrogate the logbooks.

    Args:
        shot_number: Shot number(s) of interest.
        probe_pin_number: Particluar probe tip usually from 0 to 3.
        variable_name: The variable of interests
            variables_dictionary_asp_mlp = {
            "ne": "DENSITY_FIT",
            "Is": "ISAT_FIT",
            "Js": "JSAT_FIT",
            "Vp": "PHI_FIT",
            "Te": "TE_FIT",
            "Vf": "VF_FIT"}
        time_start: Start time of interest. Set to first frame by default.
        time_end: End time of interest. Set to last frame by default.

    Returns:
        asp_mlp_time: Time data for MLP.
        asp_mlp_data: Raw MLP data of a particular variable.
    """

    c = mds.Connection("alcdata")
    c.openTree("edge", shot_number)

    dataname = f"\EDGE::TOP.PROBES.ASP.MLP.P{probe_pin_number}:{variables_dictionary_asp_mlp[variable_name]}"

    asp_mlp_data = c.get(dataname).data()

    asp_mlp_time = c.get(f"dim_of({dataname})").data()

    time_interval = (asp_mlp_time > time_start) & (asp_mlp_time < time_end)
    return asp_mlp_time[time_interval], asp_mlp_data[time_interval]


def generate_average_mlp_data(shot_number: int, variable_name: str):
    """
    Generates average raw MLP data.

    Args:
        shot_number: Shot number(s) of interest.
        probe_pin_number: Particluar probe tip usually from 0 to 3.
        variable_name: The variable of interests
            variables_dictionary_asp_mlp = {
            "ne": "DENSITY_FIT",
            "Is": "ISAT_FIT",
            "Js": "JSAT_FIT",
            "Vp": "PHI_FIT",
            "Te": "TE_FIT",
            "Vf": "VF_FIT"}

    Returns:
        time_common: Common time data of all four probe bins collecting data.
        time_series_average: Average raw MLP data of a particular variable from all four pin.
    """

    from functools import reduce

    time_series_list = []
    time_list = []

    for probe_pin_number in [0, 1, 2, 3]:
        asp_mlp_time, asp_mlp_data = get_raw_asp_mlp_data(
            variable_name, probe_pin_number, shot_number
        )

        time_series_list.append(asp_mlp_data)
        time_list.append(asp_mlp_time)

    time_common = reduce(np.intersect1d, time_list)

    time_series_average = 0.25 * (
        time_series_list[0]
        + time_series_list[1]
        + time_series_list[2]
        + time_series_list[3]
    )

    return (
        time_common,
        time_series_average,
    )


def get_asp_dataset(shot_number: int):
    """
    Comprehensive data retrieval function for ASP MLP (Mirror Langmuir Probe) datasets.

    Args:
        shot_number (int): Shot number for data retrieval

    Returns:
        xr.Dataset: Consolidated dataset with probe measurements and metadata
    """
    # Probe variables to extract
    variables = list(variables_dictionary_asp_mlp.keys())
    pins = [0, 1, 2, 3]

    # Prepare storage for data
    data_vars = {}
    coords = {}

    # Retrieve probe origin
    try:
        probe_origin = get_probe_origin(shot_number)
    except Exception as e:
        print(f"Warning: Could not retrieve probe origin: {e}")
        probe_origin = None

    # Retrieve plunge depth
    try:
        plunge_time, plunge_depth = get_plunge_depth(shot_number)
    except Exception as e:
        print(f"Warning: Could not retrieve plunge depth: {e}")
        plunge_time = None
        plunge_depth = None

    # Collect MLP data for all pins and variables
    for variable in variables:
        # Create a dictionary to store times and data for each pin
        pin_data_dict = {}
        pin_rho_dict = {}

        for pin in pins:
            try:
                # Retrieve rho for the pin
                rho_time, rho = get_asp_mlp_rho(shot_number, pin)

                # Retrieve raw data for the variable and pin
                time, data = get_raw_asp_mlp_data(shot_number, pin, variable)

                # Store data with its original time base
                pin_data_dict[f"pin_{pin}"] = (f"time_pin_{pin}", data)
                pin_data_dict[f"time_pin_{pin}"] = (f"time_pin_{pin}", time)

                # Store rho information
                pin_rho_dict[f"rho_pin_{pin}"] = rho
                pin_rho_dict[f"rho_time_pin_{pin}"] = rho_time

            except Exception as e:
                print(
                    f"Warning: Could not retrieve data for pin {pin}, variable {variable}: {e}"
                )
                continue

        # Add data variables and coordinates to the dataset
        for key, value in pin_data_dict.items():
            data_vars[key] = value

        # Add rho information as coordinates
        for key, value in pin_rho_dict.items():
            coords[key] = value

    # Create xarray Dataset
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "shot_number": shot_number,
            "probe_origin": probe_origin,
            "plunge_time": plunge_time,
            "plunge_depth": plunge_depth,
            "probe_type": "Mirror Langmuir Probe (MLP)",
        },
    )

    return dataset
