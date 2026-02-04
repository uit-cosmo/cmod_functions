import MDSplus as mds
import numpy as np
from scipy import interpolate


def get_limiter_coordinates(shot_number: int):
    """
    Extracts the radial and poloidal positions of the limiter shadow in major radius coordinates for shots.
    Args:
        shot_number: Shot number of interest.
    Returns:
        R_limiter: Major radius coordinates in centimetres.
        Z_limiter: Height array (above the machine midplane) in centimetres.
    """
    c = mds.Connection("alcdata")
    c.openTree("mhd", shot_number)
    R_limiter = c.get("\MHD::TOP.analysis.limiters.gh_limiter:R")
    Z_limiter = c.get("\MHD::TOP.analysis.limiters.gh_limiter:Z")

    return R_limiter, Z_limiter


def get_separatrix_coordinates(shot_number: int):
    """
    Extracts the radial and poloidal positions of the last closed flux surface (LCFS) in major radius coordinates for shots.
    Args:
        shot_number: Shot number(s) of interest.
    Returns:
        R_LCFS: Major radius coordinates in centimetres.
        Z_LCFS: Height array (above the machine midplane) in centimetres.
        time_LCFS: time array

    """
    c = mds.Connection("alcdata")
    c.openTree("analysis", shot_number)
    rbbbs = c.get("\efit_geqdsk:rbbbs")
    zbbbs = c.get("\efit_geqdsk:zbbbs")
    nbbbs = c.get("\efit_geqdsk:nbbbs")
    efit_time = c.get("dim_of(\efit_geqdsk:rbbbs)")

    return rbbbs, zbbbs, nbbbs, efit_time


def calculate_splinted_LCFS(
    time_step: float,
    efit_time: np.ndarray,
    rbbbs: np.ndarray,
    zbbbs: np.ndarray,
):
    rbbbs = np.array(rbbbs)
    zbbbs = np.array(zbbbs)

    time_difference = np.absolute(time_step - efit_time)
    time_index = time_difference.argmin()

    closest_rbbbs = rbbbs[:, time_index]
    closest_zbbbs = zbbbs[:, time_index]

    f = interpolate.interp1d(
        closest_zbbbs[closest_rbbbs >= 0.86],
        closest_rbbbs[closest_rbbbs >= 0.86],
        kind="cubic",
    )
    z_fine = np.linspace(-0.08, 0.01, 100)
    r_fine = f(z_fine)

    return r_fine, z_fine


def calculate_splinted_limiter(R_limiter: np.ndarray, Z_limiter: np.ndarray):

    f = interpolate.interp1d(
        Z_limiter,
        R_limiter,
        kind="cubic",
    )
    z_fine = np.linspace(-0.08, 0.01, 100)
    r_fine = f(z_fine)

    return r_fine, z_fine
