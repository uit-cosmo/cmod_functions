import MDSplus as mds
import numpy as np
import xarray as xr
from scipy import interpolate
from typing import Union
import matplotlib.pyplot as plt
from .base_functions import *


@xr.register_dataset_accessor("phantom")
class PhantomAccessor:
    def __init__(self, xarray_obj):

        self._obj = xarray_obj
        self.shot_number = self._obj.shot_number

    def plot(self, time_index=0):
        R_limiter, Z_limiter = get_limiter_coordinates(self.shot_number)
        R_limiter, Z_limiter = calculate_splinted_limiter(R_limiter, Z_limiter)

        rbbbs, zbbbs, _, efit_time = get_separatrix_coordinates(self.shot_number)
        R_LCFS, Z_LCFS = calculate_splinted_LCFS(
            time_step=self._obj.time.values[time_index],
            efit_time=efit_time,
            rbbbs=rbbbs,
            zbbbs=zbbbs,
        )
        CS = plt.contourf(
            self._obj.R / 100,
            self._obj.Z / 100,
            self._obj.frames.isel(time=time_index).values[:, ::-1],
            64,
        )
        plt.colorbar(CS)
        plt.plot(R_LCFS, Z_LCFS, c="tab:orange")
        plt.plot(R_limiter, Z_limiter, c="tab:orange")
        plt.xlim([0.85, 0.93])
        plt.ylim([-0.07, 0.01])
        plt.xlabel("R_maj [m]")
        plt.ylabel("Z [m]")


def get_integration_time(shot_number: int):
    """
    Extract integration time .
    Args:
        shot_number: Shot number of interest.
    Returns:
        integration time
    """
    c = mds.Connection("alcdata")
    c.openTree("spectroscopy", shot_number)

    return c.get("GPI.PHANTOM.SETTINGS:EXPOS").data()


def get_sensitivity_callibration(shot_number: int, filter: str):
    """
    Extract sensitivity callibration data.
    Args:
        shot_number: Shot number of interest.
        filter: H_alpha or He-587 filter
    Returns:
        inverse sensitivity for every pixel
    """
    assert filter in {"H_alpha", "He-587"}

    c = mds.Connection("alcdata")
    c.openTree("spectroscopy", shot_number)

    if filter == "H_alpha":
        return c.get("GPI.PHANTOM:SENS_ARR_DA").data()
    if filter == "He-587":
        return c.get("GPI.PHANTOM:SENS_ARR_HEI").data()


def get_major_radius_phantom_coordinates(shot_number: int):
    """
    Extracts the radial and poloidal positions of the pixels in major radius coordinates for shots.
    Args:
        shot_number: Shot number(s) of interest.
    Returns:
        R: Major radius coordinates in centimetres.
        Z: Height array (above the machine midplane) in centimetres.
    """

    c = mds.Connection("alcdata")
    c.openTree("spectroscopy", shot_number)

    R_array_path = "GPI.PHANTOM.IMAGE_POS.R_ARR"
    Z_array_path = "GPI.PHANTOM.IMAGE_POS.Z_ARR"

    R = c.get(R_array_path).data()
    Z = c.get(Z_array_path).data()

    return R, Z


def get_phantom_frames(shot_number: int):
    """
    Extracts the frames and time array for a shot.

    Args:
        shot_number: Shot number(s) of interest.

    Returns:
        times: Time array.
        frames: Raw frames extracted for all pixels.
    """

    c = mds.Connection("alcdata")
    c.openTree("spectroscopy", shot_number)
    frames_path = "GPI.PHANTOM.FRAMES"
    frame_rate_path = "GPI.PHANTOM.SETTINGS.FRAME_RATE"
    t_hists_path = "GPI.PHANTOM.SETTINGS.TRIG_TIME"

    frames = c.get(frames_path).data()

    frame_rate = c.get(frame_rate_path).data()
    start_time = c.get(t_hists_path).data()

    number_of_frames = frames.shape[0]
    stop_time = start_time + number_of_frames / frame_rate

    times = np.linspace(start=start_time, stop=stop_time, num=number_of_frames)

    return times, frames


def _normalize_frames(
    frames: np.ndarray,
    shot_number: int,
    filter: Union[None, str] = None,
):
    """
    Sensitivity calibration using D_alpha or He-587 filter.

    Further testing recommended since this function hasn't been used so far.
    """
    try:
        inverse_sensitivity = get_sensitivity_callibration(
            shot_number=shot_number, filter=filter
        )
        integration_time = get_integration_time(shot_number)
        return (frames[None, :, :] * inverse_sensitivity)[0] / integration_time
    except Exception:
        print(f"No sensitivity callibration data available")
        return frames


def generate_phantom_dataset(
    shot_number: int,
    time_start: float = -np.inf,
    time_end: float = np.inf,
    filter: Union[None, str] = None,
):
    """
    Generates an xarray dataset containing raw outboard midplane Phantom camera data for a shot

    Args:
        shot_number: Shot number(s) of interest.
        time_start: The beginning of the time window in seconds. Set to first frame by default.
        time_end: The end of the time window in seconds. Set to last frame by default.
        filter: if not None, using callibration filter

    Returns:
        dataset: An xarray dataset containing raw phantom camera data for all pixels:
            x, y = [0,0] refers to the lowest R and Z values.

            time: Time array in seconds.
            frames: Raw frames extracted for all pixels.
            R: Major radius coordinates in centimetres if available.
            Z: Height array (above the machine midplane) in centimetres if avaliable.
    """
    assert filter in {None, "H_alpha", "He-587"}

    time, frames = get_phantom_frames(shot_number)
    frames = np.flip(frames, axis=2)

    time_interval = (time > time_start) & (time < time_end)
    frames = frames[time_interval, :, :]
    time = time[time_interval]

    if filter is not None:
        frames = _normalize_frames(frames, shot_number, filter)

    rlimit, zlimit = get_limiter_coordinates(shot_number)
    rlcfs, zlcfs, _, efit_time = get_separatrix_coordinates(shot_number)

    R, Z = get_major_radius_phantom_coordinates(shot_number)
    R = np.flip(R, axis=1)
    Z = np.flip(Z, axis=1)

    # Not all shots provide R and Z data (I removed try/catch as it was bad for debugging)
    return xr.Dataset(
        {"frames": (["time", "y", "x"], frames),
         "rlimit": rlimit*100,
         "zlimit": zlimit*100,
         "rlcfs": (["xlcfs", "efit_time"], rlcfs*100),
         "zlcfs": (["ylcfs", "efit_time"], zlcfs*100)},
        coords={
            "R": (["y", "x"], R),
            "Z": (["y", "x"], Z),
            "time": (["time"], time),
            "efit_time": (["efit_time"], efit_time)
        },
        attrs=dict(shot_number=shot_number),
    )
