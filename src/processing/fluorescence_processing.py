import numpy as np


def calculate_average_fluorescence(fluo_image, mask):
    """
    Calculates the average fluorescence intensity within a given mask.

    Args:
        fluo_image: NumPy array representing the fluorescence image.
        mask: NumPy array representing the binary mask (0 or 1).

    Returns:
        The average fluorescence intensity within the mask, or NaN if the mask is empty.
    """
    masked_fluorescence = fluo_image[mask > 0]
    if len(masked_fluorescence) == 0:
        return np.nan  # Return NaN if the mask doesn't cover any pixels
    return np.mean(masked_fluorescence)

def calculate_sum_fluorescence(fluo_image, mask):
    """
    Calculates the sum of fluorescence intensity within a given mask.

    Args:
        fluo_image: NumPy array representing the fluorescence image.
        mask: NumPy array representing the binary mask (0 or 1).

    Returns:
        The sum of fluorescence intensity within the mask, or NaN if the mask is empty.
    """
    masked_fluorescence = fluo_image[mask > 0]
    if len(masked_fluorescence) == 0:
        return np.nan  # Return NaN if the mask doesn't cover any pixels
    return np.sum(masked_fluorescence)
