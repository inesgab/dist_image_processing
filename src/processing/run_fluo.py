import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from processing.fluorescence_processing import (
    calculate_average_fluorescence,
    calculate_sum_fluorescence,
)
from utils.file_utils import get_results_path
from skimage.filters import threshold_otsu, threshold_yen

def calculate_fluorescence(main_folder_name: str, t1: int, t2: int) -> None:
    """
    Iterates through folders in folder_path, loads masks and fluorescence images,
    and calculates the sum of fluorescence pixels in each mask.

    Args:
        main_folder_name (str): Name of the main folder containing the subfolders.
        t1 (int): Start time index.
        t2 (int): End time index.

    Returns:
        None
    """
    fluorescence_results = {}
    folder_path = get_results_path(main_folder_name)
    print(f"Folder path: {folder_path}")
    print("Calculating fluorescence...")
    for droplet_number_and_coords in os.listdir(folder_path):
        folder_full_path = os.path.join(folder_path, droplet_number_and_coords)
        droplet_number, coords = droplet_number_and_coords.split("_")

        # Check if the folder is numbered
        if os.path.isdir(folder_full_path) and droplet_number.isdigit():
            fluorescence_results[droplet_number] = {}

            for t in range(t1, t2 + 1):
                try:
                    # File paths
                    mask_path = os.path.join(folder_full_path, f"mask/{t}.npy")
                    fluo_path = os.path.join(folder_full_path, f"fluo/{t}.png")

                    # Load the mask and fluorescence image
                    if os.path.exists(mask_path) and os.path.exists(fluo_path):
                        mask = np.load(mask_path)
                        fluo_image = cv2.imread(fluo_path, cv2.IMREAD_GRAYSCALE)
                        pixels = fluo_image[mask > 0]
                        if len(pixels) > 0:
                            # Apply Otsu's thresholding method
                            thresh = threshold_otsu(pixels)
                            binary = np.where(
                                (fluo_image > thresh) & (mask > 0), 255, 0
                            ).astype(np.uint8)
                            selected = np.sum((binary == 255) & (mask > 0))
                            total = np.sum(mask > 0)
                            if total > 0 and selected / total > 0.9: #if the mask covers most of the droplet -> out
                                binary = np.zeros_like(fluo_image, dtype=np.uint8)
                        else:
                            binary = np.zeros_like(fluo_image, dtype=np.uint8)
                        # display binary on fluo
                        # enhance contrast in image
                        # fluo_image = cv2.normalize(fluo_image, None, 0, 255, cv2.NORM_MINMAX)
                        # overlay = cv2.addWeighted(
                        #     fluo_image, 0.5, binary, 0.5, 0
                        # )
                        # cv2.imshow(f"Overlay, time {t}, droplet {droplet_number}", overlay)
                        # #cv2.imshow(f"Binary, time {t}, droplet {droplet_number}", binary)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()

                        # Check that dimensions match
                        if mask.shape != fluo_image.shape:
                            raise ValueError(
                                f"Dimensions mismatch for t={t} in {droplet_number}"
                            )

                        # Calculate fluorescence in the mask
                        otsu_fluo = calculate_sum_fluorescence(fluo_image, binary)
                        if otsu_fluo is np.nan:
                            otsu_fluo = 0
                        fluorescence_results[droplet_number][t] = {
                            "avg": calculate_average_fluorescence(fluo_image, mask), # average pixel value in full mask
                            "sum": calculate_sum_fluorescence(fluo_image, mask), # sum of pixels in full mask
                            "otsu": otsu_fluo, # sum of pixels in the Otsu mask
                        }
                    else:
                        print(f"Missing files for t={t} in {droplet_number}")
                except Exception as e:
                    print(f"Error processing t={t} in {droplet_number}: {e}")

    # Save results to a table and a global plot
    print("Saving results...")
    save_fluorescence_results(folder_path, fluorescence_results)

def save_fluorescence_results(folder_path: str, fluorescence_results: dict) -> None:
    """
    Saves fluorescence results to a CSV file and generates a plot.

    Args:
        folder_path (str): Path to the folder containing the numbered subfolders.
        fluorescence_results (dict): Fluorescence results.
    """
    data = []
    for folder, times in fluorescence_results.items():
        for t, values in times.items():
            data.append(
                {
                    "Folder": folder,
                    "Time": t,
                    "Fluorescence_Avg": values["avg"],
                    "Fluorescence_Sum": values["sum"],
                    "Fluorescence_Otsu": values["otsu"],
                }
            )
    df = pd.DataFrame(data)

    # Saving the data
    csv_path = os.path.join(folder_path, "fluorescence_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Fluorescence results saved to {csv_path}")

    data_title = os.path.basename(folder_path)

    # Generate plot
    plt.figure(figsize=(10, 6))
    for folder in fluorescence_results:
        times = sorted(fluorescence_results[folder].keys())
        values = [fluorescence_results[folder][t]["avg"] for t in times]
        plt.plot(times, values, label=f"Folder {folder}")

    plt.xlabel("Time")
    plt.ylabel("Fluorescence (Average)")
    plt.title(f"Average Fluorescence Over Time: {data_title}")
    plt.grid(True)
    plot_path = os.path.join(folder_path, "fluorescence_avg.png")
    plt.savefig(plot_path)
    print(f"Fluorescence plot saved to {plot_path}")
    plt.close()

    plt.figure(figsize=(10, 6))
    for folder in fluorescence_results:
        times = sorted(fluorescence_results[folder].keys())
        values = [fluorescence_results[folder][t]["otsu"] for t in times]
        plt.plot(times, values, label=f"Folder {folder}")

    plt.xlabel("Time")
    plt.ylabel("Fluorescence Sum Otsu")
    plt.title(f"Sum of Otsu Fluorescence Over Time: {data_title}")
    plt.grid(True)
    plot_path = os.path.join(folder_path, "fluorescence_otsu.png")
    plt.savefig(plot_path)
    print(f"Otsu Fluorescence plot saved to {plot_path}")
    plt.close()
