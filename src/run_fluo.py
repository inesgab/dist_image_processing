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


def calculate_fluorescence(main_folder_name: str, t1: int, t2: int) -> None:
    """
    Parcourt les dossiers dans folder_path, charge les masques et les images de fluorescence,
    et calcule la somme des pixels de fluorescence dans chaque masque.

    Args:
        folder_path (str): Chemin vers le dossier contenant les sous-dossiers numérotés.
        t1 (int): Indice temporel de début.
        t2 (int): Indice temporel de fin.

    Returns:
        dict: Un dictionnaire contenant la somme des pixels de fluorescence pour chaque dossier et chaque temps.
    """
    fluorescence_results = {}
    folder_path = get_results_path(main_folder_name)
    print(f"Folder path: {folder_path}")
    print("Calculating fluorescence...")
    for droplet_number_and_coords in os.listdir(folder_path):
        folder_full_path = os.path.join(folder_path, droplet_number_and_coords)
        droplet_number, coords = droplet_number_and_coords.split("_")

        # Vérifie si le dossier est numéroté
        if os.path.isdir(folder_full_path) and droplet_number.isdigit():
            fluorescence_results[droplet_number] = {}

            for t in range(t1, t2 + 1):
                try:
                    # Chemins des fichiers
                    mask_path = os.path.join(folder_full_path, f"mask/{t}.npy")
                    fluo_path = os.path.join(folder_full_path, f"fluo/{t}.png")

                    # Charger le masque et l'image de fluorescence
                    if os.path.exists(mask_path) and os.path.exists(fluo_path):
                        mask = np.load(mask_path)
                        fluo_image = cv2.imread(fluo_path, cv2.IMREAD_GRAYSCALE)

                        # Vérifier que les dimensions correspondent
                        if mask.shape != fluo_image.shape:
                            raise ValueError(
                                f"Dimensions mismatch for t={t} in {droplet_number}"
                            )

                        # Calculer la fluorescence dans le masque
                        fluorescence = calculate_average_fluorescence(fluo_image, mask)
                        fluorescence_results[droplet_number][t] = fluorescence
                    else:
                        print(f"Missing files for t={t} in {droplet_number}")
                except Exception as e:
                    print(f"Error processing t={t} in {droplet_number}: {e}")

    # Sauvegarder les résultats dans un tableau et un graphique
    print("Saving results...")
    save_fluorescence_results(folder_path, fluorescence_results)


def save_fluorescence_results(folder_path: str, fluorescence_results: dict) -> None:
    """
    Sauvegarde les résultats de fluorescence dans un fichier CSV et génère un graphique.

    Args:
        folder_path (str): Chemin vers le dossier contenant les sous-dossiers numérotés.
        fluorescence_results (dict): Résultats de fluorescence.
    """
    # Convertir les résultats en DataFrame
    data = []
    for folder, times in fluorescence_results.items():
        for t, fluorescence in times.items():
            data.append({"Folder": folder, "Time": t, "Fluorescence": fluorescence})
    df = pd.DataFrame(data)

    # Sauvegarder le tableau dans un fichier CSV
    csv_path = os.path.join(folder_path, "fluorescence_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Fluorescence results saved to {csv_path}")

    data_title = os.path.basename(folder_path)

    # Générer un graphique
    plt.figure(figsize=(10, 6))
    for folder in fluorescence_results:
        times = sorted(fluorescence_results[folder].keys())
        values = [fluorescence_results[folder][t] for t in times]
        plt.plot(times, values, label=f"Folder {folder}")

    plt.xlabel("Time")
    plt.ylabel("Fluorescence")
    plt.title(f"Fluorescence Over Time: {data_title}")
    plt.grid(True)

    # Sauvegarder le graphique
    plot_path = os.path.join(folder_path, "fluorescence_plot.png")
    plt.savefig(plot_path)
    print(f"Fluorescence plot saved to {plot_path}")
    plt.close()


main_folder_name = input(
    "Enter the folder name in 'results' where the masks and images are located: "
)


def run_fluo():
    t1 = int(input("Enter the first timepoint (t1): "))
    t2 = int(input("Enter the last timepoint (t2): "))
    calculate_fluorescence(main_folder_name, t1, t2)

if __name__ == "__main__":
    run_fluo()
