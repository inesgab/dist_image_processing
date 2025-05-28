import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.file_utils import get_results_path

fluorescence_file = "/fluorescence_results.csv"

def log_plot(folder_name):
    """
    Generate log plots for average and Otsu fluorescence over time.
    """
    folder_path = get_results_path(folder_name)
    doc_path = folder_path + fluorescence_file
    df = pd.read_csv(doc_path)

    epsilon = 1e-2  # To avoid log(0)

    plt.figure(figsize=(10, 6))
    min_avg_fluo = min(df["Fluorescence_Avg"])
    for folder, group in df.groupby("Folder"):
        y = group["Fluorescence_Avg"] - min_avg_fluo + epsilon
        plt.plot(group["Time"], np.log10(y), label=folder)

    plt.xlabel("Time")
    plt.ylabel(f"log10(Fluorescence(avg) - min + ε={epsilon})")
    plt.title(f"Average Fluorescence (log) over Time:{folder_name}")
    plt.grid(True)
    plot_path = os.path.join(folder_path, "fluorescence_avglogplot.png")
    plt.savefig(plot_path)
    print(f"Average Fluorescence log plot saved to {plot_path}")

    plt.figure(figsize=(10, 6))

    epsilon2 = 1e3
    for folder, group in df.groupby("Folder"):
        y = group["Fluorescence_Otsu"] + epsilon2
        plt.plot(group["Time"], np.log10(y), label=folder)

    plt.xlabel("Time")
    plt.ylabel(f"log10(Fluorescence (Otsu) + ε= {epsilon2})")
    plt.title(f"Total Otsu Fluorescence (log) over Time:{folder_name}")
    plt.grid(True)
    plot_path = os.path.join(folder_path, "fluorescence_otsulogplot.png")
    plt.savefig(plot_path)
    print(f"Total Otsu Fluorescence log plot saved to {plot_path}")

def indiv_plot(folder_name):
    """
    Generate individual log plots for average and Otsu fluorescence over time for each droplet.
    """
    folder_path = get_results_path(folder_name)
    doc_path = folder_path + fluorescence_file

    df = pd.read_csv(doc_path)
    # Create a unique output folder for all plots
    indiv_out = os.path.join(folder_path, "fluorescences_indiv")
    os.makedirs(indiv_out, exist_ok=True)

    min_avg_fluo = min(df["Fluorescence_Avg"])
    epsilon = 1e-2  # To avoid log(0)

    for folder, group in df.groupby("Folder"):
        # for average
        plt.figure(figsize=(8, 5))
        plt.plot(
            group["Time"], np.log10(group["Fluorescence_Avg"] - min_avg_fluo + epsilon)
        )
        plt.xlabel("Time")
        plt.ylabel(f"Average Fluorescence: log10(Fluorescence_Avg - min(Fluorescence_Avg) + ε={epsilon})")
        plt.title(f"Log Average Fluorescence over Time - Droplet {folder}")
        plt.grid(True)
        plot_path = os.path.join(indiv_out, f"average_fluorescence_plot_{folder}.png")
        plt.savefig(plot_path)
        plt.close()

        # for otsu

        plt.figure(figsize=(8, 5))
        plt.plot(group["Time"], np.log10(group["Fluorescence_Otsu"] + epsilon))
        plt.xlabel("Time")
        plt.ylabel(f"Total Fluorescence: log10(Fluorescence_Otsu + ε= {epsilon})")
        plt.title(f"Log Total Fluorescence over Time - Droplet {folder}")
        plt.grid(True)
        plot_path = os.path.join(indiv_out, f"total_fluorescence_plot_{folder}.png")
        plt.savefig(plot_path)
        plt.close()

        if int(folder) % 20 == 0:
            print(f"Saved plots for folder {folder} to {plot_path}")

def get_plots(folder_name):
    """
    Generate all fluorescence plots for the given folder.
    """
    log_plot(folder_name) # log global plot
    indiv_plot(folder_name) # log individual plots
