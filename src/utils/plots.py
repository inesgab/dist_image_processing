import matplotlib.pyplot as plt
import numpy as np


def plot_fluorescences(fluorescences):
    # Tracer la fluorescence en fonction du temps
    plt.figure(figsize=(10, 6))
    for fluorescence_data in fluorescences:
        for mask_fluorescence in fluorescence_data:
            timepoints, fluorescence_values = zip(*fluorescence_data[mask_fluorescence])
            plt.plot(timepoints, fluorescence_values)

    plt.xlabel("Timepoint")
    plt.ylabel("Average Fluorescence")
    plt.title("Fluorescence moyenne par masque en fonction du temps")
    plt.show()


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
