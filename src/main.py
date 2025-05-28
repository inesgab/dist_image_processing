#In[]
import torch
import torchvision
import os
from PIL import Image

from processing.generate_masks import (
    save_image_droplets_and_masks,
)
from processing.run_fluo import (
    calculate_fluorescence,
)
from processing.plots import get_plots

# SAM2 model
from sam2.build_sam import (
    build_sam2_video_predictor,
)
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import (
    SAM2AutomaticMaskGenerator,
)


def main():
    """
    Main function to execute the image processing pipeline.
    This function sets up the environment, loads the SAM2 model, and processes images based on user inputs.
    """
    # Environment setup
    print(f"{'-' * 10} Environment setup {'-' * 10}")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
        "1"  # Enable MPS fallback for PyTorch (for Apple Silicon users)
    )
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())

    # Use GPU if available, else CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load the Segment Anything 2 (SAM2) model
    sam2_checkpoint = "sam2.1_hiera_large.pt"  # Path to the model checkpoint
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"  # Path to the model configuration

    sam2 = build_sam2(
        model_cfg,
        sam2_checkpoint,
        device=device,
        apply_postprocessing=False,  # Build the SAM2 model
    )
    mask_generator = SAM2AutomaticMaskGenerator(sam2)  # SAM2 mask generator

    predictor = build_sam2_video_predictor(
        model_cfg, sam2_checkpoint, device=device
    )  # SAM2 video predictor

    # User entries
    print(f"{'-' * 10} User entries {'-' * 10}")
    folder_name = input(
        "Enter the folder name in 'data' where the images are located: "  # Ask for the folder name containing the images
    )

    t1 = int(input("Enter the first timepoint (t1): "))  # Ask for the first timepoint
    t2 = int(input("Enter the last timepoint (t2): "))  # Ask for the last timepoint
    tracking_option = (
        input("Was the tracking of droplets already done? (yes/no):")
        .strip()
        .lower()  # Ask if the tracking of droplets has already been done
    )
    tracked = tracking_option == "yes"

    if tracked:
        fluo_option = (
            input(
                "Was the fluorescence of droplets already calculated? (yes/no):"
            )  # Ask if the fluorescence of droplets has already been calculated
            .strip()
            .lower()
        )
        fluo = fluo_option == "yes"
    else:
        fluo = False
    print(f"Folder name: {folder_name}, t1: {t1}, t2: {t2}, tracked: {tracked}")
    if not tracked:
        roi_width = int(input("Enter the ROI width: "))
        roi_height = int(input("Enter the ROI height: "))

        roi_size = (roi_width, roi_height)

        # Ask the user whether to sort the validated masks (automatic + manual)
        auto_sort_option = (
            input("Do you want to automatically sort the generated masks? (yes/no): ")
            .strip()
            .lower()
        )
        asort = auto_sort_option == "yes"
        man_sort_option = (
            input(
                "Do you want to manually sort the validated masks? If on distant computer -> no. (yes/no): "
            )
            .strip()
            .lower()
        )
        msort = man_sort_option == "yes"

        # Tracking droplets and saving processed data
        save_image_droplets_and_masks(
            folder_name,
            t1,
            t2,
            roi_size,
            mask_generator,
            
            predictor,
            circularity_threshold=0.8,  # Mask circularity threshold
            margin=2,  # Maximum number of pixels between mask and borders of the ROIs: sorting cropped masks out
            msort=msort,
            asort=asort,
            overlap=250,  # Overlap between ROIs
        )

    # Calculating fluorescence for each droplet
    if not fluo:
        calculate_fluorescence(folder_name, t1, t2)

    get_plots(folder_name)  # Generate plots


if __name__ == "__main__":
    main()

# %%
