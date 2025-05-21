import torch
import torchvision
import os
from PIL import Image

from processing.generate_masks import save_image_droplets_and_masks

from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def main():
    print(f"{'-'*10} Environment setup {'-'*10}")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    # use gpu if available, else cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load the Segment Anything 2 model
    sam2_checkpoint = "sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    print(f"{'-'*10} User entries {'-'*10}")
    folder_name = input(
        "Enter the folder name in 'data' where the images are located: "
    )

    t1 = int(input("Enter the first timepoint (t1): "))
    t2 = int(input("Enter the last timepoint (t2): "))
    roi_width = int(input("Enter the ROI width: "))
    roi_height = int(input("Enter the ROI height: "))

    roi_size = (roi_width, roi_height)

    # Ask the user whether to sort the validated masks
    auto_sort_option = (
        input("Do you want to automatically sort the generated masks? (yes/no): ").strip().lower()
    )
    asort = auto_sort_option == "yes"
    man_sort_option = (
        input("Do you want to manually sort the validated masks? (yes/no): ").strip().lower()
    )
    msort = man_sort_option == "yes"

    save_image_droplets_and_masks(
        folder_name,
        t1,
        t2,
        roi_size,
        mask_generator,
        predictor,
        circularity_threshold=0.8,
        margin=2,
        msort=msort,
        asort=asort,
        overlap=250,
    )


if __name__ == "__main__":
    main()
