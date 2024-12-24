import os
import numpy as np
from skimage import io
import torch
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)
import argparse


def calculate_metrics(folder_path):
    # Initialize LPIPS, SSIM, and PSNR metrics
    lpips_metric = LearnedPerceptualImagePatchSimilarity()
    ssim_metric = StructuralSimilarityIndexMeasure(
        gaussian_kernel=True, sigma=1.5, kernel_size=11, reduction="elementwise_mean"
    )
    psnr_metric = PeakSignalNoiseRatio()

    psnr_a, ssim_a, lpips_a = [], [], []
    psnr_b, ssim_b, lpips_b = [], [], []

    for i in range(11):  # Assuming the range of X is from 0 to 10
        try:
            orig_img_path = os.path.join(folder_path, f"orig_{i}.png")
            method_a_img_path = os.path.join(folder_path, f"y0_{i}.png")
            method_b_img_path = os.path.join(folder_path, f"{i}_-1.png")

            orig_img = io.imread(orig_img_path)
            method_a_img = io.imread(method_a_img_path)
            method_b_img = io.imread(method_b_img_path)

        except:
            continue

        # If one of them is not found, continue
        if orig_img is None or method_a_img is None or method_b_img is None:
            continue

        # Convert images to tensors for PSNR, SSIM, and LPIPS
        orig_img_tensor = (
            torch.tensor(orig_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )
        method_a_img_tensor = (
            torch.tensor(method_a_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )
        method_b_img_tensor = (
            torch.tensor(method_b_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )

        # Calculate PSNR
        psnr_a.append(psnr_metric(method_a_img_tensor, orig_img_tensor).item())
        psnr_b.append(psnr_metric(method_b_img_tensor, orig_img_tensor).item())

        # Calculate SSIM
        ssim_a.append(ssim_metric(method_a_img_tensor, orig_img_tensor).item())
        ssim_b.append(ssim_metric(method_b_img_tensor, orig_img_tensor).item())

        # Calculate LPIPS
        lpips_a.append(lpips_metric(orig_img_tensor, method_a_img_tensor).item())
        lpips_b.append(lpips_metric(orig_img_tensor, method_b_img_tensor).item())

    # Calculate average metrics
    # print(psnr_a, psnr_b)

    avg_psnr_a = np.mean(psnr_a)
    avg_ssim_a = np.mean(ssim_a)
    avg_lpips_a = np.mean(lpips_a)

    avg_psnr_b = np.mean(psnr_b)
    avg_ssim_b = np.mean(ssim_b)
    avg_lpips_b = np.mean(lpips_b)

    str_print = f"HIO - PSNR: {avg_psnr_a}, SSIM: {avg_ssim_a}, LPIPS: {avg_lpips_a}\nDDRM-PR - PSNR: {avg_psnr_b}, SSIM: {avg_ssim_b}, LPIPS: {avg_lpips_b}"
    print(str_print)

    # Write to file
    with open(os.path.join(folder_path, "metrics.txt"), "w") as f:
        f.write(str_print)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate average PSNR, SSIM, and LPIPS for two methods."
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Path to the folder containing the images.",
    )

    args = parser.parse_args()
    calculate_metrics(args.folder_path)
