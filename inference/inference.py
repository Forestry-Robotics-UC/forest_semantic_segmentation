"""Run OneFormer semantic segmentation inference on a folder of images.

The script expects a YAML config with model, input, and output paths.
It supports configurable inference resolution and autocast precision.
"""

import yaml
import argparse
import sys
from PIL import Image
import numpy as np
import torch
import os
import time
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import transformers.utils.import_utils as tf_import_utils

tf_import_utils._natten_available = True

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation


def parse_args():
    """Parse command-line arguments for the inference runner."""
    default_config = os.environ.get(
        "INFERENCE_CONFIG",
        os.path.join(os.path.dirname(__file__), "inference.yaml"),
    )
    parser = argparse.ArgumentParser(description="Run OneFormer forest segmentation inference.")
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to inference.yaml (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    """Load configuration and run semantic segmentation inference."""
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as file:
        config_yaml = yaml.safe_load(file)

    model_dir = config_yaml["model_dir"]
    image_dir = config_yaml["image_dir"]
    output_dir = config_yaml["output_dir"]
    input_height = int(config_yaml.get("input_height", 640))
    input_width = int(config_yaml.get("input_width", 1280))
    keep_input_resolution = bool(config_yaml.get("keep_input_resolution", False))
    autocast_precision = str(config_yaml.get("autocast_precision", "bfloat16")).lower()

    model = OneFormerForUniversalSegmentation.from_pretrained(model_dir)
    processor = OneFormerProcessor.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    config = model.config

    if autocast_precision in {"fp16", "float16"}:
        autocast_dtype = torch.float16
        autocast_enabled = device.type == "cuda"
    elif autocast_precision in {"bf16", "bfloat16"}:
        autocast_dtype = torch.bfloat16
        autocast_enabled = device.type == "cuda"
    elif autocast_precision in {"fp32", "float32", "off", "none"}:
        autocast_dtype = torch.float32
        autocast_enabled = False
    else:
        raise ValueError(
            "Unsupported autocast_precision value in config. "
            "Use one of: fp16, bfloat16, fp32/off."
        )

    print("")
    print(f"Total number of model classes: {config.num_labels}")
    print(
        "Inference preprocessing: "
        f"{'original image size' if keep_input_resolution else f'{input_width}x{input_height}'}"
    )
    print(f"Autocast precision: {autocast_precision}")

    custom_palette = np.array([
        [0, 0, 0], [120, 120, 70], [255, 170, 146], [61, 230, 250], [204, 255, 4], [4, 250, 7], [12, 189, 102],
        [255, 41, 10], [51, 0, 255], [150, 5, 61], [255, 0, 122], [0, 140, 0], [100, 65, 0]
    ])

    num_images = 0
    start_time = time.time()

    image_filenames = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))]
    )
    for filename in tqdm(image_filenames, desc="Inference"):
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path).convert("RGB")

        print("")
        print(f"Image mode: {image.mode}")

        # Keep default preprocessing aligned with training unless explicitly overridden.
        if keep_input_resolution:
            processor.image_processor.size = {"height": image.height, "width": image.width}
        else:
            processor.image_processor.size = {"height": input_height, "width": input_width}

        inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                outputs = model(**inputs)

        segmentation_mask = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        segmentation_mask_np = segmentation_mask.cpu().numpy()
        unique_classes = np.unique(segmentation_mask_np)
        print(f"Identified classes: {unique_classes}")

        if segmentation_mask_np.max(initial=0) >= len(custom_palette):
            raise ValueError(
                "The segmentation output contains class IDs outside the hardcoded palette size. "
                "Please extend the palette in inference.py for this model."
            )

        segmentation_colored = custom_palette[segmentation_mask_np]
        segmentation_colored_image = Image.fromarray(segmentation_colored.astype(np.uint8), mode="RGB")

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{filename}")
        segmentation_colored_image.save(output_path)
        print(f"Segmented image saved at: {output_path}")
        num_images += 1

    total_time = time.time() - start_time
    images_per_second = num_images / total_time if total_time > 0 else 0
    print(f"\nTotal images processed: {num_images}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Segmented images per second: {images_per_second:.2f}")


if __name__ == "__main__":
    main()
