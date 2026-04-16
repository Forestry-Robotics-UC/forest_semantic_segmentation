import argparse
import os
import sys
from pprint import pprint

import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import transformers.utils.import_utils as tf_import_utils

tf_import_utils._natten_available = True

from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor


def parse_args():
    default_config = os.environ.get(
        "ARCHITECTURE_MODIFICATION_CONFIG",
        os.path.join(os.path.dirname(__file__), "architecture_modification.yaml"),
    )
    parser = argparse.ArgumentParser(description="Adapt OneFormer classifier head to custom forest classes.")
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to architecture_modification.yaml (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)

    pretrained_model_path = config_yaml["pretrained_model_path"]
    model_path = config_yaml["model_path"]
    new_classes = config_yaml["new_classes"]
    existing_class_map = config_yaml.get("existing_class_map", {})

    num_classes = len(new_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = OneFormerProcessor.from_pretrained(pretrained_model_path)
    model = OneFormerForUniversalSegmentation.from_pretrained(pretrained_model_path).to(device)

    original_id2label = model.config.id2label
    original_label2id = {v: k for k, v in original_id2label.items()}

    print("")
    print(original_id2label)
    print("")

    old_class_embed = model.model.transformer_module.decoder.class_embed
    in_features = old_class_embed.in_features
    new_class_embed = nn.Linear(in_features, num_classes).to(device)

    for idx, label in enumerate(new_classes):
        if label in existing_class_map:
            old_labels = existing_class_map[label]
            if isinstance(old_labels, str):
                old_labels = [old_labels]

            missing_labels = [ol for ol in old_labels if ol not in original_label2id]
            valid_labels = [ol for ol in old_labels if ol in original_label2id]

            if valid_labels:
                valid_indices = [original_label2id[ol] for ol in valid_labels]
                new_weight = torch.stack([old_class_embed.weight.data[i] for i in valid_indices]).mean(dim=0)
                new_bias = torch.stack([old_class_embed.bias.data[i] for i in valid_indices]).mean(dim=0)

                new_class_embed.weight.data[idx] = new_weight
                new_class_embed.bias.data[idx] = new_bias

                print(f"Class '{label}' initialized from: {valid_labels}")
                if missing_labels:
                    print(
                        "The following labels were ignored because they don't exist in the original model: "
                        f"{missing_labels}\n"
                    )
                else:
                    print()
            else:
                print(f"No valid label found for class '{label}'. Will initialize randomly.\n")
        else:
            print(f"Class '{label}' will be initialized randomly.\n")

    print("")

    model.model.transformer_module.decoder.class_embed = new_class_embed
    model.config.num_labels = num_classes
    model.criterion.empty_weight = nn.Parameter(torch.ones(num_classes))

    print("criterion.empty_weight shape:", model.criterion.empty_weight.shape)
    print("")

    processor.label2id = {label: idx for idx, label in enumerate(new_classes)}
    processor.id2label = {idx: label for idx, label in enumerate(new_classes)}
    model.config.id2label = processor.id2label
    model.config.label2id = processor.label2id

    print("model.config.id2label:")
    pprint(model.config.id2label)
    print("")
    print("\nmodel.config.label2id:")
    pprint(model.config.label2id)
    print("")
    print(f"\nmodel.config.num_labels = {model.config.num_labels}")
    print("len(id2label) =", len(model.config.id2label))
    print("len(label2id) =", len(model.config.label2id))
    print("")
    print(f"***** New Model ({num_classes} classes) *****")
    print(model.model.transformer_module.decoder.class_embed)
    print("id2label:", processor.id2label)
    print("label2id:", processor.label2id)
    print("")
    print("Weight for new class_embed", new_class_embed.weight.shape)
    print("Config num_labels:", model.config.num_labels)
    print("")

    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    processor.save_pretrained(model_path)
    print("Adapted model successfully saved at:", model_path)


if __name__ == "__main__":
    main()
