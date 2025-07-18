from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from pprint import pprint
import torch.nn as nn
import torch
import os


# Define path to save the new model
model_path = ""


# Define list of new semantic classes
new_classes = [
    "other", "soil", "trunk", "water",
    "vegetation", "low_grass", "high_grass", "stone", "stump", "person",
    "animal", "canopy", "mud"
]

NUM_CLASSES = len(new_classes)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load pre-trained model and processor
processor = OneFormerProcessor.from_pretrained("")
model = OneFormerForUniversalSegmentation.from_pretrained("").to(device)


existing_class_map = {}


original_id2label = model.config.id2label
original_label2id = {v: k for k, v in original_id2label.items()}

print("")

print(original_id2label)

print("")


# Replace model head and update config
old_class_embed = model.model.transformer_module.decoder.class_embed
in_features = old_class_embed.in_features


new_class_embed = nn.Linear(in_features, NUM_CLASSES).to(device)


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
                print(f"The following labels were ignored because they don't exist in the original model: {missing_labels}\n")
            else:
                print()
        else:
            print(f"No valid label found for class '{label}'. Will initialize randomly.\n")
    else:
        print(f"Class '{label}' will be initialized randomly.\n")


print("")



model.model.transformer_module.decoder.class_embed = new_class_embed
model.config.num_labels = NUM_CLASSES

model.criterion.empty_weight = nn.Parameter(torch.ones(NUM_CLASSES))


print("criterion.empty_weight shape:", model.criterion.empty_weight.shape)

print("")

# Update processor metadata
processor.label2id = {label: idx for idx, label in enumerate(new_classes)}
processor.id2label = {idx: label for idx, label in enumerate(new_classes)}

model.config.id2label = processor.id2label
model.config.label2id = processor.label2id


# Show index-to-name mapping
print("model.config.id2label:")
pprint(model.config.id2label)

print("")

# Show name-to-index mapping
print("\nmodel.config.label2id:")
pprint(model.config.label2id)

print("")

# Check if num_labels matches the size of the dictionaries
print(f"\nmodel.config.num_labels = {model.config.num_labels}")
print("len(id2label) =", len(model.config.id2label))
print("len(label2id) =", len(model.config.label2id))

print("")


print(f"***** New Model ({NUM_CLASSES} classes) *****")
print(model.model.transformer_module.decoder.class_embed)
print("id2label:", processor.id2label)
print("label2id:", processor.label2id)

print("")

print("Weight for new class_embed", new_class_embed.weight.shape)
print("Config num_labels:", model.config.num_labels)

print("")

# Create directory if it doesn't exist
os.makedirs(model_path, exist_ok=True)

# Save model and processor
model.save_pretrained(model_path)
processor.save_pretrained(model_path)

print("Adapted model successfully saved at:", model_path)
