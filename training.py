import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import optuna
import wandb
from tqdm import tqdm
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
from sklearn.metrics import jaccard_score


# Paths
train_image_dir = ""
train_mask_dir = ""

val_image_dir = ""
val_mask_dir = ""

model_dir = ""
output_dir = ""



device = "cuda" if torch.cuda.is_available() else "cpu"



# Global variables to store the best model
global_best_val_loss = float('inf')
global_best_state = None

global_encoder_frozen_epochs = 0
global_encoder_unfrozen_epochs = 0



global_best_metrics = {
    "final_epoch": None,
    "final_val_loss": None,
    "final_val_iou": None,
    "final_train_loss": None,
    "final_train_iou": None
}




def save_global_best(trial_loss, model_state, epoch, val_iou, train_loss, train_iou, encoder_frozen_epochs, encoder_unfrozen_epochs):
    global global_best_val_loss, global_best_state, global_best_metrics, global_encoder_frozen_epochs, global_encoder_unfrozen_epochs
    if trial_loss < global_best_val_loss:
        global_best_val_loss = trial_loss
        global_best_state = model_state
        global_best_metrics["final_epoch"] = epoch
        global_best_metrics["final_val_loss"] = trial_loss
        global_best_metrics["final_val_iou"] = val_iou
        global_best_metrics["final_train_loss"] = train_loss
        global_best_metrics["final_train_iou"] = train_iou
        global_encoder_frozen_epochs = encoder_frozen_epochs
        global_encoder_unfrozen_epochs = encoder_unfrozen_epochs




# Function to check if all images have a corresponding mask
def verificar_pares_correspondentes(image_dir, mask_dir, str, prefix_mask="new_classes_"):
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('jpg', 'jpeg', 'png', 'tif'))])
    
    missing_masks = []
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        expected_mask = f"{prefix_mask}{base_name}.png"
        mask_path = os.path.join(mask_dir, expected_mask)
        if not os.path.exists(mask_path):
            missing_masks.append(expected_mask)

    print("")
    print(f"Total images for {str}: {len(image_files)}")
    print(f"Missing masks: {len(missing_masks)}")
    if missing_masks:
        print("Examples of missing masks:")
        for m in missing_masks[:5]:
            print(f"- {m}")
    else:
        print(f"All images for {str} have corresponding masks!")




# Dataset
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, processor):
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('jpg', 'jpeg', 'png', 'tif'))])
        self.image_paths = [os.path.join(image_dir, f) for f in self.image_files]
        self.mask_paths = [os.path.join(mask_dir, f"new_classes_{os.path.splitext(f)[0]}.png") for f in self.image_files]
        self.processor = processor
        self.target_size = (1280, 640)  # width, height

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])
        self.processor.image_processor.size = {"height": self.target_size[1], "width": self.target_size[0]}
        inputs = self.processor(images=image, task_inputs=["semantic"], return_tensors="pt")
  

        # Resize mask
        mask = mask.resize(self.target_size, resample=Image.NEAREST)
        mask_tensor = torch.tensor(np.array(mask), dtype=torch.long)

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "task_inputs":  inputs["task_inputs"].squeeze(0),
            "mask": mask_tensor
        }



# Main training function for Optuna
def train_oneformer(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    num_epochs = trial.suggest_int("num_epochs", 50, 100)


    wandb.init(
        project="project_name",
        name=f"trial_{trial.number}",
        config={
            "learning_rate": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "epochs": num_epochs
        },
        reinit=True
    )



    # Load model and processor
    processor = OneFormerProcessor.from_pretrained(model_dir)
    model = OneFormerForUniversalSegmentation.from_pretrained(model_dir, ignore_mismatched_sizes=True).to(device)
    
    # Access model configuration
    config = model.config

    # Check number of classes
    print(f"Number of classes: {config.num_labels}")


    # Check if images have corresponding masks
    verificar_pares_correspondentes(train_image_dir, train_mask_dir, "training")
    verificar_pares_correspondentes(val_image_dir, val_mask_dir, "validation")

    

    # DataLoaders
    train_loader = DataLoader(SegmentationDataset(train_image_dir, train_mask_dir, processor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SegmentationDataset(val_image_dir, val_mask_dir, processor), batch_size=1, shuffle=False)


    # Optimizer and loss
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Early stopping setup
    best_val_loss = float('inf')
    encoder_unfrozen = True
    patience = 7
    counter = 0

    encoder_frozen_epochs = 0
    encoder_unfrozen_epochs = 0


    # Training/validation loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss, train_iou = 0.0, []


        # Freeze encoder
        if encoder_unfrozen and epoch == num_epochs // 2:
            print("\n!!!!!!!!!!!!!!!!!!!! Encoder frozen, adjusting batch_size to 2 !!!!!!!!!!!!!!!!!!!!\n")
            for name, param in model.named_parameters():
                param.requires_grad = not ("encoder" in name)
                print("**************************************************")
            encoder_unfrozen = False
            batch_size = 2
            train_loader = DataLoader(SegmentationDataset(train_image_dir, train_mask_dir, processor), batch_size=batch_size, shuffle=True)


        if encoder_unfrozen:
            encoder_unfrozen_epochs += 1
        else:
            encoder_frozen_epochs += 1


        for batch in tqdm(train_loader, desc=f"Training - Trial_{trial.number} - Epoch {epoch+1}"):

            print(f"Batch size atual: {batch['pixel_values'].shape[0]}")

            optimizer.zero_grad()
            inputs = {"pixel_values": batch["pixel_values"].to(device),
                      "task_inputs":  batch["task_inputs"].to(device)}
            
            masks = batch["mask"].to(device)

            print("Unique values in the mask:", torch.unique(masks))
            print("******************************************************")

            outputs = model(**inputs)

            
            class_probs = torch.softmax(outputs.class_queries_logits, dim=-1)
            mask_probs = torch.softmax(outputs.transformer_decoder_mask_predictions, dim=1)
            logits = torch.einsum("bqhw,bqc->bchw", mask_probs, class_probs)
            logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            logits = logits[:, :model.config.num_labels, :, :]

            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = masks.cpu().numpy()
            for p, t in zip(preds, targets):
                train_iou.append(jaccard_score(t.flatten(), p.flatten(), average="weighted", zero_division=0))




        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = np.mean(train_iou)

        # Validation
        model.eval()
        val_loss, val_iou = 0.0, []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation - Trial_{trial.number} - Epoch {epoch+1}"):
                inputs = {"pixel_values": batch["pixel_values"].to(device),
                          "task_inputs":  batch["task_inputs"].to(device)}
                
                masks = batch["mask"].to(device)

                outputs = model(**inputs)

                class_logits = outputs.class_queries_logits
                mask_logits = outputs.transformer_decoder_mask_predictions
                class_probs = torch.softmax(class_logits, dim=-1)
                mask_probs = torch.softmax(mask_logits, dim=1)
                logits = torch.einsum("bqhw,bqc->bchw", mask_probs, class_probs)
                logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)

                logits = logits[:, :model.config.num_labels, :, :]

                val_loss += criterion(logits, masks).item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                targets = masks.cpu().numpy()
                for p, t in zip(preds, targets):
                    val_iou.append(jaccard_score(t.flatten(), p.flatten(), average="weighted", zero_division=0))



        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = np.mean(val_iou)

        wandb.log({
            "Training Loss": avg_train_loss,
            "Validation Loss": avg_val_loss,
            "Training mIoU": avg_train_iou,
            "Validation mIoU": avg_val_iou,
            "Epoch": epoch+1
        })

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | mIoU Val: {avg_val_iou:.4f}")

        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            
            if counter >= patience:
                print("Early stopping triggered.")
                break

    # Update global if this trial is the best one
    save_global_best(best_val_loss, best_model_state, epoch + 1, avg_val_iou, avg_train_loss, avg_train_iou, encoder_frozen_epochs, encoder_unfrozen_epochs)



    wandb.finish()
    return best_val_loss




if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(train_oneformer, n_trials=10)

    
    os.makedirs(output_dir, exist_ok=True)
    # Reload architecture and inject optimal weights
    processor = OneFormerProcessor.from_pretrained(model_dir)
    model = OneFormerForUniversalSegmentation.from_pretrained(model_dir, ignore_mismatched_sizes=True)
    model.load_state_dict(global_best_state)
    model.to(device)
    # Save
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Global best model saved at {output_dir} with loss {global_best_val_loss:.4f}")

    # Save only the best trial number and its hyperparameters
    best_trial = study.best_trial



