import yaml
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch
import os
import time

# Load YAML configuration
with open("/path/to/inference.yaml", "r") as file:
    config_yaml = yaml.safe_load(file)

# Paths from YAML file
model_dir = config_yaml["model_dir"]
image_dir = config_yaml["image_dir"]
output_dir = config_yaml["output_dir"]

# Load model and processor
model = OneFormerForUniversalSegmentation.from_pretrained(model_dir)
processor = OneFormerProcessor.from_pretrained(model_dir)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Access model configuration
config = model.config


print("")

print(f"Total number of model classes: {config.num_labels}")


model.to(device)



# Color palette
Custom_Palette = np.array([
                [0, 0, 0], [120, 120, 70], [255, 170, 146], [61, 230, 250], [204, 255, 4], [4, 250, 7], [12, 189, 102],
                [255, 41, 10], [51, 0, 255], [150, 5, 61], [255, 0, 122], [0, 140, 0], [100, 65, 0]
])


# List of classes
Custom_Classes = {
    0: 'other', 1: 'soil', 2: 'trunk', 3: 'water', 4: 'vegetation', 5: 'low grass', 6: 'high grass', 7: 'stone',
    8: 'stump', 9: 'person', 10: 'animal', 11: 'canopy', 12: 'mud'
}



# Processed image counter
num_images = 0

# Start time
start_time = time.time()




for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):  
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)

        print("")
        
        print(f"Image mode: {image.mode}")

        inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(device)

        with torch.no_grad():              
            model.eval()                   
            outputs = model(**inputs)

        
        segmentation_mask = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

        segmentation_colored = Custom_Palette[segmentation_mask.cpu().numpy()]    

        # Create the colorized segmented image
        segmentation_colored_image = Image.fromarray(segmentation_colored.astype(np.uint8), mode='RGB')

        # Identify classes present in the segmented image
        unique_classes = np.unique(segmentation_mask.cpu().numpy())


        print(f"Identified classes: {unique_classes}")

 

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save segmented image
        output_path = os.path.join(output_dir, f"{filename}")
        segmentation_colored_image.save(output_path)
        
        
        print(f"Segmented image saved at: {output_path}")

        num_images += 1  




end_time = time.time()


total_time = end_time - start_time
images_per_second = num_images / total_time if total_time > 0 else 0

print(f"\nTotal images processed: {num_images}")
print(f"Total time: {total_time:.2f} seconds")
print(f"Segmented images per second: {images_per_second:.2f}")
