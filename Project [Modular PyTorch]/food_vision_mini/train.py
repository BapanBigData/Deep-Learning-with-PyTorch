"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils
# Measure time
from timeit import default_timer as timer

from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.0001

# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
train_transform = transforms.Compose([
    # resize the image to 128X128
    transforms.Resize(size=(128, 128)),
    
    # flip the image randomly on the horizontal axis
    transforms.RandomHorizontalFlip(p=0.5),
    
    #transforms.RandomRotation(10),
    #transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    #transforms.ColorJitter(brightness=0.1, contrast=0.1),
    
    # turn the image into torch.Tensor
    transforms.ToTensor()
])


test_transform = transforms.Compose([
    # resize the image to 128X128
    transforms.Resize(size=(128, 128)),
    
    # turn the image into torch.Tensor
    transforms.ToTensor()
])


# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    train_transform=train_transform,
    test_transforms=test_transform,
    batch_size=BATCH_SIZE
)

train_time_start_model = timer()

# Create model with help from model_builder.py
model = model_builder.FoodModel(
    input_channels=3,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                                lr=LEARNING_RATE)

# Start training with help from engine.py
results = engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device
            )

train_time_end_model = timer()
total_train_time_model = utils.print_train_time(start=train_time_start_model,
                                        end=train_time_end_model,
                                        device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                    target_dir="models",
                    model_name="00_going_modular_script_mode_food_model.pth"
                )

# Save the training results
utils.save_results_to_json(results=results, 
                        filepath='training_artifact/training_results.json')
