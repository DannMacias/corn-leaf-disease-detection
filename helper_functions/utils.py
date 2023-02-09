"""
Contains utility function for save the model, plot loss curves and
visualize images from dataloaders
"""
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
from PIL import Image

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """ Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should inclue
      either ".pth" or ".pt" as the file extension.

  Example usage:
  save_model(model = effnetb2,
            target_dir = "models",
            model_name = "01_effnetb2_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents = True,
                        exist_ok = True)
  
  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj = model.state_dict(),
             f = model_save_path)

def plot_loss_curves(results):
  """Plots training curves of a results dictionary (from train.py) 
  """
  # Get the loss and accuracy from the results dictionary.
  # The dictionary "results" is the output from train.py
  loss = results["train_loss"]
  test_loss = results["test_loss"]
  accuracy = results["train_acc"]
  test_accuracy = results["test_acc"]

  epochs = range(len(results["train_loss"]))

  plt.figure(figsize = (15, 7))

  # Plot loss curves
  plt.subplot(1, 2, 1)
  plt.plot(epochs, loss, label="train_loss")
  plt.plot(epochs, test_loss, label="test_loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.legend

  # Plot accuracy curves
  plt.subplot(1, 2, 2)
  plt.plot(epochs, accuracy, label="train_accuracy")
  plt.plot(epochs, test_accuracy, label="test_accuracy")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.legend

def view_dataloader_images(dataloader, n=10):
  """Visualize images from a dataloader (with and without data augmentation)

  Args:
    dataloader: A dataloader instance (torch.utils.data.DataLoader)
  """
  # From visualization purpose, the number of images (n) should be 10 or lower
  if n > 10:
    print(f"Having n higher than 10 will create messy plot, lowering to 10")
    n = 10
  
  imgs, labels = next(iter(dataloader))
  plt.figure(figsize = (16, 8))
  for i in range(n):
    # Min max scale the image for display purpose
    targ_image = imgs[i]
    sample_min, sample_max = targ_image.min(), targ_image.max()
    sample_scaled = (targ_image - sample_min)/(sample_max - sample_min)

    # Plot images with appropiate axes information
    plt.subplot(1, 10, i+1)
    plt.imshow(sample_scaled.permute(1, 2, 0)) # resize for matplotlib requirements
    plt.title(class_names[labels[i]])
    plt.axis(False)
