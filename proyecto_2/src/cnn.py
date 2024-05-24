import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models, datasets, transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SubsetRandomSampler, SubsetRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import numpy as np
import wandb

class CNN_Model:
  def __init__(self, num_classes, class_names, project_name="my-awesome-project"):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.num_classes = num_classes
    self.class_names = class_names
    self.project_name = project_name
    self.model = self.initialize_model()
    self.criterion = nn.CrossEntropyLoss()

  def initialize_model(self):
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    for param in model.parameters():
      param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
      nn.Linear(num_features, 128),
      nn.ReLU(inplace=True),
      nn.Linear(128, self.num_classes)
    )
    return model.to(self.device)

  def train(self, train_loader, val_loader, epochs=20, patience=5, learning_rate=0.001):
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience)
    self.init_wandb(epochs, patience, learning_rate)
    for epoch in range(epochs):
      # Initialize empty numpy arrays for predictions and targets
      all_predictions = np.array([])
      all_targets = np.array([])
      all_outputs_proba = np.empty((0, self.num_classes))
    
      # Training phase
      for data, target in train_loader:
        # Move data and target to device
        data, target = data.to(self.device), target.to(self.device)

        # Forward pass, calculate loss
        output = self.model(data)
        loss = self.criterion(output, target)

        # Backpropagation, update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

      # Validation phase
      val_loss = 0.0
      with torch.no_grad():
        for data, target in val_loader:
          # Move data and target to device
          data, target = data.to(self.device), target.to(self.device)

          # Forward pass
          output = self.model(data)

          # Calculate validation loss
          val_loss += self.criterion(output, target).item()
          # Apply softmax to get probabilities
          probabilities = F.softmax(output, dim=1)
          probabilities = probabilities.unsqueeze(0) if probabilities.ndim == 1 else probabilities

          # Concatenate predictions and targets for the current batch
          all_predictions = np.concatenate((all_predictions, output.argmax(dim=1).cpu().numpy()))
          all_targets = np.concatenate((all_targets, target.cpu().numpy()))
          all_outputs_proba = np.concatenate((all_outputs_proba, probabilities.cpu().numpy()), axis=0)

        # Calculate average validation loss
        val_loss /= len(val_loader)

        # Log validation and metrics for the current epoch
        self.log_metrics(val_loss, all_targets, all_predictions, all_outputs_proba)

        if not early_stopping(epochs, logs={'val_loss': val_loss}): break
    if self.wandb is not None:
      wandb.finish()
  
  
  def test(self, test_loader):
    self.init_wandb_testing()
    loss = 0.0
    # Initialize empty numpy arrays for predictions and targets
    all_predictions = np.array([])
    all_targets = np.array([])
    all_outputs_proba = np.empty((0, self.num_classes))
  
    with torch.no_grad():
      for data, target in test_loader:
        # Move data and target to device
        data, target = data.to(self.device), target.to(self.device)

        # Forward pass
        output = self.model(data)

        # Calculate validation loss
        loss += self.criterion(output, target).item()
        # Apply softmax to get probabilities
        probabilities = F.softmax(output, dim=1)
        probabilities = probabilities.unsqueeze(0) if probabilities.ndim == 1 else probabilities

        # Concatenate predictions and targets for the current batch
        all_predictions = np.concatenate((all_predictions, output.argmax(dim=1).cpu().numpy()))
        all_targets = np.concatenate((all_targets, target.cpu().numpy()))
        all_outputs_proba = np.concatenate((all_outputs_proba, probabilities.cpu().numpy()), axis=0)

      # Calculate average validation loss
      loss /= len(test_loader)

      # Log validation and metrics for the current epoch
      self.log_test_metrics(loss, all_targets, all_predictions, all_outputs_proba)

    if self.wandb is not None:
      wandb.finish()

  
  def init_wandb(self, epochs, patience, learning_rate):
    self.wandb = wandb_run = wandb.init(
      # set the wandb project where this run will be logged
      project=self.project_name,

      # track hyperparameters and run metadata
      config={
          "pretrained_model": "RestNet-50",
          "architecture": "CNN",
          "optimizer": "Adam",
          "criterion": "Cross entropy loss",
          "dataset": "COVID-19 Chest X-Ray Database",
          "learning_rate": learning_rate,
          "epochs": epochs,
          "patience": patience
      },
    )

  def init_wandb_testing(self):
    self.wandb = wandb_run = wandb.init(
      # set the wandb project where this run will be logged
      project=self.project_name,

      # track hyperparameters and run metadata
      config={
          "pretrained_model": "RestNet-50",
          "architecture": "CNN",
          "optimizer": "Adam",
          "criterion": "Cross entropy loss",
          "dataset": "COVID-19 Chest X-Ray Database",
      },
    )

  def log_metrics(self, val_loss, all_targets, all_predictions, all_outputs_proba):
      accuracy = accuracy_score(all_targets, all_predictions)
      precision = precision_score(all_targets, all_predictions, average='weighted')
      recall = recall_score(all_targets, all_predictions, average='weighted')
      all_targets_binary = label_binarize(all_targets, classes=range(self.num_classes))
      fpr, tpr, roc_auc = self.calculate_roc(all_targets_binary, all_outputs_proba)

      fpr_tpr_table = wandb.Table(columns=["fpr", "tpr"])
      fpr_tpr_table.add_data(
        fpr,
        tpr,
      )

      # Log to wandb
      self.wandb.log({
        "val_loss": val_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=all_targets, preds=all_predictions, class_names=self.class_names
        ),
        "roc_auc": roc_auc,
        "fpr_tpr": fpr_tpr_table
      })

  def log_test_metrics(self, loss, all_targets, all_predictions, all_outputs_proba):
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    all_targets_binary = label_binarize(all_targets, classes=range(self.num_classes))
    fpr, tpr, roc_auc = self.calculate_roc(all_targets_binary, all_outputs_proba)
    # create a Table with the same columns as above,
    # plus confidence scores for all labels
    columns = ["loss", "accuracy", "precision", "recall", "fpr", "tpr", "roc_auc"]

    test_table = wandb.Table(columns=columns)
    test_table.add_data(
      loss,
      accuracy,
      precision,
      recall,
      fpr,
      tpr,
      roc_auc,
    )

    self.wandb.log({
      "test_results": test_table,
      "confusion_matrix": wandb.plot.confusion_matrix(
        y_true=all_targets, preds=all_predictions, class_names=self.class_names
      ),
    })

    # self.wandb.log({
    #   "val_loss": loss,
    #   "accuracy": accuracy,
    #   "precision": precision,
    #   "recall": recall,
    #   "confusion_matrix": wandb.plot.confusion_matrix(
    #       y_true=all_targets, preds=all_predictions, class_names=self.class_names
    #   ),
    #   "fpr": fpr,
    #   "tpr": tpr,
    #   "roc_auc": roc_auc
    # })
    
  def calculate_roc(self, all_targets_binary, all_outputs_proba):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # Calculate ROC curve and ROC area for each class
    for i in range(self.num_classes):
      fpr[i], tpr[i], _ = roc_curve(all_targets_binary[:, i], all_outputs_proba[:, i])
      roc_auc[i] = roc_auc_score(all_targets_binary[:, i], all_outputs_proba[:, i])
    return fpr, tpr, roc_auc
  
  def save_model(self, model_path):
    torch.save(self.model.state_dict(), model_path)
    
  @staticmethod
  def load_dataset(dataset_path, transform):
    # Create the dataset
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # Get the labels
    labels = np.array([label for _, label in full_dataset])

    train_indices, temp_indices = train_test_split(np.arange(len(full_dataset)), test_size=0.2, stratify=labels)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, stratify=labels[temp_indices])

    # Create Samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create DataLoaders
    train_loader = DataLoader(full_dataset, batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(full_dataset, batch_size=32, sampler=val_sampler)
    test_loader = DataLoader(full_dataset, batch_size=32, sampler=test_sampler)
    return train_loader, val_loader, test_loader

class EarlyStopping(object):
  def __init__(self, patience=5):
    self.patience = patience
    self.best_val_loss = float('inf')
    self.counter = 0

  def __call__(self, epoch, logs):
    val_loss = logs.get('val_loss')
    if val_loss < self.best_val_loss:
      self.best_val_loss = val_loss
      self.counter = 0
    else:
      self.counter += 1
      if self.counter >= self.patience:
        print(f"Early stopping triggered after {self.patience} epochs with no improvement.")
        return False
    return True
