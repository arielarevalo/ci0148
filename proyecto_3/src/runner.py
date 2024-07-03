import torch
from torch import nn
import torch.nn.functional as F
from tqdm.notebook import tqdm, trange

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize

from IPython.display import display

class Runner:
    def __init__(self, name, model, optimizer, criterion, device='cpu'):
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(self.device)

    def train(self, train_loader, val_loader, num_epochs=5, patience=5, val_loss_target=0.05):
        train_loss = float('inf')
        val_loss = float('inf')
        best_val_loss = float('inf')

        epochs_without_improvement = 0

        for epoch in trange(num_epochs, desc='Training', unit=' epoch'):
            train_loss = self._train_model(train_loader)
            val_loss = self._validate_model(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self._save_model(f'models/{self.name}.pth')
            else:
                epochs_without_improvement += 1

            print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

            if val_loss <= val_loss_target or epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
        return train_loss, val_loss

    def test(self, test_loader):
        self._load_model(f'models/{self.name}.pth')
        self.model.eval()

        num_classes = self._get_num_classes_from_loader(test_loader)

        loss = 0.0
        # Initialize empty numpy arrays for predictions and labels
        all_predictions = np.array([])
        all_labels = np.array([])
        all_outputs_proba = np.empty((0, num_classes))

        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc='Testing', unit='batch'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss += self.criterion(outputs, targets).item()

                # Apply softmax to get probabilities
                probabilities = F.softmax(outputs, dim=1)
                probabilities = probabilities.unsqueeze(0) if probabilities.ndim == 1 else probabilities

                # Concatenate predictions and labels for the current batch
                all_predictions = np.concatenate((all_predictions, outputs.argmax(dim=1).cpu().numpy()))
                all_labels = np.concatenate((all_labels, targets.cpu().numpy()))
                all_outputs_proba = np.concatenate((all_outputs_proba, probabilities.cpu().numpy()), axis=0)

            # Calculate average validation loss
            loss /= len(test_loader)

            # Log validation and metrics for the current epoch
            self.log_test_metrics(loss, all_labels, all_predictions, all_outputs_proba)

    def log_test_metrics(self, loss, all_labels, all_predictions, all_outputs_proba):
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        all_labels_binary = label_binarize(all_labels, classes=range(self.num_classes))
        fpr, tpr, roc_auc = self.calculate_roc(all_labels_binary, all_outputs_proba)

        # Display metrics in a table
        metrics_df = pd.DataFrame({
            'Metric': ['Loss', 'Accuracy', 'Precision', 'Recall'],
            'Value': [loss, accuracy, precision, recall]
        })

        display(metrics_df)

        # Display ROC AUC for each class
        auc_df = pd.DataFrame({
            'Class': list(range(self.num_classes)),
            'AUC': [roc_auc[i] for i in range(self.num_classes)]
        })

        display(auc_df)

        # Plot ROC Curves
        plt.figure(figsize=(10, 8))
        for i in range(self.num_classes):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.show()

        # Confusion Matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(self.num_classes),
                    yticklabels=range(self.num_classes))
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

    def calculate_roc(self, all_labels_binary, all_outputs_proba):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # Calculate ROC curve and ROC area for each class
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(all_labels_binary[:, i], all_outputs_proba[:, i])
            roc_auc[i] = roc_auc_score(all_labels_binary[:, i], all_outputs_proba[:, i])
        return fpr, tpr, roc_auc

    def _train_model(self, train_loader):
        self.model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(train_loader, leave=False, desc='Training', unit='batch'):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        return avg_loss

    def _validate_model(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, leave=False, desc='Validating', unit='batch'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

        avg_loss = running_loss / len(val_loader)
        return avg_loss

    def _get_num_classes_from_loader(test_loader):
        all_labels = []
        for _, labels in test_loader:
            all_labels.extend(labels.numpy())
        num_classes = len(set(all_labels))
        return num_classes

    def _save_model(self, path):
        torch.save(self.model, path)

    def _load_model(self, path):
        self.model = torch.load(path)
        self.model.to(self.device)

    def _reset_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()
