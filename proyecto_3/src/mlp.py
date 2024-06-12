import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import numpy as np
import wandb


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.network.apply(self._init_weights)

        self.device = 'cpu'

    def forward(self, x):
        return self.network(x)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class MlpRun:
    def __init__(self, dataset_path, class_names, batch_size, hidden_sizes, output_size, dropout_rate, learning_rate,
                 device='cpu'):
        self.device = device
        self.batch_size = batch_size
        self.num_classes = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.class_names = class_names
        self.train_loader, self.val_loader, self.test_loader = self.load_dataset(dataset_path)

        # Get a single batch from the DataLoader
        data_iter = iter(self.train_loader)
        batch = next(data_iter)

        # Assuming the batch is a tuple of (inputs, labels)
        features, labels = batch

        self.model = MLP(input_size=features.shape[1],
                         hidden_sizes=self.hidden_sizes,
                         output_size=self.num_classes,
                         dropout_rate=self.dropout_rate).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.scheduler = None

        self.wandb = None

    def train(self, epochs):
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.01,
        #                                                      steps_per_epoch=len(self.train_loader), epochs=epochs)
        self.init_wandb(epochs)

        best_accuracy = 0.0
        for epoch in range(epochs):
            # Initialize empty numpy arrays for predictions and labels
            all_predictions = np.array([])
            all_labels = np.array([])
            all_outputs_proba = np.empty((0, self.num_classes))

            self.model.train()
            train_loss = 0.0
            for features, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                if self.scheduler is not None:
                    self.scheduler.step()

            train_loss /= len(self.train_loader)

            # Validation step after training
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for features, labels in self.val_loader:
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

                    # Apply softmax to get probabilities
                    probabilities = F.softmax(outputs, dim=1)
                    probabilities = probabilities.unsqueeze(0) if probabilities.ndim == 1 else probabilities

                    # Concatenate predictions and labels for the current batch
                    all_predictions = np.concatenate((all_predictions, outputs.argmax(dim=1).cpu().numpy()))
                    all_labels = np.concatenate((all_labels, labels.cpu().numpy()))
                    all_outputs_proba = np.concatenate((all_outputs_proba, probabilities.cpu().numpy()), axis=0)

                # Calculate average validation loss
                val_loss /= len(self.val_loader)

            # Log validation and metrics for the current epoch
            accuracy = self.log_metrics(train_loss, val_loss, all_labels, all_predictions, all_outputs_proba) * 100
            print(f"[{epoch} Val loss: {val_loss}")

            # Checkpointing
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_model(f"./models/{self.wandb.name}_mlp-model_{best_accuracy:.0f}.pth")
                print(f'New best model saved with accuracy: {best_accuracy:.2f}%')

        if self.wandb is not None:
            wandb.finish()

    def test(self):
        self.init_wandb_testing()
        self.model.eval()
        loss = 0.0
        # Initialize empty numpy arrays for predictions and labels
        all_predictions = np.array([])
        all_labels = np.array([])
        all_outputs_proba = np.empty((0, self.num_classes))

        with torch.no_grad():
            for data, label in self.test_loader:
                # Move data and label to device
                data, label = data.to(self.device), label.to(self.device)

                # Forward pass
                output = self.model(data)

                # Calculate validation loss
                loss += self.criterion(output, label).item()
                # Apply softmax to get probabilities
                probabilities = F.softmax(output, dim=1)
                probabilities = probabilities.unsqueeze(0) if probabilities.ndim == 1 else probabilities

                # Concatenate predictions and labels for the current batch
                all_predictions = np.concatenate((all_predictions, output.argmax(dim=1).cpu().numpy()))
                all_labels = np.concatenate((all_labels, label.cpu().numpy()))
                all_outputs_proba = np.concatenate((all_outputs_proba, probabilities.cpu().numpy()), axis=0)

            # Calculate average validation loss
            loss /= len(self.test_loader)

            # Log validation and metrics for the current epoch
            self.log_test_metrics(loss, all_labels, all_predictions, all_outputs_proba)

        if self.wandb is not None:
            wandb.finish()

    def log_metrics(self, train_loss, val_loss, all_labels, all_predictions, all_outputs_proba):
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='micro')
        recall = recall_score(all_labels, all_predictions, average='micro')
        # all_labels_binary = label_binarize(all_labels, classes=range(self.num_classes))
        #fpr, tpr, roc_auc = self.calculate_roc(all_labels_binary, all_outputs_proba)

        # Log to wandb
        self.wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=all_labels, preds=all_predictions, class_names=self.class_names
            ),
            # "fpr_hist": wandb.Histogram(np.histogram(fpr)),
            # "tpr_hist": wandb.Histogram(np.histogram(tpr)),
            # "roc_auc": roc_auc
        })

        return accuracy

    def log_test_metrics(self, loss, all_labels, all_predictions, all_outputs_proba):
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        all_labels_binary = label_binarize(all_labels, classes=range(self.num_classes))
        fpr, tpr, roc_auc = self.calculate_roc(all_labels_binary, all_outputs_proba)
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
                y_true=all_labels, preds=all_predictions, class_names=self.class_names
            ),
        })

    def calculate_roc(self, all_labels_binary, all_outputs_proba):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # Calculate ROC curve and ROC area for each class
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(all_labels_binary[:, i], all_outputs_proba[:, i])
            roc_auc[i] = roc_auc_score(all_labels_binary[:, i], all_outputs_proba[:, i])
        return fpr, tpr, roc_auc

    def load_dataset(self, dataset_path):
        features = np.load(f'{dataset_path}/features.npy')  # [num, 256]
        labels = np.load(f'{dataset_path}/labels.npy')  # [num]

        features = torch.tensor(features, dtype=torch.float, device=self.device)
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)

        print(f'Features dtype: {features.dtype}')
        print(f'Features shape: {features.shape}')
        print(f'Labels dtype: {labels.dtype}')
        print(f'Labels shape: {labels.shape}')

        # Create the dataset
        full_dataset = TensorDataset(features, labels)

        # Get the labels
        labels = np.array([label for _, label in full_dataset])

        train_indices, temp_indices = train_test_split(np.arange(len(full_dataset)), test_size=0.2, stratify=labels)
        val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, stratify=labels[temp_indices])

        # Create Samplers
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        # Create DataLoaders
        train_loader = DataLoader(full_dataset, batch_size=self.batch_size, sampler=train_sampler)
        val_loader = DataLoader(full_dataset, batch_size=self.batch_size, sampler=val_sampler)
        test_loader = DataLoader(full_dataset, batch_size=self.batch_size, sampler=test_sampler)
        return train_loader, val_loader, test_loader

    def init_wandb(self, epochs):
        self.wandb = wandb_run = wandb.init(
            project="covid19-ChestXRay",
            entity="university-of-costa-rica",
            config={
                "architecture": "MLP",
                "features": "LBP",
                "activation_function": "ReLU",
                "optimizer": self.optimizer.__class__.__name__,
                "criterion": self.criterion.__class__.__name__,
                "scheduler":  self.scheduler.__class__.__name__ if self.scheduler is not None else None,
                "dataset": "COVID-19 Chest X-Ray Database",
                "batch_size": self.batch_size,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "dropout_rate": self.dropout_rate,
                "hidden_sizes": self.hidden_sizes,
                "epochs": epochs,
                "device": self.device
            },
        )

    def init_wandb_testing(self):
        self.wandb = wandb_run = wandb.init(
            project="covid19-ChestXRay",
            entity="university-of-costa-rica",
            config={
                "architecture": "MLP",
                "optimizer": self.optimizer.__class__.__name__,
                "criterion": self.criterion.__class__.__name__,
                "dataset": "COVID-19 Chest X-Ray Database",
            },
        )

    def save_model(self, model_path):
        torch.save(self.model, model_path)
