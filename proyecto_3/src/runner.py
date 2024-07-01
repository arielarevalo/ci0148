import torch
from torch.utils.data import DataLoader, Subset
from torch import nn, optim
import numpy as np
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm, trange


class Runner:
    def __init__(self, name, model, optimizer, criterion, dataset, num_folds=5, batch_size=128, device='cpu'):
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataset = dataset
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.device = device
        self.model.to(self.device)

    def cross_validate(self):
        kf = KFold(n_splits=self.num_folds, shuffle=True)
        fold_results = []
        train_results = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(self.dataset)):
            print(f'Fold {fold + 1}/{self.num_folds}')
            train_subset = Subset(self.dataset, train_idx)
            test_subset = Subset(self.dataset, test_idx)
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)

            # Reinitialize the model and optimizer for each fold
            self.model.apply(self._reset_weights)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.optimizer.defaults['lr'])

            train_accuracy, train_loss = self._train_model(train_loader)
            fold_results.append((train_loss, train_accuracy))
            test_accuracy, test_loss = self.test(test_loader)
            fold_results.append((test_loss, test_accuracy))
            print(f'Fold {fold + 1} - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')

        avg_loss = np.mean([result[0] for result in fold_results])
        avg_accuracy = np.mean([result[1] for result in fold_results])
        print(f'Cross-Validation - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}')
        print(f'Training - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}')

    def train(self, train_loader, val_loader, num_epochs=5, patience=5, val_loss_target=0.05):
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        val_loss = float('inf')
        train_accuracy = float('inf')
        train_loss = float('inf')

        for epoch in trange(num_epochs, desc='Training', unit=' epoch'):
            train_accuracy, train_loss = self._train_model(train_loader)
            val_loss = self._validate_model(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self.save_model(f'models/{self.name}.pth')
            else:
                epochs_without_improvement += 1

            print(f'Epoch {epoch + 1}/{num_epochs} - Train accuracy: {train_accuracy:.4f},'
                  f' Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

            if val_loss <= val_loss_target or epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
        return train_accuracy, train_loss, val_loss

    def test(self, test_loader):
        self.load_model(f'models/{self.name}.pth')
        self.model.eval()

        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc='Testing', unit='batch'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = correct / total
        avg_loss = test_loss / len(test_loader)

        print(f'Test accuracy: {accuracy:.4f}, Test Loss: {avg_loss:.4f}')

        return accuracy, avg_loss

    def save_model(self, path):
        torch.save(self.model, path)

    def load_model(self, path):
        self.model = torch.load(path)
        self.model.to(self.device)

    def _reset_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    def _train_model(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in tqdm(train_loader, leave=False, desc='Training', unit='batch'):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        accuracy = correct / total
        avg_loss = running_loss / len(train_loader)
        return accuracy, avg_loss

    def _validate_model(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, leave=False, desc='Validating', unit='batch'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

        avg_loss = val_loss / len(val_loader)
        return avg_loss
