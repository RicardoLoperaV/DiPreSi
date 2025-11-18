# import necessary libraries
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    ConfusionMatrixDisplay, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, roc_curve
)


class PlantDataset(Dataset):
    """Custom Dataset for loading plant Fourier magnitude images"""
    
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing images (e.g., Day_0, Day_1, etc.)
            transform: Optional transform to be applied on images
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.samples = []
        
        # Regex pattern to extract information from filename
        # Sample_(?P<id>\d+)_(?P<estado>[a-zA-Z]+)_(?P<valor>\d+(?:\.\d+)?)_magnitude
        self.pattern = re.compile(r'Sample_(?P<id>\d+)_(?P<estado>[a-zA-Z]+)_(?P<valor>\d+(?:\.\d+)?)_magnitude\.png')
        
        # Load all images and their labels
        self._load_samples()
    
    def _load_samples(self):
        """Load all image paths and extract labels from filenames"""
        if not self.image_dir.exists():
            raise ValueError(f"Directory {self.image_dir} does not exist")
        
        for img_file in self.image_dir.glob("*.png"):
            match = self.pattern.match(img_file.name)
            if match:
                sample_id = int(match.group('id'))
                estado = match.group('estado')
                valor = float(match.group('valor'))
                
                # Convert estado to binary label: Healthy=1, Unhealthy=0
                label = 1 if estado == 'Healthy' else 0
                
                self.samples.append({
                    'path': img_file,
                    'label': label,
                    'id': sample_id,
                    'estado': estado,
                    'valor': valor
                })
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid images found in {self.image_dir}")
        
        print(f"Loaded {len(self.samples)} samples from {self.image_dir}")
        print(f"Healthy: {sum(1 for s in self.samples if s['label'] == 1)}, "
              f"Unhealthy: {sum(1 for s in self.samples if s['label'] == 0)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['path']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return image, label, sample['id']
    
    def get_labels(self):
        """Return all labels for stratified splitting"""
        return [s['label'] for s in self.samples]


class PlantCNN(nn.Module):
    """Convolutional Neural Network for plant health classification"""
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(PlantCNN, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Calculate the size after conv layers (assuming 224x224 input)
        # After 3 MaxPool2d layers: 224 -> 112 -> 56 -> 28
        self.fc_input_size = 128 * 28 * 28
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class CNNModelProductor:
    """Model producer for CNN-based plant health classification"""
    
    def __init__(self, 
                 days,
                 root_dir='c:/Users/ricar/Documents/GitHub/DiPreSi/Fourier imaging/Magnitude/Fourier_Images',
                 img_size=224,
                 batch_size=32,
                 test_size=0.2,
                 val_size=0.1,
                 random_state=42,
                 num_epochs=50,
                 learning_rate=0.001,
                 device=None):
        """
        Args:
            days (list or int): Day number(s) to use for training (e.g., [0, 1, 2] or 0)
            root_dir (str): Root directory containing Day_# folders
            img_size (int): Size to resize images to
            batch_size (int): Batch size for training
            test_size (float): Proportion of data to use for testing
            val_size (float): Proportion of training data to use for validation
            random_state (int): Random seed for reproducibility
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate for optimizer
            device: PyTorch device (cuda/cpu)
        """
        self.days = [days] if isinstance(days, int) else days
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize model and training components
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.criterion = None
        self.optimizer = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def load_data(self):
        """Load and split data into train, validation, and test sets"""
        print(f"\nLoading data from days: {self.days}")
        
        # Load datasets from specified days
        datasets = []
        for day in self.days:
            day_dir = self.root_dir / f"Day_{day}"
            dataset = PlantDataset(day_dir, transform=None)  # We'll apply transforms later
            datasets.append(dataset)
        
        # Combine all datasets
        if len(datasets) == 1:
            full_dataset = datasets[0]
        else:
            # Manually combine samples from multiple datasets
            combined_samples = []
            for dataset in datasets:
                combined_samples.extend(dataset.samples)
            
            full_dataset = PlantDataset.__new__(PlantDataset)
            full_dataset.image_dir = self.root_dir
            full_dataset.pattern = datasets[0].pattern
            full_dataset.samples = combined_samples
            full_dataset.transform = None
            
            print(f"Combined dataset: {len(full_dataset)} total samples")
            print(f"Healthy: {sum(1 for s in full_dataset.samples if s['label'] == 1)}, "
                  f"Unhealthy: {sum(1 for s in full_dataset.samples if s['label'] == 0)}")
        
        # Split into train+val and test
        torch.manual_seed(self.random_state)
        test_size_int = int(len(full_dataset) * self.test_size)
        train_val_size = len(full_dataset) - test_size_int
        
        train_val_dataset, test_dataset = random_split(
            full_dataset, 
            [train_val_size, test_size_int],
            generator=torch.Generator().manual_seed(self.random_state)
        )
        
        # Split train_val into train and validation
        val_size_int = int(train_val_size * self.val_size)
        train_size = train_val_size - val_size_int
        
        train_dataset, val_dataset = random_split(
            train_val_dataset,
            [train_size, val_size_int],
            generator=torch.Generator().manual_seed(self.random_state)
        )
        
        # Apply transforms by wrapping datasets
        train_dataset.dataset.transform = self.train_transform
        val_dataset.dataset.transform = self.test_transform
        test_dataset.dataset.transform = self.test_transform
        
        print(f"\nDataset splits:")
        print(f"  Training: {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples")
        print(f"  Testing: {len(test_dataset)} samples")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def build_model(self, num_classes=2, dropout_rate=0.5):
        """Build and initialize the CNN model"""
        print("\nBuilding CNN model...")
        self.model = PlantCNN(num_classes=num_classes, dropout_rate=dropout_rate)
        self.model = self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels, _ in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train_model(self, early_stopping_patience=10):
        """Train the model"""
        print("\nStarting training...")
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            print(f"Epoch [{epoch+1}/{self.num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate the model on test set"""
        print("\nEvaluating model on test set...")
        self.model.eval()
        
        all_labels = []
        all_predictions = []
        all_probabilities = []
        all_ids = []
        
        with torch.no_grad():
            for images, labels, ids in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_ids.extend(ids.numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
        
        print("\n" + "="*50)
        print("TEST SET RESULTS")
        print("="*50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, 
                                    target_names=['Unhealthy', 'Healthy']))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Store results
        results = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'sample_ids': all_ids
        }
        
        return results
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.history['train_acc'], label='Train Accuracy')
        axes[1].plot(self.history['val_acc'], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, results, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        
        disp = ConfusionMatrixDisplay(
            confusion_matrix=results['confusion_matrix'],
            display_labels=['Unhealthy', 'Healthy']
        )
        disp.plot(cmap='Blues', values_format='d')
        plt.title('Confusion Matrix - Test Set')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'days': self.days,
            'img_size': self.img_size
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if self.model is None:
            self.build_model()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        print(f"Model loaded from {filepath}")
        return self.model
    
    def run_full_pipeline(self, save_dir=None):
        """Run the complete training and evaluation pipeline"""
        # Load data
        self.load_data()
        
        # Build model
        self.build_model()
        
        # Train model
        self.train_model()
        
        # Evaluate model
        results = self.evaluate_model()
        
        # Plot results
        self.plot_training_history(
            save_path=os.path.join(save_dir, 'training_history.png') if save_dir else None
        )
        self.plot_confusion_matrix(
            results,
            save_path=os.path.join(save_dir, 'confusion_matrix.png') if save_dir else None
        )
        
        # Save model
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.save_model(os.path.join(save_dir, 'plant_cnn_model.pth'))
            
            # Save results to CSV
            results_df = pd.DataFrame({
                'sample_id': results['sample_ids'],
                'true_label': results['labels'],
                'predicted_label': results['predictions'],
                'prob_unhealthy': [p[0] for p in results['probabilities']],
                'prob_healthy': [p[1] for p in results['probabilities']]
            })
            results_df.to_csv(os.path.join(save_dir, 'test_predictions.csv'), index=False)
            print(f"Results saved to {save_dir}")
        
        return results

