"""Training engine for GAT model supervised learning.

This module handles:
1. Data splitting with stratified sampling
2. Class weight calculation for imbalanced data
3. Training loop with early stopping
4. Evaluation metrics (F1, Accuracy, Confusion Matrix)

Logic adapted from colab-code/train.py (Training section, lines 188-421)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from torch_geometric.data import Data
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import time

from models.gat_model import SybilGAT


@dataclass
class TrainingConfig:
    """Configuration for training."""
    hidden_channels: int = 32
    heads: int = 4
    dropout: float = 0.3
    learning_rate: float = 0.005
    weight_decay: float = 5e-4
    epochs: int = 300
    patience: int = 40
    test_size: float = 0.4
    val_size: float = 0.5  # Of test_size
    random_state: int = 42


@dataclass
class TrainingHistory:
    """Container for training history."""
    epochs: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    val_f1: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_f1: float = 0.0


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    accuracy: float
    f1_macro: float
    confusion_matrix: np.ndarray
    classification_report: str
    y_true: np.ndarray
    y_pred: np.ndarray


class GATrainer:
    """
    Trainer for Graph Attention Network.
    
    Logic adapted from colab-code/train.py (lines 188-407)
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        device: str = None
    ):
        self.config = config or TrainingConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.history = TrainingHistory()
        self.best_model_state = None
        
    def prepare_data(
        self, 
        data: Data, 
        labels: np.ndarray
    ) -> Tuple[Data, torch.Tensor]:
        """
        Prepare data with train/val/test masks.
        
        Logic from colab-code/train.py (lines 193-229)
        """
        num_nodes = data.num_nodes
        indices = list(range(num_nodes))
        
        # Stratified split
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=self.config.test_size,
            stratify=labels,
            random_state=self.config.random_state
        )
        
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=self.config.val_size,
            stratify=labels[temp_idx],
            random_state=self.config.random_state
        )
        
        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data.y = torch.tensor(labels, dtype=torch.long)
        
        # Calculate class weights
        y_train = labels[train_idx]
        num_sybils = (y_train == 1).sum()
        num_nonsybils = (y_train == 0).sum()
        total_train = num_sybils + num_nonsybils
        
        weight_sybil = total_train / (2 * num_sybils) if num_sybils > 0 else 1.0
        weight_nonsybil = total_train / (2 * num_nonsybils) if num_nonsybils > 0 else 1.0
        
        class_weights = torch.tensor([weight_nonsybil, weight_sybil], dtype=torch.float)
        
        return data, class_weights
    
    def initialize_model(
        self, 
        num_features: int,
        class_weights: torch.Tensor
    ) -> None:
        """
        Initialize model, optimizer, and loss function.
        
        Logic from colab-code/train.py (lines 273-302)
        """
        self.model = SybilGAT(
            num_features=num_features,
            hidden_channels=self.config.hidden_channels,
            num_classes=2,
            heads=self.config.heads,
            dropout=self.config.dropout
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.criterion = nn.NLLLoss(weight=class_weights.to(self.device))
    
    def train_step(self, data: Data) -> float:
        """
        Single training step.
        
        Logic from colab-code/train.py (lines 304-313)
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(data.x, data.edge_index)
        loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, data: Data, mask: torch.Tensor) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Evaluate model on a subset of data.
        
        Logic from colab-code/train.py (lines 317-334)
        """
        self.model.eval()
        
        out = self.model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        
        return acc, f1, y_true, y_pred
    
    def train(
        self,
        data: Data,
        labels: np.ndarray,
        callback: Optional[Callable[[int, Dict[str, float]], None]] = None
    ) -> TrainingHistory:
        """
        Full training loop with early stopping.
        
        Logic from colab-code/train.py (lines 336-367)
        
        Args:
            data: PyTorch Geometric Data object
            labels: Node labels (0=NON-SYBIL, 1=SYBIL)
            callback: Optional callback for progress updates
                     callback(epoch, {'loss': ..., 'val_f1': ..., 'val_acc': ...})
        
        Returns:
            TrainingHistory with metrics
        """
        # Prepare data
        data, class_weights = self.prepare_data(data, labels)
        data = data.to(self.device)
        
        # Initialize model
        self.initialize_model(data.num_features, class_weights)
        
        # Reset history
        self.history = TrainingHistory()
        
        best_val_f1 = 0
        patience_counter = 0
        
        for epoch in range(1, self.config.epochs + 1):
            # Training step
            train_loss = self.train_step(data)
            
            # Validation
            val_acc, val_f1, _, _ = self.evaluate(data, data.val_mask)
            
            # Record history
            self.history.epochs.append(epoch)
            self.history.train_loss.append(train_loss)
            self.history.val_acc.append(val_acc)
            self.history.val_f1.append(val_f1)
            
            # Early stopping logic
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                self.history.best_epoch = epoch
                self.history.best_val_f1 = val_f1
            else:
                patience_counter += 1
            
            # Callback for UI updates
            if callback:
                callback(epoch, {
                    'loss': train_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'best_val_f1': best_val_f1,
                    'patience': patience_counter
                })
            
            # Early stopping
            if patience_counter >= self.config.patience:
                break
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        return self.history
    
    def evaluate_test(self, data: Data) -> EvaluationResult:
        """
        Final evaluation on test set.
        
        Logic from colab-code/train.py (lines 396-407)
        """
        data = data.to(self.device)
        
        acc, f1, y_true, y_pred = self.evaluate(data, data.test_mask)
        
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, y_pred, 
            target_names=['Non-Sybil', 'Sybil']
        )
        
        return EvaluationResult(
            accuracy=acc,
            f1_macro=f1,
            confusion_matrix=cm,
            classification_report=report,
            y_true=y_true,
            y_pred=y_pred
        )
    
    def save_model(self, path: str) -> None:
        """Save model weights to file."""
        if self.model:
            torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str, num_features: int) -> None:
        """Load model weights from file."""
        self.model = SybilGAT(
            num_features=num_features,
            hidden_channels=self.config.hidden_channels,
            num_classes=2,
            heads=self.config.heads,
            dropout=self.config.dropout
        ).to(self.device)
        
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
    
    def get_predictions(self, data: Data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions and probabilities for all nodes.
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            probs = torch.exp(out)
            preds = out.argmax(dim=1)
        
        return preds.cpu().numpy(), probs.cpu().numpy()


def create_training_summary(
    history: TrainingHistory,
    eval_result: EvaluationResult
) -> Dict[str, Any]:
    """Create summary of training results."""
    return {
        'total_epochs': len(history.epochs),
        'best_epoch': history.best_epoch,
        'best_val_f1': history.best_val_f1,
        'final_train_loss': history.train_loss[-1] if history.train_loss else None,
        'test_accuracy': eval_result.accuracy,
        'test_f1_macro': eval_result.f1_macro,
        'confusion_matrix': eval_result.confusion_matrix.tolist(),
    }
