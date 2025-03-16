import os
import random
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from Model.CyclicTransformer import Translator
from typing import Dict, Any, Tuple

class Model:
    def __init__(self, config, subject: str) -> None:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        
        self.config = config

        self.subject = subject
        self.num_epochs = int(config.get('epoch', 100))
        self.learning_rate = float(config.get('learning_rate', 0.001))
        self.adam_epsilon = float(config.get('epsilon', 1e-8))
        self.l2 = float(config.get('l2', 0.0))
        self.save_root = os.path.join(config['classifier_path'], subject)
        self.summary_path = os.path.join(config['summary_path'], subject)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.seed = self._set_seed(2048)

        # Initialize model
        self.model = self._build_model()
        self.model.to(self.device)

        # Create optimizer and criterion
        self.optimizer = self._get_optimizer(self.model.parameters())
        self.scheduler = self._get_scheduler(self.optimizer)
        self.writer = SummaryWriter(self.summary_path)

    def _build_model(self) -> nn.Module:
        return Translator(self.config)

    def _set_seed(self, seed: int) -> int:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        np.random.seed(seed)
        random.seed(seed)
        return seed

    def _initialize_params(self, model: nn.Module) -> None:
        """Initialize model parameters."""
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_optimizer(self, parameters) -> torch.optim.Optimizer:
        """Get optimizer for model training."""
        return torch.optim.Adam(
            parameters, 
            lr=self.learning_rate, 
            eps=self.adam_epsilon, 
            weight_decay=self.l2
        )

    def _get_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        """Get scheduler for adjusting learning rate."""
        if self.config.get('scheduler', 'Lambda') == 'Lambda':
            lambda_1 = lambda epoch: 0.1 ** (epoch // 5)
            return LambdaLR(optimizer, lr_lambda=lambda_1)
        elif self.config['scheduler'] == 'exponential':
            return ExponentialLR(optimizer, gamma=0.75)

    def load_weights(self, path: str) -> None:
        """Load pre-trained weights."""
        self.model.load_state_dict(torch.load(path))

    def save_model(self, subject: str) -> None:
        """Save the model."""
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)
        model_path = os.path.join(self.save_root, f'classifier-{subject}.pth')
        torch.save(self.model.state_dict(), model_path)

    def mloss(self, 
              x1: torch.Tensor, 
              x2: torch.Tensor, 
              y: torch.Tensor,  
              x1_resc: torch.Tensor, 
              x2_resc: torch.Tensor, 
              y_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = y.to(torch.float32)
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)

        resc_loss_x1 = nn.MSELoss()(x1_resc, x1)
        resc_loss_x2 = nn.MSELoss()(x2_resc, x2)
        
        if self.config['ClassificationLoss'] == 'MAE':
            classify_loss = nn.L1Loss()(y_pred, y)
        elif self.config['ClassificationLoss'] == 'BCE':
            classify_loss = nn.BCELoss()(y_pred, y)

        return resc_loss_x1, resc_loss_x2, classify_loss

    def loss_batch(self, 
                   inputs1: torch.Tensor, 
                   inputs2: torch.Tensor, 
                   outputs: torch.Tensor, 
                   labels: torch.Tensor, 
                   optimizer: torch.optim.Optimizer = None) -> torch.Tensor:
        """Compute loss and update model weights on a batch of data."""
        y_pred, _, x3_resc, x1_resc = self.model(inputs1, inputs2, outputs)

        resc_loss_x1, resc_loss_x3, classify_loss = self.mloss(inputs1, outputs, labels, x1_resc, x3_resc, y_pred)
        loss = resc_loss_x1 + resc_loss_x3 + classify_loss

        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return loss

    def train_epoch(self, 
                    train_dataloader: torch.utils.data.DataLoader, 
                    optimizer: torch.optim.Optimizer, 
                    epoch: int) -> float:
        """Train the model for one single epoch."""
        self.model.train(True)
        train_loss = 0.0

        for i, (x1, x2, x3, y) in enumerate(train_dataloader):
            x1, x2, x3, y = x1.to(self.device), x2.to(self.device), x3.to(self.device), y.to(self.device)
            batch_loss = self.loss_batch(x1, x2, x3, y, optimizer=optimizer)
            train_loss += batch_loss.item()

            if self.writer:
                self.writer.add_scalar('batch_loss', batch_loss.item(), epoch * len(train_dataloader) + i + 1)

        epoch_loss = train_loss / len(train_dataloader)
        self.scheduler.step()

        return epoch_loss

    def evaluate(self, eval_dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluate the model, return average loss and accuracy."""
        self.model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for i, (x1, x2, x3, y) in enumerate(eval_dataloader):
                x1, x2, x3, y = x1.to(self.device), x2.to(self.device), x3.to(self.device), y.to(self.device)
                y_pred, _, x3_resc, x1_resc = self.model(x1, x2, x3)

                resc_loss_x1, resc_loss_x3, classify_loss = self.mloss(x1, x3, y, x1_resc, x3_resc, y_pred)
                loss = resc_loss_x1 + resc_loss_x3 + classify_loss
                eval_loss += loss.item()

        avg_loss = eval_loss / len(eval_dataloader)
        return avg_loss

    def fit(self, 
            train_dataloader: torch.utils.data.DataLoader, 
            eval_dataloader: torch.utils.data.DataLoader) -> None:
        """Model training."""
        best_loss = float('inf')

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(train_dataloader, self.optimizer, epoch)
            eval_loss = self.evaluate(eval_dataloader)

            print(f'Epoch: {epoch}')
            print(f'Training Loss: {train_loss} Validation Loss: {eval_loss}')

            if self.writer:
                self.writer.add_scalar('epoch_loss', train_loss, epoch)
                self.writer.add_scalar('eval_loss', eval_loss, epoch)
                self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], epoch)

            if eval_loss < best_loss:
                best_loss = eval_loss
                self.save_model(self.subject)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Model inference."""
        self.model.eval()
        with torch.no_grad():
            return self.model.predict(inputs)