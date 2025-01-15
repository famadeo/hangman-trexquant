import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from deeplearning.models import AttentionDiffusionRNN
from deeplearning.dataloader import WordDataLoader
from IPython.display import clear_output

# Set up notebook-wide settings
%matplotlib inline
plt.style.use('seaborn')

class AttentionModelTrainer:
    def __init__(self, config_path='./deeplearning/config.yaml'):
        # Load configuration
        with open(config_path, 'r') as stream:
            self.config = yaml.safe_load(stream)
            
        # Set up CUDA
        self.cuda = torch.cuda.is_available() and self.config['cuda']
        if self.cuda:
            torch.cuda.set_device(3)  # Using GPU 3 as in original code
            print("Using CUDA device:", torch.cuda.get_device_name())
        
        # Set up model name and paths
        self.setup_paths()
        
        # Initialize data loaders
        self.train_loader = WordDataLoader('train', self.config)
        self.val_loader = WordDataLoader('test', self.config)
        
        # Initialize model
        self.model = AttentionDiffusionRNN(self.config)
        if self.cuda:
            self.model = self.model.cuda()
            
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.attention_metrics = []
        
    def setup_paths(self):
        """Setup directory structure for model artifacts"""
        self.model_name = f"{self.config['rnn']}_{self.config['num_layers']}_{self.config['hidden_dim']}"
        
        # Create necessary directories
        for path in ['models', 'plots']:
            full_path = os.path.join(self.config[path], self.model_name)
            os.makedirs(full_path, exist_ok=True)
            
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        attention_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch [{epoch}/{self.config["epochs"]}]')
        
        for batch_idx, (inputs, labels, miss_chars, input_lens) in enumerate(pbar):
            # Prepare batch
            if self.config['use_embedding']:
                inputs = torch.from_numpy(inputs).long()
            else:
                inputs = torch.from_numpy(inputs).float()
                
            labels = torch.from_numpy(labels).float()
            miss_chars = torch.from_numpy(miss_chars).float()
            input_lens = torch.from_numpy(input_lens).long()
            
            if self.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
                miss_chars = miss_chars.cuda()
                input_lens = input_lens.cuda()
            
            # Forward pass
            self.model.optimizer.zero_grad()
            outputs = self.model(inputs, input_lens, miss_chars)
            loss, miss_penalty = self.model.calculate_loss(outputs, labels, input_lens, miss_chars, self.cuda)
            
            # Backward pass
            loss.backward()
            self.model.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            attention_loss += miss_penalty.item() if isinstance(miss_penalty, torch.Tensor) else miss_penalty
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{epoch_loss/(batch_idx+1):.4f}',
                'att_loss': f'{attention_loss/(batch_idx+1):.4f}'
            })
            
        return epoch_loss / len(self.train_loader), attention_loss / len(self.train_loader)
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        attention_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels, miss_chars, input_lens in tqdm(self.val_loader, desc='Validating'):
                # Prepare batch
                if self.config['use_embedding']:
                    inputs = torch.from_numpy(inputs).long()
                else:
                    inputs = torch.from_numpy(inputs).float()
                    
                labels = torch.from_numpy(labels).float()
                miss_chars = torch.from_numpy(miss_chars).float()
                input_lens = torch.from_numpy(input_lens).long()
                
                if self.cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    miss_chars = miss_chars.cuda()
                    input_lens = input_lens.cuda()
                
                # Forward pass
                outputs = self.model(inputs, input_lens, miss_chars)
                loss, miss_penalty = self.model.calculate_loss(outputs, labels, input_lens, miss_chars, self.cuda)
                
                val_loss += loss.item()
                attention_loss += miss_penalty.item() if isinstance(miss_penalty, torch.Tensor) else miss_penalty
        
        val_loss = val_loss / len(self.val_loader)
        attention_loss = attention_loss / len(self.val_loader)
        
        print(f'\nValidation - Loss: {val_loss:.4f}, Attention Loss: {attention_loss:.4f}')
        return val_loss, attention_loss
    
    def plot_metrics(self, save=True):
        """Plot training metrics"""
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot attention metrics
        plt.subplot(1, 2, 2)
        plt.plot(self.attention_metrics, label='Attention Loss')
        plt.title('Attention Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        if save:
            plt.savefig(os.path.join(self.config['plots'], self.model_name, f'training_plot.png'))
        plt.show()
    
    def train(self):
        """Main training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Training
            train_loss, train_att_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validation
            if epoch % self.config['test_every_epoch'] == 0:
                val_loss, val_att_loss = self.validate(epoch)
                self.val_losses.append(val_loss)
                self.attention_metrics.append(val_att_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.model.save_model(
                        is_best=True,
                        epoch=epoch,
                        train_loss=train_loss,
                        test_loss=val_loss,
                        rnn_name=self.config['rnn'],
                        layers=self.config['num_layers'],
                        hidden_dim=self.config['hidden_dim']
                    )
                
                # Plot metrics
                if epoch % self.config['plot_every'] == 0:
                    clear_output(wait=True)
                    self.plot_metrics()
            
            # Save periodic checkpoint
            if epoch % self.config['save_every'] == 0:
                self.model.save_model(
                    is_best=False,
                    epoch=epoch,
                    train_loss=train_loss,
                    test_loss=val_loss if len(self.val_losses) > 0 else None,
                    rnn_name=self.config['rnn'],
                    layers=self.config['num_layers'],
                    hidden_dim=self.config['hidden_dim']
                )

# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = AttentionModelTrainer()
    
    # Start training
    trainer.train()