import os
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.utils import get_masked_correct_and_total

def train_model(model, x_train, y_train, x_val, y_val, model_name, epochs=10, batch_size=64):
    """
    Trains a PyTorch sequence labeling model on train data, validates on val data,
    measures timing, implements early stopping, and saves the best model state.
    """
    print(f"\n==========================================")
    print(f"Starting Training for: {model_name} (PyTorch)")
    print(f"==========================================")
    
    # 1. Setup CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)
    
    # 2. Setup DataLoaders
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.long), torch.tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Setup optimizer & loss function (ignoring index -1 corresponding to tag padding)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    # 4. History metrics tracking
    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_times": [],
        "total_train_time": 0.0
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 3
    epochs_no_improve = 0
    
    start_total_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        # --- Training Loop ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            
            # Reshape logits to (batch_size * seq_len, num_classes)
            # Reshape targets to (batch_size * seq_len)
            num_classes = logits.size(-1)
            loss = criterion(logits.view(-1, num_classes), batch_y.view(-1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
            # Calculate training accuracy (excluding pad index -1)
            correct, total = get_masked_correct_and_total(logits, batch_y, ignore_index=-1)
            train_correct += correct
            train_total += total
            
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_correct / (train_total + 1e-8)
        
        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                logits = model(batch_x)
                num_classes = logits.size(-1)
                loss = criterion(logits.view(-1, num_classes), batch_y.view(-1))
                
                val_loss += loss.item() * batch_x.size(0)
                
                correct, total = get_masked_correct_and_total(logits, batch_y, ignore_index=-1)
                val_correct += correct
                val_total += total
                
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_acc = val_correct / (val_total + 1e-8)
        
        duration = time.time() - epoch_start_time
        
        # Save metrics to history
        history["loss"].append(avg_train_loss)
        history["accuracy"].append(avg_train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(avg_val_acc)
        history["epoch_times"].append(duration)
        
        print(f"Epoch {epoch}/{epochs} - {duration:.2f}s - loss: {avg_train_loss:.4f} - accuracy: {avg_train_acc:.4f} - val_loss: {avg_val_loss:.4f} - val_accuracy: {avg_val_acc:.4f}")
        
        # Early stopping and best model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
                
    total_duration = time.time() - start_total_time
    history["total_train_time"] = total_duration
    
    # Save the model state dict
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"{model_name}.pth")
    if best_model_state is not None:
        torch.save(best_model_state, model_path)
        # Restore best weights to model for evaluation
        model.load_state_dict(best_model_state)
    else:
        torch.save(model.state_dict(), model_path)
    print(f"Successfully saved best model weights to {model_path}")
    
    # Save history dict as json
    history_path = os.path.join("models", f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Successfully saved history to {history_path}")
    
    return history
