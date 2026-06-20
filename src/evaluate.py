import time
import numpy as np
import torch
from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import precision_score as seqeval_precision
from seqeval.metrics import recall_score as seqeval_recall
from seqeval.metrics import classification_report as seqeval_report

def evaluate_model(model, x_test, y_test, label_list):
    """
    Evaluates model performance on the test set, computing token-level accuracy,
    entity-level metrics (precision, recall, F1 using seqeval), and average
    single-sentence inference latency with proper CUDA kernel synchronization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")
    model.to(device)
    model.eval()
    
    # 1. Running batch predictions
    print("Running batch inference on test set...")
    pred_ids = []
    with torch.no_grad():
        for i in range(0, len(x_test), 128):
            batch_x = torch.tensor(x_test[i:i+128], dtype=torch.long).to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            pred_ids.append(preds)
    y_pred_ids = np.vstack(pred_ids)
    
    # 2. Map predictions and true labels back to label names, ignoring pad tokens (-1)
    true_tags_list = []
    pred_tags_list = []
    
    for i in range(len(y_test)):
        true_sent = []
        pred_sent = []
        for j in range(len(y_test[i])):
            if y_test[i][j] == -1:
                continue # Skip padding
            
            true_tag_idx = y_test[i][j]
            pred_tag_idx = y_pred_ids[i][j]
            
            true_sent.append(label_list[true_tag_idx])
            pred_sent.append(label_list[pred_tag_idx])
            
        true_tags_list.append(true_sent)
        pred_tags_list.append(pred_sent)
        
    print("Computing entity-level metrics...")
    f1 = seqeval_f1(true_tags_list, pred_tags_list)
    precision = seqeval_precision(true_tags_list, pred_tags_list)
    recall = seqeval_recall(true_tags_list, pred_tags_list)
    report = seqeval_report(true_tags_list, pred_tags_list)
    
    # Calculate token-level accuracy (excluding pads)
    flat_true = []
    flat_pred = []
    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            if y_test[i][j] != -1:
                flat_true.append(y_test[i][j])
                flat_pred.append(y_pred_ids[i][j])
                
    flat_true = np.array(flat_true)
    flat_pred = np.array(flat_pred)
    token_acc = np.mean(flat_true == flat_pred) if len(flat_true) > 0 else 0.0
    
    # 3. Measure inference latency with batch size 1 (real-time prediction latency)
    print("Measuring single-sentence inference latency...")
    num_samples = min(len(x_test), 200)
    
    # Warm up first
    warmup_x = torch.tensor(x_test[0:1], dtype=torch.long).to(device)
    with torch.no_grad():
        _ = model(warmup_x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.time()
    with torch.no_grad():
        for i in range(num_samples):
            tensor_x = torch.tensor(x_test[i:i+1], dtype=torch.long).to(device)
            _ = model(tensor_x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
    total_time = time.time() - start_time
    latency_ms = (total_time / num_samples) * 1000
    
    metrics = {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "token_accuracy": float(token_acc),
        "latency_ms": float(latency_ms),
        "report": report
    }
    
    print(f"Evaluation complete. F1-Score: {f1:.4f} | Token Accuracy: {token_acc:.4f} | Latency: {latency_ms:.2f} ms/sentence")
    return metrics
