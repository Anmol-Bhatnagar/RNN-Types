import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Import project modules
from src.data_preprocessing import load_and_preprocess_data
from src.models import (
    SimpleRNNModel,
    LSTMModel,
    GRUModel,
    BidirectionalModel,
    DeepRNNModel,
    BidirectionalGRUModel,
    AttentionLSTMModel
)
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    # 1. Load and preprocess data
    max_len = 128
    vocab_size = 15000
    
    (x_train, y_train), (x_val, y_val), (x_test, y_test), metadata = load_and_preprocess_data(
        max_len=max_len,
        vocab_size=vocab_size
    )
    
    vocab_len = metadata["vocab_size"]
    num_classes = metadata["num_classes"]
    label_list = metadata["label_list"]
    
    print(f"\nVocabulary size: {vocab_len}")
    print(f"Number of classes (tags): {num_classes}")
    print(f"Tag list: {label_list}")
    
    # 2. Define the models to train
    embedding_dim = 64
    rnn_units = 64
    epochs = 10
    batch_size = 64
    
    models_dict = {
        "SimpleRNN": SimpleRNNModel(vocab_len, num_classes, embedding_dim, rnn_units),
        "LSTM": LSTMModel(vocab_len, num_classes, embedding_dim, rnn_units),
        "GRU": GRUModel(vocab_len, num_classes, embedding_dim, rnn_units),
        "Bidirectional": BidirectionalModel(vocab_len, num_classes, embedding_dim, rnn_units),
        "BidirectionalGRU": BidirectionalGRUModel(vocab_len, num_classes, embedding_dim, rnn_units),
        "AttentionLSTM": AttentionLSTMModel(vocab_len, num_classes, embedding_dim, rnn_units),
        "DeepRNN": DeepRNNModel(vocab_len, num_classes, embedding_dim, rnn_units)
    }
    
    histories = {}
    results = {}
    
    # 3. Train and evaluate each model
    for name, model in models_dict.items():
        # Train
        history = train_model(
            model, x_train, y_train, x_val, y_val,
            model_name=name, epochs=epochs, batch_size=batch_size
        )
        histories[name] = history
        
        # Evaluate
        eval_metrics = evaluate_model(model, x_test, y_test, label_list)
        results[name] = eval_metrics
        
    # 4. Save results to DataFrame and CSV
    comparison_data = []
    for name, metrics in results.items():
        comparison_data.append({
            "Model": name,
            "Entity F1-Score": metrics["f1"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "Token Accuracy": metrics["token_accuracy"],
            "Inference Latency (ms)": metrics["latency_ms"],
            "Total Train Time (s)": histories[name]["total_train_time"],
            "Avg Time per Epoch (s)": np.mean(histories[name]["epoch_times"])
        })
        
    df_comparison = pd.DataFrame(comparison_data)
    os.makedirs("models", exist_ok=True)
    csv_path = os.path.join("models", "comparison_metrics.csv")
    df_comparison.to_csv(csv_path, index=False)
    print(f"\nSaved comparison metrics to {csv_path}")
    print(df_comparison.to_string())
    
    # 5. Generate and save plots
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Loss curves (learning curves)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for name, history in histories.items():
        epochs_range = range(1, len(history["loss"]) + 1)
        axes[0].plot(epochs_range, history["loss"], label=f"{name} Train")
        axes[0].plot(epochs_range, history["val_loss"], linestyle="--", label=f"{name} Val")
        
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    
    # Plot Accuracy curves
    for name, history in histories.items():
        epochs_range = range(1, len(history["accuracy"]) + 1)
        axes[1].plot(epochs_range, history["accuracy"], label=f"{name} Train")
        axes[1].plot(epochs_range, history["val_accuracy"], linestyle="--", label=f"{name} Val")
        
    axes[1].set_title("Training & Validation Token Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    
    plt.tight_layout()
    plot_loss_path = os.path.join("models", "learning_curves.png")
    plt.savefig(plot_loss_path)
    plt.close()
    print(f"Saved learning curves plot to {plot_loss_path}")
    
    # Plot 2: Performance vs Latency Trade-off
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # F1-Score comparison
    sns.barplot(x="Model", y="Entity F1-Score", data=df_comparison, ax=axes[0], palette="viridis")
    axes[0].set_title("Entity-Level F1-Score Comparison")
    axes[0].set_ylim(0, 1.0)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.01),
                         ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        
    # Latency vs F1 Scatter plot
    sns.scatterplot(x="Inference Latency (ms)", y="Entity F1-Score", hue="Model", style="Model",
                    data=df_comparison, s=200, ax=axes[1])
    axes[1].set_title("Latency vs Accuracy Trade-off")
    axes[1].set_xlabel("Inference Latency per Sentence (ms)")
    axes[1].set_ylabel("Entity-Level F1-Score")
    
    plt.tight_layout()
    plot_tradeoff_path = os.path.join("models", "tradeoff_comparison.png")
    plt.savefig(plot_tradeoff_path)
    plt.close()
    print(f"Saved trade-off plot to {plot_tradeoff_path}")

if __name__ == "__main__":
    main()
