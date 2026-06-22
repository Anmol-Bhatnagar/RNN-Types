# Recurrent Neural Network (RNN) Architectures for Named Entity Recognition (NER)

This repository contains a comprehensive benchmark and implementation of various **Recurrent Neural Network (RNN)** architectures for **Named Entity Recognition (NER)** on financial news texts. It evaluates the performance, training speed, and inference latency trade-offs of five different neural network models built in PyTorch (accelerated with CUDA) and compares them against a Vanilla RNN built from scratch using NumPy.

---

## 📋 Project Overview

Financial news sentences are long, complex, and full of domain-specific terminology, where context is crucial to identify named entities (e.g., distinguishing whether *"Apple"* refers to the fruit or the corporation). 

This project explores how different recurrent architectures handle this challenge. Specifically, we benchmark:
1. **SimpleRNN (Baseline)**: A single-layer vanilla recurrent neural network.
2. **LSTM (Long Short-Term Memory)**: A single-layer LSTM that addresses vanishing gradients via gating mechanisms.
3. **GRU (Gated Recurrent Unit)**: A single-layer GRU representing a more parameter-efficient variant of LSTM.
4. **Bidirectional RNN**: An LSTM wrapped in a bidirectional layer to incorporate both past and future context.
5. **Deep RNN**: A stacked 2-layer LSTM model to learn hierarchical semantic features.

In addition, we provide a complete, from-scratch NumPy walkthrough of a character-level vanilla RNN to demonstrate the underlying backpropagation through time (BPTT) mathematics.

---

## 📂 Repository Structure

The project layout is structured as follows:

| File / Folder | Description |
| :--- | :--- |
| 📁 [src/](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/src) | Main source code directory for model components, training, and evaluation. |
| ├── [data_preprocessing.py](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/src/data_preprocessing.py) | Ingests, cleans, tokenizes, and pads the dataset from Hugging Face. |
| ├── [models.py](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/src/models.py) | Defines the five PyTorch RNN model architectures. |
| ├── [train.py](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/src/train.py) | Training loop logic with CUDA support, early stopping, and history tracking. |
| ├── [evaluate.py](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/src/evaluate.py) | Computes token accuracy, seqeval entity metrics, and synchronized single-sentence latency. |
| ├── [utils.py](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/src/utils.py) | Custom metrics and helper classes (e.g. masked categorical cross-entropy). |
| 📁 [models/](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/models) | Directory storing saved weights (`.pth`, `.keras`), training logs, and metrics. |
| ├── [comparison_metrics.csv](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/models/comparison_metrics.csv) | Saved metrics (F1, Precision, Recall, Latency, Epoch times) for each model. |
| ├── [learning_curves.png](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/models/learning_curves.png) | Loss & token accuracy learning curves over training epochs. |
| ├── [tradeoff_comparison.png](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/models/tradeoff_comparison.png) | Visualization of Entity F1-Score vs. Inference Latency. |
| 📄 [run_experiment.py](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/run_experiment.py) | Main orchestration script to train, evaluate, and plot results for all models. |
| 📄 [generate_notebook.py](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/generate_notebook.py) | Utility script to compile and generate `compare_results.ipynb`. |
| 📓 [compare_results.ipynb](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/compare_results.ipynb) | Jupyter notebook containing interactive code, metrics, and visualization of results. |
| 📓 [simple_rnn_from_scratch.ipynb](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/simple_rnn_from_scratch.ipynb) | Pure NumPy and Pandas walkthrough of a Vanilla RNN, including BPTT derivation. |
| 📄 [requirements.txt](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/requirements.txt) | Python dependencies required to run this project. |

---

## 📊 Dataset & Preprocessing

The experiments utilize the **`gtfintechlab/finer-ord-bio`** dataset from Hugging Face, which contains tokenized financial news sentences annotated with BIO tags for three categories:
* **Person (PER)**
* **Location (LOC)**
* **Organization (ORG)**

### Preprocessing Steps (`src/data_preprocessing.py`)
1. **Vocabulary Construction**: Builds a lowercase vocabulary from training tokens. The top $14,998$ most frequent words are kept, reserving index `0` for `<PAD>` and `1` for `<UNK>` (total vocabulary size of $15,000$).
2. **Tokenization**: Maps tokens to vocabulary indices.
3. **Sequence Padding**:
   - **Inputs** are padded with `0` (`<PAD>`) to a maximum sequence length of $128$.
   - **Tags** are padded with `-1`. This allows us to use `ignore_index=-1` in PyTorch's `nn.CrossEntropyLoss`, ensuring padding tokens do not contribute to gradients.

---

## 🛠️ Model Architectures (`src/models.py`)

All models utilize PyTorch embedding layers with `padding_idx=0` to ignore padding during forward propagation.

* **[SimpleRNNModel](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/src/models.py#L4)**
  * A single-layer vanilla RNN (`nn.RNN`) followed by a linear classification head.
  * *Parameters*: `embedding_dim=64`, `hidden_dim=64`.
* **[LSTMModel](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/src/models.py#L21)**
  * A single-layer LSTM (`nn.LSTM`) utilizing input, forget, output, and cell gates.
* **[GRUModel](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/src/models.py#L37)**
  * A single-layer GRU (`nn.GRU`) mapping inputs using reset and update gates (fewer parameters than LSTM).
* **[BidirectionalModel](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/src/models.py#L53)**
  * A bidirectional LSTM wrapper. The linear classifier processes the concatenated forward and backward hidden states (size `hidden_dim * 2`).
* **[DeepRNNModel](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/src/models.py#L69)**
  * Stacks two LSTM layers (`num_layers=2`) to extract hierarchical features.

---

## 📈 Benchmark Results

The benchmark was performed over **10 epochs** using early stopping (patience = 3, monitoring validation loss) with an Adam optimizer ($lr=0.001$). 

The following table summarizes the metrics saved in [comparison_metrics.csv](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/models/comparison_metrics.csv):

| Model | Entity F1-Score | Precision | Recall | Token Accuracy | Inference Latency | Total Train Time | Avg Time / Epoch |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SimpleRNN** | 0.2335 | 0.3355 | 0.1791 | 93.76% | 0.337 ms | 2.336 s | 0.233 s |
| **LSTM** | 0.0395 | 0.1953 | 0.0219 | 92.98% | 0.372 ms | 2.150 s | 0.215 s |
| **GRU** | 0.2504 | 0.3790 | 0.1870 | 93.74% | 0.352 ms | 2.066 s | 0.206 s |
| **Bidirectional** | 0.2467 | 0.3840 | 0.1817 | 94.03% | 0.442 ms | 2.438 s | 0.243 s |
| **DeepRNN** | 0.1441 | 0.1778 | 0.1212 | 93.03% | 0.523 ms | 2.340 s | 0.233 s |

### Key Takeaways
1. **SimpleRNN**: Serves as a fast, baseline model but struggles with long-term dependencies due to vanishing gradients.
2. **GRU**: Outperforms the baseline and LSTM models significantly on entity F1-score while maintaining low latency and training times.
3. **Bidirectional RNN**: Achieves the highest token accuracy ($94.03\%$) and excellent precision ($38.40\%$) by utilizing forward and backward contextual flows, though it incurs slightly higher latency.
4. **Deep RNN**: Stacking layers increases parameter complexity, resulting in higher inference latency ($0.523\text{ ms}$) without observing direct improvements over the single-layer GRU/Bidirectional models in early epochs.

---

## 🚀 How to Run

### 1. Prerequisites
Create and activate a virtual environment, then install the dependencies listed in [requirements.txt](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/requirements.txt):

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows Powershell)
.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Running the Experiment Benchmarks
Execute the main orchestrator script to download the dataset, train all five architectures on your CUDA device (or CPU), evaluate them, and generate metrics and plots:

```bash
python run_experiment.py
```

### 3. Reviewing Notebooks
* **Comparative Analysis**: Run `python generate_notebook.py` to compile [compare_results.ipynb](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/compare_results.ipynb). Open this notebook in Jupyter to inspect the visual learning curves and trade-offs.
* **RNN From Scratch**: Open [simple_rnn_from_scratch.ipynb](file:///c:/Users/anmol/Desktop/Projects/RNN%20Types/simple_rnn_from_scratch.ipynb) to study the detailed mathematical derivations and train a character-level Vanilla RNN implemented in pure NumPy.
