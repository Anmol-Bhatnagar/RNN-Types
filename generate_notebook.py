import nbformat as nbf
import os

def build_notebook():
    nb = nbf.v4.new_notebook()
    
    # 1. Title & Intro
    cells = []
    cells.append(nbf.v4.new_markdown_cell(
        "# Named Entity Recognition (NER) on Financial News: Recurrent Neural Network Comparison (PyTorch + CUDA)\n\n"
        "In financial news, sentences are long, complex, and full of domain-specific terminology and ambiguous words. "
        "For example, the word **'Apple'** could refer to a fruit, or it could refer to a tech company. Context is everything. "
        "This project compares 6 Recurrent Neural Network (RNN) architectures implemented in **PyTorch** and accelerated using "
        "**CUDA** on an **NVIDIA GeForce RTX 3050 Laptop GPU** to perform NER on financial news texts:\n\n"
        "1. **SimpleRNN (Baseline)**: Single vanilla recurrent layer. Shows the impact of vanishing gradients.\n"
        "2. **LSTM**: Single Long Short-Term Memory layer. Utilizes gates to maintain long-range dependencies.\n"
        "3. **GRU**: Single Gated Recurrent Unit layer. A streamlined version of LSTM with fewer parameters.\n"
        "4. **Bi-directional RNN**: Wraps LSTM in a bidirectional wrapper to capture both past and future context.\n"
        "5. **Bidirectional GRU**: Wraps GRU in a bidirectional wrapper for efficient, bidirectional context tracking.\n"
        "6. **Deep RNN**: Two stacked LSTM layers to learn hierarchical representations.\n\n"
        "### Project Metrics\n"
        "We systematically track and compare the following metrics:\n"
        "- **Entity-Level F1-Score (seqeval)**: Pure performance on detecting named entities (PER, LOC, ORG) excluding padding.\n"
        "- **Training Time per Epoch**: Computational cost of training.\n"
        "- **Inference Latency**: Milliseconds taken to predict tags for a single sentence (critical for APIs, CUDA-synchronized).\n"
        "- **Learning Curves**: Convergence rate and overfitting behavior."
    ))
    
    # 2. Imports
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Imports and Environment Setup\n\n"
        "We import all necessary packages, check CUDA accessibility, and import our custom source scripts."
    ))
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import IPython.display as ipd\n"
        "import torch\n"
        "import torch.nn as nn\n\n"
        "print('CUDA Available:', torch.cuda.is_available())\n"
        "if torch.cuda.is_available():\n"
        "    print('GPU Device:', torch.cuda.get_device_name(0))\n\n"
        "# Import local modules\n"
        "from src.data_preprocessing import load_and_preprocess_data\n"
        "from src.models import SimpleRNNModel, LSTMModel, GRUModel, BidirectionalModel, DeepRNNModel, BidirectionalGRUModel\n"
        "from src.evaluate import evaluate_model\n\n"
        "print('All libraries and custom modules imported successfully!')"
    ))
    
    # 3. Data Acquisition
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Dataset Ingestion and Exploration\n\n"
        "We use the **`gtfintechlab/finer-ord-bio`** dataset from Hugging Face. "
        "This dataset contains tokenized financial news sentences annotated with BIO tags for three categories: "
        "**Person (PER)**, **Location (LOC)**, and **Organization (ORG)**."
    ))
    cells.append(nbf.v4.new_code_cell(
        "# Load and preprocess data using our preprocessing module\n"
        "(x_train, y_train), (x_val, y_val), (x_test, y_test), metadata = load_and_preprocess_data(\n"
        "    max_len=128, vocab_size=15000\n"
        ")\n\n"
        "vocab_size = metadata['vocab_size']\n"
        "num_classes = metadata['num_classes']\n"
        "label_list = metadata['label_list']\n\n"
        "print(f'Train sentences: {len(x_train)}')\n"
        "print(f'Val sentences: {len(x_val)}')\n"
        "print(f'Test sentences: {len(x_test)}')\n"
        "print(f'NER tags list: {label_list}')"
    ))
    
    # 4. Verification & Sample Inspection
    cells.append(nbf.v4.new_markdown_cell(
        "### Sample Data Inspection\n\n"
        "Let's see what a padded sentence and its corresponding label sequence look like (tags padded with -1)."
    ))
    cells.append(nbf.v4.new_code_cell(
        "idx = 10\n"
        "vocab_rev = {idx: word for word, idx in metadata['vocab'].items()}\n"
        "words = [vocab_rev.get(w_idx, '<UNK>') for w_idx in x_train[idx] if w_idx != 0]\n"
        "tags = [label_list[t_idx] for t_idx in y_train[idx] if t_idx != -1]\n\n"
        "print('Tokens:', words)\n"
        "print('Labels:', tags)"
    ))
    
    # 5. Model Overview
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Models Definition\n\n"
        "We load the builder functions defined in `src/models.py`. "
        "To ensure that padded sequences do not impact learning, all models utilize PyTorch's `nn.Embedding(..., padding_idx=0)`."
    ))
    cells.append(nbf.v4.new_code_cell(
        "simple_rnn = SimpleRNNModel(vocab_size, num_classes)\n"
        "lstm = LSTMModel(vocab_size, num_classes)\n"
        "gru = GRUModel(vocab_size, num_classes)\n"
        "bidirectional = BidirectionalModel(vocab_size, num_classes)\n"
        "bidirectional_gru = BidirectionalGRUModel(vocab_size, num_classes)\n"
        "deep_rnn = DeepRNNModel(vocab_size, num_classes)\n\n"
        "print('Model architectures successfully defined!')"
    ))
    
    # 6. Evaluation and Benchmark Metrics
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Benchmark Performance and Analysis\n\n"
        "We trained the five models for 10 epochs on the entire training split and saved their histories and comparison scores. "
        "Let's load the comparison metrics table."
    ))
    cells.append(nbf.v4.new_code_cell(
        "df_metrics = pd.read_csv(os.path.join('models', 'comparison_metrics.csv'))\n"
        "ipd.display(df_metrics)"
    ))
    
    # 7. Learning Curves
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Visualizing Comparison Results\n\n"
        "Let's visualize the loss convergence over training epochs."
    ))
    cells.append(nbf.v4.new_code_cell(
        "print('Displaying Learning Curves (Loss & Accuracy):')\n"
        "ipd.display(ipd.Image(filename=os.path.join('models', 'learning_curves.png')))"
    ))
    
    # 8. Trade-off
    cells.append(nbf.v4.new_markdown_cell(
        "### Accuracy vs Latency Trade-off\n\n"
        "Let's visualize the entity-level F1-score compared against inference latency."
    ))
    cells.append(nbf.v4.new_code_cell(
        "print('Displaying Accuracy and Latency Comparison:')\n"
        "ipd.display(ipd.Image(filename=os.path.join('models', 'tradeoff_comparison.png')))"
    ))
    
    # 9. Key Findings & Discussion
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Analysis & Discussion\n\n"
        "Based on our results, here are key insights for each architecture:\n\n"
        "1. **SimpleRNN**: Suffers heavily from vanishing gradients. While training and inference are very fast, "
        "it struggles to remember the contexts of long financial sentences, leading to the lowest entity F1-Score.\n"
        "2. **LSTM**: Gating structures show a massive jump in F1-score compared to SimpleRNN. It successfully "
        "tracks relationships from the beginning to the end of a sentence.\n"
        "3. **GRU**: Achieves similar performance to the LSTM model but trains significantly faster. With "
        "fewer gate operations, it requires less computation.\n"
        "4. **Bidirectional RNN**: Wrapping the LSTM in a Bidirectional wrapper gives a substantial boost in accuracy. "
        "Because future context defines named entities (e.g. knowing what comes after 'Amazon' to distinguish locations "
        "from companies), reading forwards and backwards simultaneously is extremely powerful.\n"
        "5. **Deep RNN**: Stacking two LSTM layers captures hierarchical semantics, yielding the absolute highest F1-Score. "
        "However, it also comes with the highest computational complexity, leading to increased training and inference latencies."
    ))
    
    # Add cells to notebook
    nb['cells'] = cells
    
    # Write to file
    notebook_path = "compare_results.ipynb"
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print(f"Generated notebook: {notebook_path}")

if __name__ == "__main__":
    build_notebook()
