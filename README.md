# GPT-2 from Scratch on FineWeb-EDU (MPS Optimized)

This project trains a GPT-2 language model from scratch using the [FineWeb-EDU dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), optimized for Apple Silicon (`MPS` backend). It includes:

- ⚙️ A custom training loop with gradient accumulation  
- 💾 Tokenized dataset preprocessing with `.npy` shards  
- 📊 Periodic evaluation on HellaSwag  
- 🧪 Optional text generation during training  

---

## 🛠️ Requirements

Set up a Python environment and install the required libraries:

<pre><code>python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
</code></pre>

**Contents of `requirements.txt`:**

<pre><code>torch==2.2.0
datasets
tqdm
numpy
tiktoken
transformers
</code></pre>

---

## 📦 Dataset Preparation (`fineweb.py`)

Before training, you must download and preprocess the FineWeb-EDU dataset:

<pre><code>python fineweb.py
</code></pre>

This script:

- Downloads the `"sample-10BT"` subset of FineWeb-EDU  
- Tokenizes each document using the GPT-2 tokenizer (`tiktoken`)  
- Saves tokenized documents into `.npy` shards (100M tokens each)  
- The **first shard** is used for validation  
- The **rest** are used for training  

Token files are stored in the `./edu_fineweb10B/` directory.

🔐 You may need to authenticate with Hugging Face:

<pre><code>huggingface-cli login
</code></pre>

---

## 🚀 Training (`train_gpt2.py`)

Once data is prepared, you can launch training:

<pre><code>python train_gpt2.py
</code></pre>

The training script:

- Supports `cpu`, `cuda`, and `mps` (Apple Silicon)  
- Logs training loss, validation loss, and HellaSwag accuracy  
- Saves model checkpoints and generates text samples  

**Example output (MPS backend):**

<pre><code>using device: mps
total desired batch size: 65536
=&gt; calculated gradient accumulation steps: 64
validation loss: 10.8766
HellaSwag accuracy: 2592/10042=0.2581
step     0 | loss: 10.873720 | lr 8.3916e-07 | ...
</code></pre>

Logs and checkpoints are saved to the `./log/` directory.

---

## ⚙️ Configuration

You can modify model size and behavior in `train_gpt2.py`:

<pre><code>GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=4,
    n_head=4,
    n_embd=256
)
</code></pre>

⚠️ For MPS compatibility, smaller models and batch sizes are recommended.

---

## 🧪 Evaluation

- Validation loss is computed every **100 steps**  
- HellaSwag multiple-choice accuracy is evaluated periodically  
- Text generation is triggered from a fixed prompt every **100 steps**

---

## 🙏 Acknowledgments

- [FineWeb-EDU Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)  
- [tiktoken](https://github.com/openai/tiktoken) — GPT-2 tokenizer  
- Inspired by [minGPT](https://github.com/karpathy/minGPT)  
- [YouTube explanation](https://youtu.be/l8pRSuU81PU?si=vR7C_bdsW6V_g9X-)
