# Pre-training Gemma 3 (270M) from Scratch 🚀

An end-to-end implementation and pre-training of a **Gemma 3 270M** parameter Small Language Model (SLM) using **PyTorch**. This project builds the architecture from the ground up and trains it on the **TinyStories** dataset to generate coherent English text.

## 📌 Project Overview
This repository contains the complete deep learning pipeline to build a Transformer-based language model similar to Google's Gemma 3. The goal is to demystify Large Language Model (LLM) training by implementing advanced architectural components manually and training on a single GPU.

**Key Objectives:**
- Implement state-of-the-art Transformer components (RoPE, MQA, Sliding Window Attention).
- Build a custom tokenizer and data pipeline.
- Pre-train the model efficiently using Mixed Precision and Gradient Accumulation.

## 🏗️ Architecture & Features
The model follows the **Gemma 3** architecture specifications with the following custom implementations:

-   **Tokenization:** Byte Pair Encoding (BPE) using `tiktoken` (GPT-2 vocab, size: 50,257).
-   **Positional Embeddings:** Rotary Positional Embeddings (**RoPE**) for better context handling.
-   **Attention Mechanisms:**
    -   **Multi-Query Attention (MQA):** For efficient memory usage during inference.
    -   **Sliding Window Attention (SWA):** To handle longer sequences effectively with limited compute.
-   **Normalization:** **RMSNorm** (Root Mean Square Layer Normalization) for training stability.
-   **Activation:** GeGLU / GELU (depending on specific config used in notebook).

## 📂 Dataset
The model is pre-trained on the **[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)** dataset.
-   **Source:** Hugging Face Datasets
-   **Content:** Synthetically generated short stories (simple vocabulary, high coherence).
-   **Size:** ~2.12 Million rows (Train split).

## 🛠️ Tech Stack
-   **Language:** Python
-   **Framework:** PyTorch
-   **Libraries:** `datasets`, `tiktoken`, `numpy` (Note: `transformers` and `einops` are not explicitly used in the core from-scratch implementation shown in the video's notebook, but `transformers` can be useful for broader LLM tasks.)
-   **Hardware:** Trained on a single **NVIDIA A100 GPU** (via Google Colab).

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/gemma3-270m-scratch.git
    cd gemma3-270m-scratch
    ```
2.  **Install dependencies:**
    ```bash
    pip install torch torchvision torchaudio # Use appropriate PyTorch installation command for your system/CUDA
    pip install datasets tiktoken numpy
    ```

## 🚀 Usage

This project is primarily designed to be run in the provided Google Colab notebook for an interactive, step-by-step experience, and to leverage the A100 GPU.

**[Open in Google Colab]**

The Colab notebook covers the following steps:

1.  **Data Preparation:** Tokenize the dataset and create input-output pairs for next-token prediction.
    ```python
    # Code for data loading, tokenization, and I/O pair creation is in the notebook sections 1-3.
    ```
2.  **Model Definition & Training:** Run the pre-training loop (supports Mixed Precision & Gradient Accumulation).
    ```python
    # Model architecture definition and training loop are in notebook sections 4-6.
    ```
    *Note: Training on an A100 GPU takes approximately 8 hours for 150,000 iterations on TinyStories.*
3.  **Inference:** Generate text using the trained model.
    ```python
    # Inference code is in the final section of the notebook.
    ```

## 📊 Results
After training for approximately 150,000 iterations, the model successfully learns grammar, sentence structure, and basic storytelling capabilities, producing coherent English text.

**Sample Output:**
"Once upon a time, there was a little girl named Lily. She loved to play with her blue ball..."

## 📜 References
*   **Original Paper:** [Gemma: Open Models Based on Gemini Research and Technology](https://blog.google/technology/ai/gemma-open-models/)
*   **Dataset:** [TinyStories Paper](https://arxiv.org/abs/2305.07759)
*   **Inspiration:** Vizuara's workshop on pre-training Gemma 3.



