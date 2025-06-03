# 💡 LLM-Acceleration
Comprehensive Acceleration of Llama-3.2-3B-Instruct with QLoRA Fine-tuning, GPTQ Quantization, VLLM Inference, KV-Cache Quantization, and Speculative Decoding.

## 🥰 Overview

This project implements a comprehensive optimization pipeline for the Llama-3.2-3B-Instruct model, achieving significant improvements in inference efficiency while maintaining model quality. The pipeline combines multiple acceleration techniques including QLoRA fine-tuning, GPTQ quantization, VLLM inference engine, 8-bit KV-cache, and speculative decoding.

## 🤩 Pipelines

The optimization pipeline consists of three main stages:

### 1. QLoRA Fine-tuning ([`qlora-finetune.ipynb`](https://github.com/cloud-peterjohn/LLM-Acceleration/blob/main/qlora-finetune.ipynb))
- **Purpose**: Fine-tune the base Llama-3.2-3B-Instruct model using QLoRA (Quantized Low-Rank Adaptation)
- **Quantization**: 4-bit quantization with BitsAndBytesConfig (NF4, double quantization)
- **LoRA Configuration**: 
  - Rank (r): 8
  - Alpha: 32
  - Target modules: q_proj, k_proj, v_proj
  - Dropout: 0.05
- **Dataset**: Training set of WikiText-2-raw-v1
- **Training**: 3 epochs with gradient accumulation and 8-bit AdamW optimizer
- **Key of QLoRA**
    - Implements parameter-efficient fine-tuning using Low-Rank Adaptation
    - Greatly reduces trainable parameters
    - Uses 4-bit base model quantization to reduce memory footprint
    - Gradient checkpointing for memory optimization

After fine-tuning, the model is saved as [Llama3.2-3B-Instruct-QLoRA-finetuned](https://huggingface.co/zbyzby/Llama3.2-3B-Instruct-QLoRA-finetuned) on HuggingFace. 

### 2. GPTQ Quantization ([`gptq-quant.ipynb`](https://github.com/cloud-peterjohn/LLM-Acceleration/blob/main/gptq-quant.ipynb))
- **Purpose**: Apply GPTQ quantization to both fine-tuned 3B model and base 1B model
- **Configuration**: 
  - 4-bit quantization
  - Group size: 128
  - Calibration dataset: WikiText-2-raw-v1 (1024 samples)
- **Models Quantized**:
  - Main model: Fine-tuned Llama-3.2-3B-Instruct
  - Draft model: Llama-3.2-1B-Instruct (for speculative decoding)
- **Key of GPTQ**:
    - Post-training quantization technique that maintains model quality
    - Calibration-based approach using representative dataset samples for accurate quantization
    - Applied to both main model (3B) and draft model (1B)

After quantization, the models are saved as [Llama3.2-3B-Instruct-quantized](https://huggingface.co/zbyzby/Llama3.2-3B-Instruct-quantized) and [Llama-3.2-1B-Instruct-GPTQ-Quant](https://huggingface.co/zbyzby/Llama-3.2-1B-Instruct-GPTQ-Quant).

### 3. VLLM Inference with Advanced Optimizations ([`vllm-infer.ipynb`](https://github.com/cloud-peterjohn/LLM-Acceleration/blob/main/vllm-infer.ipynb))
- **Inference Engine**: VLLM with Xformers (For Turing GPUs, Flash Attention V2 is not supported, so Xformers is used instead)
- **KV-Cache**: 8-bit quantization for memory efficiency
- **Speculative Decoding**: Using 1B model as draft model with 5 speculative tokens (For T4 GPU, the memory is not enough to speculate decoding & CUDA compilation, so you can turn it on with GPUs with larger memory)
- **Additional Optimizations**:
  - Chunked prefill
  - Prefix caching
  - CUDA graph compilation
  - Inductor backend compilation
- **Key of VLLM**:
    - Speculative Decoding: Uses smaller 1B model to predict multiple tokens, verified by 3B model
    - 8-bit KV-Cache: Reduces memory usage for key-value cache storage
    - Chunked Prefill: Optimizes long sequence processing
    - Prefix Caching: Reuses computation for similar prefixes
    - Compilation: CUDA graph and Inductor optimizations for kernel fusion

## 🥳 Model Repositories

The optimized models are available on Hugging Face:
0. **Base Model**: [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
   - Base Llama-3.2-3B-Instruct and Llama-3.2-1B-Instruct model for fine-tuning and quantization

1. **Fine-tuned Base Model**: [zbyzby/Llama3.2-3B-Instruct-QLoRA-finetuned](https://huggingface.co/zbyzby/Llama3.2-3B-Instruct-QLoRA-finetuned)
   - QLoRA fine-tuned Llama-3.2-3B-Instruct model

2. **Quantized Main Model**: [zbyzby/Llama3.2-3B-Instruct-quantized](https://huggingface.co/zbyzby/Llama3.2-3B-Instruct-quantized)
   - GPTQ 4-bit quantized version of the fine-tuned model

3. **Quantized Draft Model**: [zbyzby/Llama-3.2-1B-Instruct-GPTQ-Quant](https://huggingface.co/zbyzby/Llama-3.2-1B-Instruct-GPTQ-Quant)
   - GPTQ 4-bit quantized Llama-3.2-1B-Instruct for speculative decoding

## 😇 Results

The optimized pipeline achieves excellent performance metrics on T4 GPU in Colab:

- **Perplexity (PPL)**: 11.217229843139648 (on WikiText-2-raw-v1 test set)
- **Throughput**: 80.36136051159993 tokens/second

## 😋 Usage

### Installation

#### Run in Google Colab (**Recommanded**)
Nothing to install, just open the notebook [`vllm-infer.ipynb`](https://github.com/cloud-peterjohn/LLM-Acceleration/blob/main/vllm-infer.ipynb) and run it. The environment is already set up with all dependencies installed.

#### Run Locally
If you want to run the notebooks locally, you must have CUDA 12.6+ and NVIDIA drivers installed, along with Python 3.12+ and PyTorch 2.70. The following steps will guide you through setting up the environment and installing the required packages.

First prepare your environment. 
```bash
# Install Conda for virtual environment management
wget -c 'https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh' -P ~/Downloads
bash ~/Downloads/Anaconda3-2023.09-0-Linux-x86_64.sh
# If you use shell or dash, change it to bash
exec bash
# Add conda to PATH
export PATH=~/anaconda3/bin:$PATH
source ~/.bashrc
conda --version # conda 23.7.4
conda update -n base -c defaults conda

# Install Git
sudo apt update
sudo apt install git
git --version

# Ensure you have NVIDIA drivers and CUDA toolkit with version 12.6+ installed
nvidia-smi
nvcc -V # at least 12.6
which nvcc
# If you encounter issues, you can try: https://gist.github.com/MihailCosmin/affa6b1b71b43787e9228c25fe15aeba

# Create a new conda environment
conda create -n llm-acceleration python=3.12 -y
conda activate llm-acceleration # if CondaError: Run 'conda init' before 'conda activate', run `source ~/.bashrc`

# gcc
sudo apt install build-essential g++
sudo apt-get install gcc-12 g++-12
```
Install the required packages in the environment:
```bash
# Uninstall existing packages to avoid conflicts
pip uninstall torch torchvision torchaudio transformers vllm -y
pip cache purge

# Install PyTorch 2.70 with CUDA 12.6 support
conda install gxx_linux-64 -y
conda install -c conda-forge libstdcxx-ng
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install required packages
pip install --upgrade transformers vllm datasets tqdm
pip install -U gptqmodel --no-build-isolation -v
pip install optimum
pip install --force-reinstall triton==3.2.0
pip install bitblas>=0.1.0

python -c "import torch; print(torch.__version__)"
python -c "import vllm"
python -c "from gptqmodel import GPTQModel"
python -c "import triton; print(triton.__version__)"
```
Login to Hugging Face (replace with your token)
```bash
huggingface-cli login --token your_huggingface_token    # Read Authorization
``` 
Then you can run the notebooks in the environment.  

### Running the Pipeline

1. **Fine-tuning**: Execute `qlora-finetune.ipynb` to fine-tune the base model
2. **Quantization**: Run `gptq-quant.ipynb` to quantize both 3B and 1B models
3. **Inference**: Use `vllm-infer.ipynb` for optimized inference with all acceleration techniques

If you only want to test the speedup of inference throughput, you can skip the first two steps and directly run `vllm-infer.ipynb` with pre-quantized models on Huggingface.

## 🥹 Platforms
- **Tesla T4 GPU**: NVIDIA Tesla T4 GPU is limited to 16GB memory and 7.5 computation capability, which is unsuitable for Flash Attention V2, VLLM V1, bf16, and fp8. So we have to use Xformers, VLLM V0, fp16.
- If you want to use Flash Attention V2, you should use Ampere (A100), Ada (RTX 4090), or Hopper (H100) series GPUs.
- If you want to use bf16, your GPU should have Tensor Core and support Brain Float 16 (bf16), such as Hopper, Ampere, Volta, Ada, L40 series GPUs or RTX A5000+. 
- If you want to use fp8, you GPU should have 8.0+ computation capability, such as Hopper, Ampere, Ada Lovelace, Blackwell series GPUs.
- If you want to use Speculative Decoding, you should use GPUs with larger memory than T4, which depends on the `num_speculative_tokens` parameter.

## 🤣 Related Projects
- [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [GPTQ](https://github.com/ModelCloud/GPTQModel.git)
- [VLLM](https://docs.vllm.ai/en/stable/)
- [QLoRA](https://github.com/artidoro/qlora.git)
