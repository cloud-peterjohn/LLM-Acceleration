# LLM-Acceleration
Comprehensive Acceleration of Llama-3.2-3B-Instruct with QLoRA Fine-tuning, GPTQ Quantization, VLLM Inference, KV-Cache Quantization, and Speculative Decoding.

## Overview

This project implements a comprehensive optimization pipeline for the Llama-3.2-3B-Instruct model, achieving significant improvements in inference efficiency while maintaining model quality. The pipeline combines multiple acceleration techniques including QLoRA fine-tuning, GPTQ quantization, VLLM inference engine, 8-bit KV-cache, and speculative decoding.

## Pipeline Architecture

The optimization pipeline consists of three main stages:

### 1. QLoRA Fine-tuning (`qlora-finetune.ipynb`)
- **Purpose**: Fine-tune the base Llama-3.2-3B-Instruct model using QLoRA (Quantized Low-Rank Adaptation)
- **Quantization**: 4-bit quantization with BitsAndBytesConfig (NF4, double quantization)
- **LoRA Configuration**: 
  - Rank (r): 8
  - Alpha: 32
  - Target modules: q_proj, k_proj, v_proj
  - Dropout: 0.05
- **Dataset**: WikiText-2-raw-v1 for language modeling
- **Training**: 3 epochs with gradient accumulation and 8-bit AdamW optimizer

### 2. GPTQ Quantization (`gptq-quant.ipynb`)
- **Purpose**: Apply GPTQ quantization to both fine-tuned 3B model and base 1B model
- **Configuration**: 
  - 4-bit quantization
  - Group size: 128
  - Calibration dataset: WikiText-2-raw-v1 (1024 samples)
- **Models Quantized**:
  - Main model: Fine-tuned Llama-3.2-3B-Instruct
  - Draft model: Llama-3.2-1B-Instruct (for speculative decoding)

### 3. VLLM Inference with Advanced Optimizations (`vllm-infer.ipynb`)
- **Inference Engine**: VLLM with multiple optimization techniques
- **KV-Cache**: 8-bit quantization for memory efficiency
- **Speculative Decoding**: Using 1B model as draft model with 5 speculative tokens
- **Additional Optimizations**:
  - Chunked prefill
  - Prefix caching
  - CUDA graph compilation
  - Inductor backend compilation

## Methods

### QLoRA Fine-tuning
- Implements parameter-efficient fine-tuning using Low-Rank Adaptation
- Reduces trainable parameters from 3.2B to ~13M (0.4% of total parameters)
- Uses 4-bit base model quantization to reduce memory footprint
- Gradient checkpointing for memory optimization

### GPTQ Quantization
- Post-training quantization technique that maintains model quality
- 4-bit weight quantization with group-wise quantization (group size 128)
- Calibration-based approach using representative dataset samples
- Applied to both main model (3B) and draft model (1B)

### VLLM Inference Optimizations
- **Speculative Decoding**: Uses smaller 1B model to predict multiple tokens, verified by 3B model
- **8-bit KV-Cache**: Reduces memory usage for key-value cache storage
- **Chunked Prefill**: Optimizes long sequence processing
- **Prefix Caching**: Reuses computation for common prefixes
- **Compilation**: CUDA graph and Inductor optimizations for kernel fusion

## Model Repositories

The optimized models are available on Hugging Face:

1. **Fine-tuned Base Model**: [zbyzby/Llama3.2-3B-Instruct-QLoRA-finetuned](https://huggingface.co/zbyzby/Llama3.2-3B-Instruct-QLoRA-finetuned)
   - QLoRA fine-tuned Llama-3.2-3B-Instruct model

2. **Quantized Main Model**: [zbyzby/Llama3.2-3B-Instruct-quantized](https://huggingface.co/zbyzby/Llama3.2-3B-Instruct-quantized)
   - GPTQ 4-bit quantized version of the fine-tuned model

3. **Quantized Draft Model**: [zbyzby/Llama-3.2-1B-Instruct-GPTQ-Quant](https://huggingface.co/zbyzby/Llama-3.2-1B-Instruct-GPTQ-Quant)
   - GPTQ 4-bit quantized Llama-3.2-1B-Instruct for speculative decoding

## Results

The optimized pipeline achieves excellent performance metrics:

- **Perplexity (PPL)**: 11.22 (on WikiText-2-raw-v1 test set)
- **Throughput**: 80.36 tokens/second

## Usage

### Installation

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

# Install Git
sudo apt update
sudo apt install git
git --version

# Ensure you have NVIDIA drivers and CUDA toolkit installed
nvidia-smi
nvcc -V

# If you need to delete NVIDIA drivers and CUDA toolkit
sudo apt remove --purge nvidia-cuda-toolkit
sudo rm -rf /usr/local/cuda*

# Install NVIDIA drivers and CUDA toolkit
sudo apt install software-properties-common
sudo apt install ubuntu-drivers-common
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
ubuntu-drivers devices  # this may take a while
sudo apt install nvidia-driver-XXX  # replace XXX with the recommended driver version
# If you encounter issues, you can try: https://gist.github.com/MihailCosmin/affa6b1b71b43787e9228c25fe15aeba

# Create a new conda environment
conda create -n llm-acceleration python=3.12 -y
conda activate llm-acceleration
```
Install the required packages in the environment:
```bash
# Uninstall existing packages to avoid conflicts
pip uninstall torch torchvision torchaudio transformers vllm -y
pip cache purge

# Install PyTorch 2.70 with CUDA 12.6 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install required packages
pip install --upgrade transformers vllm datasets tqdm
pip install -U gptqmodel --no-build-isolation -v
pip install optimum
pip install --force-reinstall triton==3.2.0

# Login to Hugging Face (replace with your token)
huggingface-cli login --token your_huggingface_token
```

### Running the Pipeline

1. **Fine-tuning**: Execute `qlora-finetune.ipynb` to fine-tune the base model
2. **Quantization**: Run `gptq-quant.ipynb` to quantize both 3B and 1B models
3. **Inference**: Use `vllm-infer.ipynb` for optimized inference with all acceleration techniques

## Related Projects
- [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [GPTQ](https://github.com/ModelCloud/GPTQModel.git)
- [VLLM](https://docs.vllm.ai/en/stable/)
- [QLoRA](https://github.com/artidoro/qlora.git)
