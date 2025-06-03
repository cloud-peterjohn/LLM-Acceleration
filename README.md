# LLM-Acceleration
Comprehensive Acceleration of Llama-3.2-3B-Instruct with QLoRA Fine-tuning, GPTQ Quantization, VLLM Inference, KV-Cache Quantization, and Speculative Decoding.

## Overview

This project implements a comprehensive optimization pipeline for the Llama-3.2-3B-Instruct model, achieving significant improvements in inference efficiency while maintaining model quality. The pipeline combines multiple acceleration techniques including QLoRA fine-tuning, GPTQ quantization, VLLM inference engine, 8-bit KV-cache, and speculative decoding.

## Pipelines

The optimization pipeline consists of three main stages:

### 1. QLoRA Fine-tuning (`qlora-finetune.ipynb`)
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

### 2. GPTQ Quantization (`gptq-quant.ipynb`)
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

### 3. VLLM Inference with Advanced Optimizations (`vllm-infer.ipynb`)
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

## Model Repositories

The optimized models are available on Hugging Face:
0. **Base Model**: [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
   - Base Llama-3.2-3B-Instruct and Llama-3.2-1B-Instruct model for fine-tuning and quantization


1. **Fine-tuned Base Model**: [zbyzby/Llama3.2-3B-Instruct-QLoRA-finetuned](https://huggingface.co/zbyzby/Llama3.2-3B-Instruct-QLoRA-finetuned)
   - QLoRA fine-tuned Llama-3.2-3B-Instruct model

2. **Quantized Main Model**: [zbyzby/Llama3.2-3B-Instruct-quantized](https://huggingface.co/zbyzby/Llama3.2-3B-Instruct-quantized)
   - GPTQ 4-bit quantized version of the fine-tuned model

3. **Quantized Draft Model**: [zbyzby/Llama-3.2-1B-Instruct-GPTQ-Quant](https://huggingface.co/zbyzby/Llama-3.2-1B-Instruct-GPTQ-Quant)
   - GPTQ 4-bit quantized Llama-3.2-1B-Instruct for speculative decoding

## Results

The optimized pipeline achieves excellent performance metrics on T4 GPU in Colab:

- **Perplexity (PPL)**: 11.22 (on WikiText-2-raw-v1 test set)
- **Throughput**: 80.36 tokens/second

## Usage

### Installation

#### Run in Google Colab
Nothing to install, just open the notebook `vllm-infer.ipynb` and run it. The environment is already set up with all dependencies installed.

#### Run Locally
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

# Ensure you have NVIDIA drivers and CUDA toolkit installed
nvidia-smi
nvcc -V

# If you installed CUDA, but nvcc is not found, you may need to add it to your PATH
ls /usr/local/
export PATH=/usr/local/cuda-12.2/bin:$PATH  # Adjust the version as needed
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH  # Adjust the version as needed
which nvcc

# If you need to delete NVIDIA drivers and CUDA toolkit
sudo apt remove --purge nvidia-cuda-toolkit
sudo rm -rf /usr/local/cuda*

# Install NVIDIA drivers and CUDA toolkit if needed
## CUDA Toolkit (if not installed yet…)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-2
vim ~/.bashrc
export CUDA_HOME=/usr/local/cuda-12.2/
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
source ~/.bashrc
nvcc --version
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
# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install required packages
pip install --upgrade transformers vllm datasets tqdm
# pip install --upgrade transformers vllm==v0.8.5.post1 datasets tqdm
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

## Related Projects
- [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [GPTQ](https://github.com/ModelCloud/GPTQModel.git)
- [VLLM](https://docs.vllm.ai/en/stable/)
- [QLoRA](https://github.com/artidoro/qlora.git)

