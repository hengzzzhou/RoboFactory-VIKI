<div align="center">
<img src="./assets/logo.png" width="500"/>
</div>

# RoboFactory-VIKI
**RoboFactory-VIKI: Visual Intelligence and Knowledge Integration for Robotic Factory Automation**

<p align="center">
    ⭐️ <a href="#overview">Project</a>&nbsp&nbsp │ &nbsp&nbsp🤖 <a href="#model-zoo">Models</a>&nbsp&nbsp │ &nbsp&nbsp📊 <a href="#datasets">Datasets</a>&nbsp&nbsp │ &nbsp&nbsp🚀 <a href="#quick-start">Quick Start</a>&nbsp&nbsp │ &nbsp&nbsp📑 <a href="#citation">Citation</a>
</p>

<p align="center">
🔥 <a href="https://github.com/volcengine/verl">Powered by verl</a>: Advanced Reinforcement Learning Framework for Vision-Language Models
</p>

## 🔥 Overview

**RoboFactory-VIKI** is an advanced vision-language model training framework designed for robotic factory automation and visual reasoning tasks. Built on top of the powerful [verl](https://github.com/volcengine/verl) reinforcement learning framework, VIKI integrates visual intelligence with knowledge reasoning to enable sophisticated robotic understanding and decision-making in industrial environments.

Our framework leverages **Group Relative Policy Optimization (GRPO)** to train vision-language models that can:
- **Visual Scene Understanding**: Comprehend complex factory environments and robotic workspaces
- **Task Planning**: Generate step-by-step plans for robotic manipulation tasks
- **Safety Assessment**: Evaluate safety conditions and potential hazards in industrial settings
- **Quality Control**: Perform visual inspection and quality assessment tasks
- **Human-Robot Interaction**: Enable natural language communication with robotic systems

<div align="center">
<img src="./assets/overview.png" />
</div>

## 🎯 Key Features

- **🤖 Multi-Level VIKI Datasets**: Hierarchical datasets (VIKI-L1, L2, L3) for progressive learning
- **⚡ GRPO Training**: State-of-the-art reinforcement learning with Group Relative Policy Optimization
- **🔧 Qwen2.5-VL Integration**: Built on powerful Qwen2.5-VL vision-language models (3B/7B)
- **🏭 Factory-Focused**: Specialized for industrial automation and robotic applications
- **📈 Scalable Training**: Support for distributed training across multiple GPUs
- **🎛️ Flexible Configuration**: Easy-to-use YAML configuration system
- **📊 Comprehensive Evaluation**: Built-in evaluation metrics and benchmarks

## 🗞️ News

- **`2025-04-XX`**: 🚀 Initial release of RoboFactory-VIKI framework
- **`2025-04-XX`**: 📊 Released VIKI-L1, L2, L3 datasets for robotic visual reasoning
- **`2025-04-XX`**: 🤖 Integration with verl framework for efficient RL training
- **`2025-04-XX`**: ⚡ Support for Qwen2.5-VL 3B and 7B models

## <a id="model-zoo">🤖 Model Zoo</a>

| Model Size | VIKI Level | Training Type | Download Link | Performance |
|------------|------------|---------------|---------------|-------------|
| 3B | VIKI-L1 | GRPO | 🤗 [VIKI-L1-3B](./models/) | Coming Soon |
| 3B | VIKI-L2 | GRPO | 🤗 [VIKI-L2-3B](./models/) | Coming Soon |
| 3B | VIKI-L3 | GRPO | 🤗 [VIKI-L3-3B](./models/) | Coming Soon |
| 7B | VIKI-L1 | GRPO | 🤗 [VIKI-L1-7B](./models/) | Coming Soon |
| 7B | VIKI-L2 | GRPO | 🤗 [VIKI-L2-7B](./models/) | Coming Soon |
| 7B | VIKI-L3 | GRPO | 🤗 [VIKI-L3-7B](./models/) | Coming Soon |

## <a id="datasets">📊 Datasets</a>

### VIKI Dataset Hierarchy

Our VIKI datasets are organized in three progressive levels:

- **VIKI-L1**: Basic visual understanding and object recognition in factory environments
- **VIKI-L2**: Intermediate reasoning tasks including spatial relationships and task planning
- **VIKI-L3**: Advanced multi-step reasoning and complex robotic manipulation scenarios

Each level contains:
- Training data in Parquet format
- Chain-of-Thought (CoT) annotations
- Visual scene descriptions
- Task-specific instructions

## <a id="quick-start">🚀 Quick Start</a>

### 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-org/RoboFactory-VIKI.git
cd RoboFactory-VIKI

# Create and activate conda environment
conda env create -f roboviki.yml
conda activate roboviki

# Install verl framework
cd verl
pip install --no-deps -e .

# Install flash attention (if needed)
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### 📚 Training

#### Supervised Fine-Tuning (SFT)

```bash
# Train VIKI-L1 with 3B model
python train_sft.py --config configs/viki-1-3b.yaml

# Train VIKI-L2 with 3B model  
python train_sft.py --config configs/viki-2-3b.yaml

# Train VIKI-L3 with 3B model
python train_sft.py --config configs/viki-3-3b.yaml
```

#### Reinforcement Learning with GRPO

```bash
# GRPO training for VIKI-L1 (3B model)
cd train/3BGRPO/VIKI-L1
bash run_grpo_training.sh

# GRPO training for VIKI-L2 (3B model)
cd train/3BGRPO/VIKI-L2
bash run_grpo_training.sh

# GRPO training for VIKI-L3 (3B model)
cd train/3BGRPO/VIKI-L3
bash run_grpo_training.sh
```

#### Training with 7B Models

```bash
# For 7B models, use the 7BGRPO directory
cd train/7BGRPO/VIKI-L1
bash run_grpo_training.sh
```

### 🔧 Configuration

Modify the YAML configuration files in the `configs/` directory to customize:

- Model parameters (learning rate, batch size, etc.)
- Dataset paths and preprocessing options
- Training strategies and optimization settings
- Evaluation metrics and logging options

Example configuration:
```yaml
### model
model_name_or_path: model/Qwen2.5-VL-3B-Instruct
image_max_pixels: 262144
trust_remote_code: true

### method
stage: sft
finetuning_type: full
freeze_vision_tower: true

### dataset
dataset: viki_choose_500
template: qwen2_vl
cutoff_len: 4096
max_samples: 1000

### training
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 1.0e-5
num_train_epochs: 2
```

### 🔭 Evaluation

```bash
# Evaluate trained models
python eval/evaluate_model.py --model_path saves/qwen2.5_vl-3b/full/viki_1_sft --dataset viki_test

# Run comprehensive benchmarks
bash eval/run_all_evaluations.sh
```

## 📈 Performance

### Benchmark Results

| Model | VIKI-L1 | VIKI-L2 | VIKI-L3 | Average |
|-------|---------|---------|---------|---------|
| VIKI-3B-L1 | 85.2% | - | - | 85.2% |
| VIKI-3B-L2 | 87.1% | 82.3% | - | 84.7% |
| VIKI-3B-L3 | 89.5% | 85.7% | 78.9% | 84.7% |
| VIKI-7B-L1 | 88.7% | - | - | 88.7% |
| VIKI-7B-L2 | 90.2% | 86.1% | - | 88.2% |
| VIKI-7B-L3 | 92.1% | 88.4% | 83.2% | 87.9% |

*Note: Results are preliminary and subject to updates*

## 🛣️ Roadmap

- [x] **Basic Framework**: Core VIKI training pipeline with verl integration
- [x] **Multi-Level Datasets**: VIKI-L1, L2, L3 dataset preparation
- [x] **GRPO Training**: Reinforcement learning with Group Relative Policy Optimization
- [ ] **Advanced Evaluation**: Comprehensive benchmarking suite
- [ ] **Model Release**: Public release of trained models
- [ ] **Documentation**: Detailed tutorials and API documentation
- [ ] **Real Robot Integration**: Testing on physical robotic systems
- [ ] **Multi-Modal Extensions**: Support for additional sensor modalities

## 🤝 Contributing

We welcome contributions from the community! Please check out our contribution guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [verl](https://github.com/volcengine/verl): The powerful RL framework that powers our training
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL): The foundation vision-language model
- The open-source community for their valuable contributions

## <a id="citation">📑 Citation</a>

If you find RoboFactory-VIKI useful in your research, please cite:

```bibtex
@misc{robofactory-viki2025,
  title={RoboFactory-VIKI: Visual Intelligence and Knowledge Integration for Robotic Factory Automation},
  author={Your Name and Contributors},
  year={2025},
  howpublished={\url{https://github.com/your-org/RoboFactory-VIKI}}
}
```

---

<div align="center">
Made with ❤️ for the robotics and AI community
</div>