# Naam

Naam, for pruning and distilling Large Language Models (LLMs) using adaptive knowledge distillation. It implements the methods described in Wu et al. (2024) "Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language Models".

## Quick Start

1. Create a `data` directory and add your training text files:

    mkdir data
    cp your_text_file.txt data/

2. Run the main script:

    python naam.py

## Usage

### Basic Example
```python
    from naam import ModelPruner, KnowledgeDistiller, create_student_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load models
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Create and train student model
    pruner = ModelPruner()
    avg_activations = pruner.collect_activation_stats(teacher_model, dataloader)
    student_model = create_student_model(teacher_model, avg_activations, prune_percent=0.5)

    distiller = KnowledgeDistiller(teacher_model, student_model)
    trained_student = distiller.train(dataloader, epochs=3)
```
## How It Works

1. **Activation Collection**: The system analyzes neuron activations in the teacher model to identify important pathways.

2. **Pruning**: Creates a smaller student model by removing less important connections while maintaining the model's core capabilities.

3. **Distillation**: Uses adaptive KL divergence to transfer knowledge from teacher to student, with:
    - Dynamic importance weighting between high and low probability tokens
    - Progressive layer freezing to stabilize training
    - Basic plateau detection and optimization

## Model Saving

Models are automatically saved using safetensors format in the `models` directory. The best performing model is saved as `best_student_model.safetensors` with accompanying metadata in `metadata.json`.

## Citation

```
@article{wu2024rethinking,
  title={Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language Models},
  author={Wu, Taiqiang and Tao, Chaofan and Wang, Jiahao and Yang, Runming and Zhao, Zhe and Wong, Ngai},
  journal={arXiv preprint arXiv:2404.02657},
  year={2024}
}
```
