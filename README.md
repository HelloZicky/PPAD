# 🚀 MLLM Semantic Corrected Ping-Pong-Ahead Diffusion



------
**PyTorch** implementation of [MLLM Semantic Corrected Ping-Pong-Ahead Diffusion]() (Zheqi Lv et al.)

## 📚 Quick Start

Run the following script to start inference:

```bash
bash run.sh
```

---

## 🧾 Arguments

| Argument                   | Description                                                                  |
| -------------------------- | ---------------------------------------------------------------------------- |
| `--save_dir`               | Directory to save the generated results                                      |
| `--device`                 | Device to use for inference (`cuda`, `cpu`, etc.)                            |
| `--num_images_per_prompt`  | Number of images to generate per prompt                                      |
| `--num_inference_steps`    | Total denoising steps; higher means better quality but slower generation     |
| `--model_path`             | Path to the diffusion model (e.g., HunyuanDiT)                               |
| `--vl_model_path`          | Path to the vision-language model for evaluation (e.g., Qwen2.5-VL)          |
| `--seed`                   | Random seed for reproducibility                                              |
| `--process_steps`          | Number of self-reflection optimization steps                                 |
| `--process_steps_interval` | Interval between each self-reflection step during generation                 |
| `--process_start`          | Fraction of inference steps after which to start self-reflection (e.g., 0.1) |
| `--process_end`            | Fraction of inference steps to stop self-reflection (e.g., 0.9)              |
| `--use_self_reflection`    | Whether to enable self-reflection during image generation                    |

---

## 📁 Output Structure

Generated outputs are saved under the specified `--save_dir` directory. Example structure:

```
results/
├── image_0.png
├── image_1.png
├── ...
└── prompts.txt
```

---


## 📦 Requirements

Make sure the following libraries are installed:

* `diffusers`
* `transformers`
* `torch`

Or install from `requirements.txt`
