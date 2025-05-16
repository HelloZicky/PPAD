from dataclasses import dataclass, field

@dataclass
class HunyuanDiTConfig:
    save_dir: str = field(
        default="./results/",
    )
    device: str = field(
        default="cuda",
    )
    num_images_per_prompt: int = field(
        default=1,
    )
    num_inference_steps: int = field(
        default=50,
    )
    seed: int = field(
        default=42,
    )

    model_path: str = field(
        default="Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
    )
    vl_model_path: str = field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )

    process_steps: int = field(
        default=1,
    )
    process_steps_interval: int = field(
        default=5,
    )
    process_start: float = field(
        default=0.5,
    )
    process_end: float = field(
        default=0.8,
    )
    use_self_reflection: bool = field(
        default=False,
    )
    eta: float = field(
        default=1.0,
    )
