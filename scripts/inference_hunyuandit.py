import os
from datasets import load_dataset
import torch
from transformers import HfArgumentParser
from config import HunyuanDiTConfig
from model import HunyuanDiTPipeline
from model.hunyuan_transformer_2d import HunyuanDiT2DModel
from model.utils import save_images, save_prompts


def main():
    parser = HfArgumentParser(HunyuanDiTConfig)
    config = parser.parse_args_into_dataclasses()[0]

    prompts = load_dataset("shunk031/DrawBench", split="test")["prompts"]  # len: 200

    custom_transformer = HunyuanDiT2DModel.from_pretrained(
        config.model_path, subfolder="transformer"
    ).to("cuda")

    pipe = HunyuanDiTPipeline.from_pretrained(
        config.model_path,
        transformer=custom_transformer,
        use_self_reflection=config.use_self_reflection,
        vl_model_path=config.vl_model_path,
    ).to("cuda")

    generator = torch.Generator(config.device).manual_seed(config.seed)

    total_images = []
    for idx, prompt in enumerate(prompts):
        image_path = os.path.join(config.save_dir, f"image_{idx}.png")
        if os.path.exists(image_path) :
            print(f"Image {idx} already exists, skipping generation.")
            continue
        print(f"Generating image for prompt {idx}: {prompt}")

        images = pipe(
            [prompt],
            generator=generator,
            num_inference_steps=config.num_inference_steps,
            num_images_per_prompt=config.num_images_per_prompt,
            output_type="pil",
            save_dir=config.save_dir,
            process_steps=config.process_steps,
            process_steps_interval=config.process_steps_interval,
            process_start=config.process_start,
            process_end=config.process_end,
        )[0]

        total_images.extend(images)
        save_images(images, config.save_dir, prefix=f"image_{idx}")

    save_prompts(prompts, config.save_dir)


if __name__ == "__main__":
    main()