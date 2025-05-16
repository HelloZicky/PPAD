import os
from typing import List

def save_images(images: List, save_dir: str = "./result/test/", prefix: str = "image") -> None:
        for i, image in enumerate(images):
            image_path = os.path.join(save_dir, f"{prefix}.png")
            image.save(image_path)

def save_prompts(prompts: List[str], save_dir: str = "./result/test/"):
        with open(os.path.join(save_dir, "prompts.txt"), "w") as f:
            f.write("\n".join(prompts))
