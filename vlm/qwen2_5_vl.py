from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

device = "cuda:1"

class Qwen2_5_VLModel:
    def __init__(self, model_path):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2")
        self.processor = AutoProcessor.from_pretrained(
            model_path, use_fast=True)

        torch.cuda.empty_cache()


    def is_match(self, image_path, prompt, keywords=["yes", "Yes", "correct", "match"]):
        question = f"Be strict in your judgment. Please determine if the image matches the prompt: '{prompt}'? Answer yes or no."
        answer = self.ask_vlm(image_path, question).lower()

        return any(keyword in answer for keyword in keywords)

    def analyze_mismatches(self, image_path, original_prompt, return_ids=False):
        question_text = f"""
        Analyze the image and identify all mismatches with the original prompt.

        Original prompt: "{original_prompt}"

        Instructions:
        1. List ALL elements from the prompt that are missing in the image.
        2. List ALL elements from the prompt that appear incorrectly (wrong quantity, appearance, position, etc.).
        3. Be precise and specific in your analysis.

        Format your response as a numbered list of issues ONLY.
        """
        return self.ask_vlm(image_path, question_text, return_ids)

    def get_enhanced_prompt(self, image_path, original_prompt, diagnosis, return_ids=False):
        question_text = f"""
        You are an expert prompt engineer for image generation models.

        Original prompt: "{original_prompt}"

        Issues with the current image:
        {diagnosis}

        Instructions:
        1. Create an improved prompt that will help the image generation model better match the original intention.
        2. Add specific details, emphasis, or clarifications to address the identified issues.
        3. Maintain the core idea and style of the original prompt - do not add unrelated concepts.
        4. The goal is to get an image closer to what was originally intended.
        5. Use techniques like emphasis words, specific quantities, spatial relationships, or other details as needed.

        Return only one well-structured, fluent sentence without any explanations.
        """
        return self.ask_vlm(image_path, question_text, return_ids)

    def get_negative_prompt(self, image_path, original_prompt, diagnosis, return_ids=False):
        question_text = f"""
        Based on the original prompt and analysis of the current image, list elements that should be avoided.

        Original prompt: "{original_prompt}"

        Current image issues:
        {diagnosis}

        Instructions:
        1. List quality issues to avoid
        2. DO NOT include any objects from the prompt.

        Return only comma-separated quality terms.
        """
        return self.ask_vlm(image_path, question_text, return_ids)

    # 4 rounds
    def evaluate_image(self, image_path, original_prompt, return_ids=False):
        diagnosis = self.analyze_mismatches(image_path, original_prompt, return_ids)
        enhanced = self.get_enhanced_prompt(None, original_prompt, diagnosis, return_ids)
        negative = self.get_negative_prompt(None, original_prompt, diagnosis, return_ids)

        return {
            "enhanced_prompt": enhanced,
            "negative_prompt": negative,
            "diagnosis": diagnosis
        }

    def ask_vlm(self, image_path, question_text, return_ids=False):
        """Helper to ask VLM a specific question about the image."""
        if image_path is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": question_text}
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question_text}
                    ],
                }
            ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

        if return_ids:
            return generated_ids_trimmed

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        return output_text[0]
