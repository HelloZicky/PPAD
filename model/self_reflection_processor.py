import os
from typing import List, Tuple
import uuid
import torch
from diffusers import DPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
)

class SelfReflectionMixin:

    def self_reflection_process(
        self,
        prompts,
        latents,
        batch_size,
        device,
        prompt_embeds,
        prompt_attention_mask,
        prompt_embeds_2,
        prompt_attention_mask_2,
        extra_step_kwargs,
        add_time_ids,
        style,
        image_rotary_emb,
        current_t,
        num_inference_steps,
        *args,
        **kwargs,
    ):
        with torch.no_grad():
            preview_latents = self._preview(
                latents,
                device,
                prompt_embeds,
                prompt_attention_mask,
                prompt_embeds_2,
                prompt_attention_mask_2,
                extra_step_kwargs,
                add_time_ids,
                style,
                image_rotary_emb,
                current_t,
                num_inference_steps
            )
            images = self._decode_latents(preview_latents)

        enhanced_prompts = []
        negative_prompts = []
        full_evaluations = []
        for i in range(batch_size):
            random_uuid = uuid.uuid4().hex
            img_path = os.path.join(self.save_dir, f"temp_{random_uuid}.png")
            images[i].save(img_path)

            if self.qwen2_5_vl.is_match(img_path, prompts[i]):
                self.should_skip = True
                # print("[Early Exit] the image match original prompts. Skipping evaluation.")
                os.remove(img_path)
                return latents, prompts, None

            feedback = self.qwen2_5_vl.evaluate_image(img_path, prompts[i])

            enhanced_prompt = feedback["enhanced_prompt"]
            negative_prompt = feedback["negative_prompt"]
            full_evaluation = feedback["diagnosis"]
            enhanced_prompts.append(enhanced_prompt)
            negative_prompts.append(negative_prompt)
            full_evaluations.append(full_evaluation)

            os.remove(img_path)

        prompt_embeds = self._encode_prompts(enhanced_prompts, negative_prompts, device)

        latents, enhanced_prompts_data = self._process_with_enhanced_prompts(
            latents,
            prompt_embeds,
            batch_size,
            device,
            extra_step_kwargs,
            add_time_ids,
            style,
            image_rotary_emb,
            current_t
        )

        return latents, enhanced_prompts, enhanced_prompts_data

    def _preview(
        self,
        latents,
        device,
        prompt_embeds,
        prompt_attention_mask,
        prompt_embeds_2,
        prompt_attention_mask_2,
        extra_step_kwargs,
        add_time_ids,
        style,
        image_rotary_emb,
        current_t,
        num_inference_steps,
        *args,
        **kwargs,
    ):
        preview_steps = 2
        preview_scheduler = DPMSolverMultistepScheduler.from_config(self.scheduler.config)
        preview_scheduler.set_timesteps(num_inference_steps=preview_steps, device=device)
        valid_timesteps = preview_scheduler.timesteps[preview_scheduler.timesteps <= current_t]
        if len(valid_timesteps) > 0:
            idx = torch.argmin(torch.abs(valid_timesteps - current_t)).item()
            preview_scheduler.timesteps = valid_timesteps[idx:]
        else:
            preview_scheduler.timesteps = torch.tensor([], device=device, dtype=preview_scheduler.timesteps.dtype)


        preview_latents = latents
        for t in preview_scheduler.timesteps:
            latent_model_input = (
                torch.cat([preview_latents] * 2)
                if self.do_classifier_free_guidance
                else preview_latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            t_expand = torch.tensor(
                [t.item()] * latent_model_input.shape[0], device=device
            ).to(latent_model_input.dtype)

            noise_pred = self.transformer(
                latent_model_input,
                t_expand,
                encoder_hidden_states=prompt_embeds,
                text_embedding_mask=prompt_attention_mask,
                encoder_hidden_states_t5=prompt_embeds_2,
                text_embedding_mask_t5=prompt_attention_mask_2,
                image_meta_size=add_time_ids,
                style=style,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0]

            noise_pred, _ = noise_pred.chunk(2, dim=1)

            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                if self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

            preview_latents = preview_scheduler.step(
                noise_pred, t, preview_latents, **extra_step_kwargs, return_dict=False
            )[0]

        return preview_latents

    def _decode_latents(self, latents: torch.Tensor) -> List:
        images = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]
        images, _ = self.run_safety_checker(
            images, self._execution_device, latents.dtype
        )
        return self.image_processor.postprocess(
            images, output_type="pil", do_denormalize=[True] * len(images)
        )

    def _encode_prompts(self, prompts: List[str], negative_prompts: List[str], device: str) -> Tuple:
        embeds = self.encode_prompt(
            prompts,
            device=device,
            dtype=self.transformer.dtype,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompts,
            max_sequence_length=77,
            text_encoder_index=0,
        )
        embeds_2 = self.encode_prompt(
            prompts,
            device=device,
            dtype=self.transformer.dtype,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompts,
            max_sequence_length=256,
            text_encoder_index=1,
        )
        return embeds + embeds_2

    def _process_with_enhanced_prompts(
        self,
        latents: torch.Tensor,
        prompt_embeds: Tuple,
        batch_size,
        device,
        extra_step_kwargs,
        add_time_ids,
        style,
        image_rotary_emb,
        current_t,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        (
            enhanced_prompt_embeds,
            enhanced_negative_prompt_embeds,
            enhanced_prompt_attention_mask,
            enhanced_negative_prompt_attention_mask,
            enhanced_prompt_embeds_2,
            enhanced_negative_prompt_embeds_2,
            enhanced_prompt_attention_mask_2,
            enhanced_negative_prompt_attention_mask_2,
        ) = prompt_embeds

        if self.do_classifier_free_guidance:
            enhanced_prompt_embeds = torch.cat(
                [enhanced_negative_prompt_embeds, enhanced_prompt_embeds]
            )
            enhanced_prompt_embeds_2 = torch.cat(
                [enhanced_negative_prompt_embeds_2, enhanced_prompt_embeds_2]
            )
            enhanced_prompt_attention_mask = torch.cat(
                [
                    enhanced_negative_prompt_attention_mask,
                    enhanced_prompt_attention_mask,
                ]
            )
            enhanced_prompt_attention_mask_2 = torch.cat(
                [
                    enhanced_negative_prompt_attention_mask_2,
                    enhanced_prompt_attention_mask_2,
                ]
            )

        enhanced_prompt_embeds = enhanced_prompt_embeds.to(device)
        enhanced_prompt_attention_mask = enhanced_prompt_attention_mask.to(device)
        enhanced_prompt_embeds_2 = enhanced_prompt_embeds_2.to(device)
        enhanced_prompt_attention_mask_2 = enhanced_prompt_attention_mask_2.to(device)
        add_time_ids = add_time_ids.to(
            dtype=enhanced_prompt_embeds.dtype, device=device
        )
        style = style.to(device=device)

        add_noise_guidance_scale = 0.0
        denoise_guidance_scale = 5.0

        self.add_noise_scheduler.set_timesteps(num_inference_steps=self.scheduler.num_inference_steps, device=device)
        self.denoise_scheduler.set_timesteps(num_inference_steps=self.scheduler.num_inference_steps, device=device)

        add_noise_current_t_idx = torch.argmin(torch.abs(self.add_noise_scheduler.timesteps - current_t)).item()
        denoise_current_t_idx = torch.argmin(torch.abs(self.denoise_scheduler.timesteps - current_t)).item()

        process_step = min(self.process_steps, len(self.add_noise_scheduler.timesteps) - add_noise_current_t_idx)

        add_noise_timesteps = self.add_noise_scheduler.timesteps[add_noise_current_t_idx: add_noise_current_t_idx+process_step]
        denoise_timesteps = self.denoise_scheduler.timesteps[denoise_current_t_idx - process_step + 1 : denoise_current_t_idx + 1]

        for i, t in enumerate(add_noise_timesteps):
            latent_model_input = (
                torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            t_expand = torch.tensor(
                [t.item()] * latent_model_input.shape[0], device=device
            ).to(latent_model_input.dtype)

            noise_pred = self.transformer(
                latent_model_input,
                t_expand,
                encoder_hidden_states=enhanced_prompt_embeds,
                text_embedding_mask=enhanced_prompt_attention_mask,
                encoder_hidden_states_t5=enhanced_prompt_embeds_2,
                text_embedding_mask_t5=enhanced_prompt_attention_mask_2,
                image_meta_size=add_time_ids,
                style=style,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0]

            noise_pred, _ = noise_pred.chunk(2, dim=1)

            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + add_noise_guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                if self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale
                    )

            latents = self.add_noise_scheduler.step(
                noise_pred, t, latents, return_dict=False
            )[0]

        for i, t in enumerate(denoise_timesteps):
            latent_model_input = (
                torch.cat([latents] * 2)
                if self.do_classifier_free_guidance
                else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            t_expand = torch.tensor(
                [t.item()] * latent_model_input.shape[0], device=device
            ).to(latent_model_input.dtype)

            noise_pred = self.transformer(
                latent_model_input,
                t_expand,
                encoder_hidden_states=enhanced_prompt_embeds,
                text_embedding_mask=enhanced_prompt_attention_mask,
                encoder_hidden_states_t5=enhanced_prompt_embeds_2,
                text_embedding_mask_t5=enhanced_prompt_attention_mask_2,
                image_meta_size=add_time_ids,
                style=style,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0]

            noise_pred, _ = noise_pred.chunk(2, dim=1)

            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + denoise_guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                if self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )


            latents = self.denoise_scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs, return_dict=False
            )[0]

        enhanced_prompts_data = (enhanced_prompt_embeds, enhanced_prompt_attention_mask, enhanced_prompt_embeds_2, enhanced_prompt_attention_mask_2)

        return latents, enhanced_prompts_data
