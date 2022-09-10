from __future__ import annotations

import os
import inspect
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from PIL import Image
import numpy as np
import torch
from torch import autocast
from tqdm.auto import tqdm
from dreamkit.utils import slerp, preprocess_image


DEFAULT_MODELHUB_NAME = 'CompVis/stable-diffusion-v1-4'
DEFAULT_DEVICE = 'cuda'


class Dreamer:

    def __init__(
        self,
        root_output_dir: str,
        modelhub_name: str = DEFAULT_MODELHUB_NAME,
        use_fp16: bool = True,
        device: str = DEFAULT_DEVICE,
    ):
        self.root_output_dir = root_output_dir
        os.makedirs(self.root_output_dir, exist_ok=True)
        if use_fp16:
            self.pipe = StableDiffusionPipeline.from_pretrained(modelhub_name, revision="fp16", torch_dtype=torch.float16, use_auth_token=True)
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(modelhub_name, use_auth_token=True)
        self.device = device
        self.pipe.unet.to(self.device)
        self.pipe.vae.to(self.device)
        self.pipe.text_encoder.to(self.device)

    def init_scheduler(self, eta: float, num_inference_steps: int):
        accepts_offset = "offset" in set(inspect.signature(self.pipe.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            offset = 1
            extra_set_kwargs["offset"] = 1
        else:
            offset = 0
        self.pipe.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.pipe.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        return offset, extra_step_kwargs

    @torch.no_grad()
    def gen_noise(self, height: int, width: int, seed: int = None, generator=None):
        if (seed is None) == (generator is None):
            raise ValueError('only pass in one of seed or generator')
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        return torch.randn(
            (1, self.pipe.unet.in_channels, height // 8, width // 8),
            device=self.device,
            generator=generator,
        )

    @torch.no_grad()
    def get_text_embedding(self, text: str):
        text_input = self.pipe.tokenizer(
            text,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        embed = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
        return embed

    @torch.no_grad()
    def gen_uncond_embedding(self, max_length: int):
        uncond_input = self.pipe.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
        return self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]

    @torch.no_grad()
    def diffuse_step(self, i, t, guidance_scale, cond_latents, text_embeddings, **step_kwargs):
        # expand the latents for classifier free guidance
        latent_model_input = torch.cat([cond_latents] * 2)
        if isinstance(self.pipe.scheduler, LMSDiscreteScheduler):
            sigma = self.pipe.scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)

        # predict the noise residual
        noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        if isinstance(self.pipe.scheduler, LMSDiscreteScheduler):
            cond_latents = self.pipe.scheduler.step(noise_pred, i, cond_latents, **step_kwargs).prev_sample
        else:
            cond_latents = self.pipe.scheduler.step(noise_pred, t, cond_latents, **step_kwargs).prev_sample
        return cond_latents

    @torch.no_grad()
    def cond_latents_to_image(self, cond_latents):
        # scale and decode the image latents with vae
        cond_latents = 1 / 0.18215 * cond_latents
        img_arr = self.pipe.vae.decode(cond_latents).sample
        # generate output numpy image as uint8
        img_arr = (img_arr / 2 + 0.5).clamp(0, 1)
        img_arr = img_arr.cpu().permute(0, 2, 3, 1).numpy()
        img_arr = (img_arr[0] * 255).astype(np.uint8)
        return Image.fromarray(img_arr)

    @torch.no_grad()
    def diffuse(
        self,
        text_embedding,  # text conditioning, should be (1, 77, 768)
        cond_latents,    # image conditioning, should be (1, 4, 64, 64)
        num_inference_steps,
        guidance_scale,
        eta,
        output_dir: str,
        frame: int,
        jpg_quality: int,
        start_timestep: int = 0,
        save_frequency: int = None,
        tqdm_position: int = 0,
    ):
        # classifier guidance: add the unconditional embedding
        max_length = text_embedding.shape[1]  # 77
        uncond_embeddings = self.gen_uncond_embedding(max_length)
        text_embeddings = torch.cat([uncond_embeddings, text_embedding])

        # if we use LMSDiscreteScheduler, let's make sure latents are multiplied by sigmas
        if isinstance(self.pipe.scheduler, LMSDiscreteScheduler):
            cond_latents = cond_latents * self.pipe.scheduler.sigmas[0]

        # init scheduler and diffuse
        _, extra_step_kwargs = self.init_scheduler(eta, num_inference_steps)
        timesteps = self.pipe.scheduler.timesteps
        for i, t in tqdm(enumerate(timesteps[start_timestep:]), total=len(timesteps)-start_timestep, desc='diffuse', position=tqdm_position, leave=False):
            if save_frequency and i % save_frequency == 0:
                image = self.cond_latents_to_image(cond_latents)
                image.save(
                    os.path.join(output_dir, f'frame{frame:06d}_step{i:06d}.jpg'),
                    optimize=True,
                    jpq_quality=jpg_quality,
                )
            cond_latents = self.diffuse_step(i, t, guidance_scale, cond_latents, text_embeddings, **extra_step_kwargs)

        image = self.cond_latents_to_image(cond_latents)
        image.save(
            os.path.join(output_dir, f'frame{frame:06d}_step{num_inference_steps:06d}.jpg'),
            optimize=True,
            jpq_quality=jpg_quality,
        )
        return image

    def draw(
        self,
        name,  # name of this project, subdirectory in the root directory
        prompt: str,
        seed: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        width: int = 512,
        height: int = 512,
        jpg_quality: int = 95,
        save_frequency: int = None,
        frame: int = 0,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError('height and width must be divisible by 8')
        if not (1 <= jpg_quality <= 95):
            raise ValueError('jpg_quality must be a positive integer in the range [1, 95]')

        # init the output dir
        output_dir = os.path.join(self.root_output_dir, name)
        os.makedirs(output_dir, exist_ok=True)

        text_embedding = self.get_text_embedding(prompt)
        init_latent = self.gen_noise(height, width, seed=seed)

        with autocast('cuda'):
            image = self.diffuse(
                text_embedding=text_embedding,
                cond_latents=init_latent,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                output_dir=output_dir,
                frame=frame,
                jpg_quality=jpg_quality,
                save_frequency=save_frequency,
            )
        return image

    def draw_from_image(
        self,
        name,  # name of this project, subdirectory in the root directory
        prompt: str,
        seed: int,
        init_image: Image,
        init_image_strength: float = 0.2,  # higher values make the output image match the input more closely
        resize_image: bool = False,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        jpg_quality: int = 95,
        save_frequency: int = None,
        frame: int = 0,
    ):
        if not (0 <= init_image_strength <= 1.0):
            raise ValueError('strength must be in range [0.0, 1.0]')
        if not (1 <= jpg_quality <= 95):
            raise ValueError('jpg_quality must be a positive integer in the range [1, 95]')

        # init the output dir
        output_dir = os.path.join(self.root_output_dir, name)
        os.makedirs(output_dir, exist_ok=True)

        text_embedding = self.get_text_embedding(prompt)

        with autocast('cuda'):
            # preprocess init image and create starting latent
            preprocessed_image = preprocess_image(init_image, resize=resize_image)
            base_latent = 0.18215 * self.pipe.vae.encode(preprocessed_image.to(self.device)).latent_dist.sample()

            # get starting point based on strength
            offset, _ = self.init_scheduler(eta, num_inference_steps)
            init_timestep = min(int(num_inference_steps * (1 - init_image_strength)) + offset, num_inference_steps)
            timesteps = torch.tensor(
                [self.pipe.scheduler.timesteps[-init_timestep]],
                dtype=torch.long,
                device=self.device,
            )

            # add noise to starting latent
            noise = torch.randn(
                base_latent.shape,
                generator=torch.Generator(device=self.device).manual_seed(seed),
                device=self.device,
            )
            init_latent = self.pipe.scheduler.add_noise(base_latent, noise, timesteps)

            start_timestep = max(num_inference_steps - init_timestep + offset, 0)
            image = self.diffuse(
                text_embedding=text_embedding,
                cond_latents=init_latent,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                start_timestep=start_timestep,
                output_dir=output_dir,
                frame=frame,
                jpg_quality=jpg_quality,
                save_frequency=save_frequency,
            )
        return image

    def animate_random_walk(
        self,
        name: str,    # name of this project, subdirectory in the root directory
        prompt: str,  # prompts to dream about
        seed: int,
        walk_std: float = 0.1,
        num_walk_steps: int = 75,  # number of steps to walk for
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        width: int = 512,
        height: int = 512,
        jpg_quality: int = 95,
    ):
        """This doesn't work as well as hoped.

        If you want to explore a single prompt, just use animate_interpolate. Pass in the same prompt multiple times,
        but use different seeds.

        Not really any great choice for walk_std. High values give erratic animations. Low values keep things smooth,
        but result in the walk not deviating from the original image much. Could maybe use some combo of taking a step
        and then interpolating between steps, but might as well just interpolate between completely different seeds.
        """
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError('height and width must be divisible by 8')
        if not (1 <= jpg_quality <= 95):
            raise ValueError('jpg_quality must be a positive integer in the range [1, 95]')

        # init the output dir
        output_dir = os.path.join(self.root_output_dir, name)
        os.makedirs(output_dir, exist_ok=True)

        text_embedding = self.get_text_embedding(prompt)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        init_latent = self.gen_noise(height, width, generator=generator)

        for frame_index in tqdm(range(num_walk_steps), desc='random_walk', position=0):
            with autocast('cuda'):
                _ = self.diffuse(
                    text_embedding=text_embedding,
                    cond_latents=init_latent,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    output_dir=output_dir,
                    frame=frame_index,
                    jpg_quality=jpg_quality,
                    tqdm_position=1,
                )
            step_latent = torch.normal(0, std=walk_std, size=init_latent.size(), device=self.device, generator=generator)
            init_latent = (init_latent + step_latent) / (1 + walk_std**2)**0.5  # preserve unit variance

    def animate_interpolate(
        self,
        name: str,            # name of this project, subdirectory in the root directory
        prompts: list[str],   # prompts to dream about
        seeds: list[int],
        num_interpolate_steps: int = 75,  # number of steps between each pair of sampled points
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        width: int = 512,
        height: int = 512,
        jpg_quality: int = 95,
    ):
        if len(prompts) != len(seeds):
            raise ValueError('Must pass in the same number of prompts and seeds')
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError('height and width must be divisible by 8')
        if not (1 <= jpg_quality <= 95):
            raise ValueError('jpg_quality must be a positive integer in the range [1, 95]')

        # init the output dir
        output_dir = os.path.join(self.root_output_dir, name)
        os.makedirs(output_dir, exist_ok=True)

        # get the conditional text embeddings based on the prompts
        text_embedding_a, *text_embeddings = [self.get_text_embedding(p) for p in prompts]

        # Take first seed and use it to generate init noise
        init_seed, *seeds = seeds
        init_latent_a = self.gen_noise(height, width, seed=init_seed)

        frame_index = 0
        for p, text_embedding_b in tqdm(
            enumerate(text_embeddings), total=len(prompts) - 1, desc='prompts', position=0
        ):
            init_latent_b = self.gen_noise(height, width, seed=seeds[p])

            for i, t in tqdm(enumerate(np.linspace(0, 1, num_interpolate_steps)), total=num_interpolate_steps, desc='interpolate', position=1, leave=False):
                text_embedding = slerp(float(t), text_embedding_a, text_embedding_b)
                init_latent = slerp(float(t), init_latent_a, init_latent_b)

                with autocast('cuda'):
                    _ = self.diffuse(
                        text_embedding=text_embedding,
                        cond_latents=init_latent,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        eta=eta,
                        output_dir=output_dir,
                        frame=frame_index,
                        jpg_quality=jpg_quality,
                        tqdm_position=2,
                    )
                frame_index += 1

            text_embedding_a = text_embedding_b
            init_latent_a = init_latent_b
