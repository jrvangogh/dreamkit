import os
import inspect
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from PIL import Image
import numpy as np
import torch
from torch import autocast
from tqdm.auto import tqdm
from dreamkit.utils import slerp


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
        if use_fp16:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                modelhub_name, revision="fp16", torch_dtype=torch.float16, use_auth_token=True
            )
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(modelhub_name, use_auth_token=True)
        self.device = device
        self.pipe.unet.to(self.device)
        self.pipe.vae.to(self.device)
        self.pipe.text_encoder.to(self.device)

    @torch.no_grad()
    def diffuse(
        self,
        cond_embeddings,  # text conditioning, should be (1, 77, 768)
        cond_latents,     # image conditioning, should be (1, 4, 64, 64)
        num_inference_steps,
        guidance_scale,
        eta,
    ):
        torch_device = cond_latents.get_device()

        # classifier guidance: add the unconditional embedding
        max_length = cond_embeddings.shape[1]  # 77
        uncond_input = self.pipe.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(torch_device))[0]
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.pipe.scheduler, LMSDiscreteScheduler):
            cond_latents = cond_latents * self.pipe.scheduler.sigmas[0]

        # init the scheduler
        accepts_offset = "offset" in set(inspect.signature(self.pipe.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1
        self.pipe.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.pipe.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # diffuse!
        for i, t in tqdm(
                enumerate(self.pipe.scheduler.timesteps), total=num_inference_steps,
                desc='diffuse', position=2, leave=False,
        ):

            # expand the latents for classifier free guidance
            # TODO: gross much???
            latent_model_input = torch.cat([cond_latents] * 2)
            if isinstance(self.pipe.scheduler, LMSDiscreteScheduler):
                sigma = self.pipe.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)

            # predict the noise residual
            noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # cfg
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            # TODO: omfg...
            if isinstance(self.pipe.scheduler, LMSDiscreteScheduler):
                cond_latents = self.pipe.scheduler.step(noise_pred, i, cond_latents, **extra_step_kwargs)["prev_sample"]
            else:
                cond_latents = self.pipe.scheduler.step(noise_pred, t, cond_latents, **extra_step_kwargs)["prev_sample"]

        # scale and decode the image latents with vae
        cond_latents = 1 / 0.18215 * cond_latents
        image = self.pipe.vae.decode(cond_latents)

        # generate output numpy image as uint8
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image[0] * 255).astype(np.uint8)

        return image

    def interpolate(
        self,
        prompts: list[str],   # prompts to dream about
        seeds: list[int],
        name: str,            # name of this project, for the output directory
        num_steps: int = 72,  # number of steps between each pair of sampled points
        jpg_quality: int = 90,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        width: int = 512,
        height: int = 512,
    ):
        assert len(prompts) == len(seeds)
        assert height % 8 == 0 and width % 8 == 0
        assert 0 < jpg_quality <= 100

        # init the output dir
        output_dir = os.path.join(self.root_output_dir, name)
        os.makedirs(output_dir, exist_ok=True)

        # get the conditional text embeddings based on the prompts
        prompt_embeddings = []
        for prompt in prompts:
            text_input = self.pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            with torch.no_grad():
                embed = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]

            prompt_embeddings.append(embed)

        # Take first embed and set it as starting point, leaving rest as list we'll loop over.
        prompt_embedding_a, *prompt_embeddings = prompt_embeddings

        # Take first seed and use it to generate init noise
        init_seed, *seeds = seeds
        init_a = torch.randn(
            (1, self.pipe.unet.in_channels, height // 8, width // 8),
            device=self.device,
            generator=torch.Generator(device='cuda').manual_seed(init_seed)
        )

        frame_index = 0
        for p, prompt_embedding_b in tqdm(
            enumerate(prompt_embeddings), total=len(prompts) - 1, desc='prompts', position=0
        ):

            init_b = torch.randn(
                (1, self.pipe.unet.in_channels, height // 8, width // 8),
                generator=torch.Generator(device='cuda').manual_seed(seeds[p]),
                device=self.device,
            )

            for i, t in tqdm(
                    enumerate(np.linspace(0, 1, num_steps)), total=num_steps,
                    desc='interpolate', position=1, leave=False,
            ):

                cond_embedding = slerp(float(t), prompt_embedding_a, prompt_embedding_b)
                init = slerp(float(t), init_a, init_b)

                with autocast("cuda"):
                    image = self.diffuse(self.pipe, cond_embedding, init, num_inference_steps, guidance_scale, eta)

                im = Image.fromarray(image)
                output_path = os.path.join(output_dir, f'frame{frame_index:06d}.jpg')
                if jpg_quality < 100:
                    im.save(output_path, optimize=True, quality=jpg_quality)
                else:
                    im.save(output_path)
                frame_index += 1

            prompt_embedding_a = prompt_embedding_b
            init_a = init_b
