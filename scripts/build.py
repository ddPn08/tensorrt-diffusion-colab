import os
import torch
import importlib
from diffusers.models import UNet2DConditionModel, AutoencoderKL
from models import UNet, VAE, CLIP
from transformers import CLIPTextModel

demo_diffusion = importlib.import_module("demo-diffusion", "demo_diffusion")


unet_pretrained_model_id = os.environ.get("UNET_PRETRAINED_MODEL_ID", "CompVis/stable-diffusion-v1-4")
vae_pretrained_model_id = os.environ.get("VAE_PRETRAINED_MODEL_ID", "CompVis/stable-diffusion-v1-4")
clip_pretrained_model_id = os.environ.get("CLIP_PRETRAINED_MODEL_ID", "openai/clip-vit-large-patch14")

def load_unet(model_opts, device="cuda", token=""):
    return UNet2DConditionModel.from_pretrained(
        unet_pretrained_model_id, subfolder="unet", use_auth_token=token, **model_opts
    ).to(device)


def load_vae(device="cuda", token=""):
    vae = AutoencoderKL.from_pretrained(
        vae_pretrained_model_id, subfolder="vae", use_auth_token=token
    ).to(device)
    vae.forward = vae.decode
    return vae


def load_clip(device="cuda", token=""):
    return CLIPTextModel.from_pretrained(
        clip_pretrained_model_id, use_auth_token=token
    ).to(device)


class OriginalUnet(UNet):
    def get_model(self):
        model_opts = (
            {"revision": "fp16", "torch_dtype": torch.float16} if self.fp16 else {}
        )
        return load_unet(model_opts, self.device, self.hf_token)


class OriginalVAE(VAE):
    def get_model(self):
        return load_vae(self.device, self.hf_token)


class OriginalClip(CLIP):
    def get_model(self):
        return load_clip(self.device, self.hf_token)


if __name__ == "__main__":    
    args = demo_diffusion.parseArgs()

    max_batch_size = 16

    demo = demo_diffusion.DemoDiffusion(
        denoising_steps=args.denoising_steps,
        denoising_fp16=(args.denoising_prec == "fp16"),
        output_dir=args.output_dir,
        scheduler=args.scheduler,
        hf_token=args.hf_token,
        verbose=args.verbose,
        nvtx_profile=args.nvtx_profile,
        max_batch_size=max_batch_size,
    )

    demo.models = {
        "clip": CLIP(
            hf_token=demo.hf_token,
            device=demo.device,
            verbose=demo.verbose,
            max_batch_size=max_batch_size,
        ),
        demo.unet_model_key: OriginalUnet(
            hf_token=demo.hf_token,
            fp16=demo.denoising_fp16,
            device=demo.device,
            verbose=demo.verbose,
            max_batch_size=max_batch_size,
        ),
        "vae": OriginalVAE(
            hf_token=demo.hf_token,
            device=demo.device,
            verbose=demo.verbose,
            max_batch_size=max_batch_size,
        ),
    }

    demo.loadEngines(
        args.engine_dir,
        args.onnx_dir,
        args.onnx_opset,
        opt_batch_size=1,
        opt_image_height=512,
        opt_image_width=512,
        force_export=args.force_onnx_export,
        force_optimize=args.force_onnx_optimize,
        force_build=args.force_engine_build,
        minimal_optimization=args.onnx_minimal_optimization,
        static_batch=args.build_static_batch,
        static_shape=not args.build_dynamic_shape,
        enable_preview=args.build_preview_features,
    )
