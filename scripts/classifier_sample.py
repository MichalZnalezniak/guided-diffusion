"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""
import torch
from PIL import Image
import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from torchvision.utils import make_grid

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
import wandb


def main():
    args = create_argparser().parse_args()
    os.environ["OPENAI_LOGDIR"] = f"results/{args.experiment_name}"
    os.makedirs(f"results/{args.experiment_name}/example_outputs", exist_ok=True)
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    wandb.init(project="ssl_guided_ddgm", name=args.experiment_name, config=args, entity="kdeja")
    # dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(args.model_path))
    if args.use_fp16:
        model.convert_to_fp16()
    model.cuda()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(th.load(args.classifier_path))
    classifier.cuda()
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()
    resnet50_dino = th.hub.load('facebookresearch/dino:main', 'dino_resnet50').eval().cuda()

    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                 std=[0.26862954, 0.26130258, 0.27577711])

    init = Image.open('./example_inputs/tucan.png')
    wandb.log({f"target": wandb.Image(init)})
    init = torchvision.transforms.ToTensor()(init).cuda().unsqueeze(0).mul(2).sub(1)
    init = normalize(init)
    with torch.no_grad():
        init_latent_space_dino = resnet50_dino(init).repeat([args.batch_size,1]) #TODO - Now global variable blee

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def cond_fn_ssl(x, t, y=None):
        with th.enable_grad():
            x = x.detach().requires_grad_()
            n = x.shape[0]
            cur_t = t[0].item()
            my_t = th.ones([n], dtype=th.long).cuda() * cur_t
            out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={})
            x_in = out['pred_xstart']
            if t[0].item() % 50 == 0:
                sample_temp = ((out['pred_xstart'] + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample_temp = sample_temp.contiguous()
                samples_grid = make_grid(sample_temp, normalize=False)
                wandb.log({f"intermediate_gen": wandb.Image(samples_grid)})
                sample_temp = sample_temp.permute(0, 2, 3, 1)
                # for image in sample_temp:
                image = Image.fromarray(sample_temp[0].cpu().numpy())
                image.save(f'./results/{args.experiment_name}/example_outputs/pred_xstart_iter{t[0].item()}.png')
                # wandb.log({f"intermediate_gen": wandb.Image(image)})

            # x_in_grad = th.zeros_like(x_in)
            x_in_normalized = normalize(x_in)
            x_in_latent_space_dino = resnet50_dino(x_in_normalized)
            losses = th.nn.L1Loss()(x_in_latent_space_dino, init_latent_space_dino)  # Calculated the loss
            wandb.log({"distance": losses})
            x_in_grad = th.autograd.grad(losses.sum() * args.classifier_scale, x_in)[
                0]  # Calculated the gradient of L1Loss with respect to `pred_xstart`
            grad = -th.autograd.grad(x_in, x, x_in_grad)[
                0]  # Apply the chain rule to calculate the gradient of L1Loss with respect to 'x'
            return grad

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        classes = th.randint(
            low=99, high=100, size=(args.batch_size,), device=th.device('cuda')
        )
        print(classes)
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn_ssl,
            device=th.device('cuda:0'),
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.contiguous()
        samples_grid = make_grid(sample, normalize=False)
        wandb.log({f"final_gen": wandb.Image(samples_grid)})
        sample = sample.permute(0, 2, 3, 1)
        image = Image.fromarray(sample[0].cpu().numpy())
        image.save(f'./results/{args.experiment_name}/example_outputs/final_output.png')
        # samples_to_generate = samples_to_generate - 1

        gathered_samples = sample
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = classes
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = all_images
    arr = arr[: args.num_samples]
    label_arr = all_labels
    label_arr = label_arr[: args.num_samples]
    out_path = os.path.join(logger.get_dir(), f"samples_{len(arr)}_{arr[0].shape}.npz")
    logger.log(f"saving to {out_path}")
    np.savez(out_path, arr, label_arr)

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        wandb_api_key="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        experiment_name="test",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
