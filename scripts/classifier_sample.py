"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""
from PIL import Image
import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
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

from vgg_perceptual_loss import VGGPerceptualLoss

def main():

    args = create_argparser().parse_args()

    
    os.environ["OPENAI_LOGDIR"] = f"results/{args.experiment_name}"
    os.makedirs(f"results/{args.experiment_name}/example_outputs", exist_ok=True)
    os.environ["WANDB_API_KEY"] = 'a82c21cba8df1816e4bfa8d8f52efafcc85b6a12'
    wandb.init(project="guided-diffusion", name=args.experiment_name, config=args, entity="micha-znale-niak")

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
    vgg_loss = VGGPerceptualLoss().cuda()
    resnet50_dino = th.hub.load('facebookresearch/dino:main', 'dino_resnet50').eval().cuda()

    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
    
    # Define the custom x axis metric
    wandb.define_metric("custom_step")

    # Define which metrics to plot against that x-axis
    wandb.define_metric("validation/*", step_metric='custom_step')

    def cond_fn(x, t, y=None):
        assert y is not None
        x = x.detach().requires_grad_()
        map_dif_steps_to_num_steps = {key : item.unsqueeze(0) for key, item in zip(diffusion.timestep_map, th.range(0, diffusion.num_timesteps).long())}
        out = diffusion.p_mean_variance(model, x, map_dif_steps_to_num_steps[t.item()].cuda(), clip_denoised=False, model_kwargs={})
        x_pred = out['pred_xstart']
        sample_temp = ((x_pred + 1) * 127.5).clamp(0, 255).to(th.uint8)
        samples_grid = torchvision.utils.make_grid(sample_temp, normalize=False)
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t.cuda())
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            log_dict = {
                "custom_step": wandb.run.step % 250,
                f"validation/pred_xstart {samples_to_generate}": wandb.Image(samples_grid),
                f"validation/loss {samples_to_generate}": F.softmax(logits, dim=-1)[range(len(logits)), y.view(-1)]
            }
            wandb.log(log_dict)
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def cond_fn_ssl(x, t, y=None):
        init = Image.open('./diffusion_samples_output/example_image.png')
        init = torchvision.transforms.ToTensor()(init).cuda().unsqueeze(0).mul(2).sub(1)
        init = normalize(init)
        with th.enable_grad():
            x = x.detach().requires_grad_()
            n = x.shape[0]
            cur_t = t[0].item()
            my_t = th.ones([n], dtype=th.long).cuda() * cur_t
            out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={})
            x_in = out['pred_xstart']
            if t[0].item() % 50 == 0:
                sample_temp = ((out['pred_xstart'] + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample_temp = sample_temp.permute(0, 2, 3, 1)
                sample_temp = sample_temp.contiguous()
                # for image in sample_temp:
                # image = Image.fromarray(sample_temp[0].cpu().numpy())
                # image.save(f'./pred_xstart_iter{t[0].item()}_{samples_to_generate}.png')

            # x_in_grad = th.zeros_like(x_in)
            x_in_normalized = normalize(x_in)
            x_in_latent_space_dino = resnet50_dino(x_in_normalized)
            init_latent_space_dino = resnet50_dino(init)
            losses = th.nn.L1Loss()(x_in_latent_space_dino, init_latent_space_dino) # Calculated the loss          
            x_in_grad = th.autograd.grad(losses.sum() * args.classifier_scale, x_in)[0] # Calculated the gradient of L1Loss with respect to `pred_xstart`
            grad = -th.autograd.grad(x_in, x, x_in_grad)[0] # Apply the chain rule to calculate the gradient of L1Loss with respect to 'x'
            return grad

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)
    samples_to_generate = 0
    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        classes = th.randint(
            low=99, high=100, size=(args.batch_size,), device=th.device('cuda')
        )
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else  diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn ,
            device=th.device('cuda'),
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        # image = Image.fromarray(sample[0].cpu().numpy())
        # image.save('./final_output_VGG.png')
        samples_to_generate = samples_to_generate - 1

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
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        experiment_name="test",
        wandb_api_key="",

    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
