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
import torch
import torch.nn as nn
import lpips
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
# Set the seed to reproduce
torch.manual_seed(0)
import clip
import glob
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
import torchvision
from PIL import Image
import wandb

from vgg_perceptual_loss import VGGPerceptualLoss


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

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
    loss_fn_vgg = lpips.LPIPS(net='vgg') # best forward scores
    loss_fn_vgg.cuda()

    logger.log("loading classifier...")
    # classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    # classifier.load_state_dict(th.load(args.classifier_path))
    # classifier.cuda()
    # if args.classifier_use_fp16:
    #     classifier.convert_to_fp16()
    # classifier.eval()
    classifier = torchvision.models.alexnet(pretrained=True)
    classifier.eval()
    classifier.cuda()
    classifier_model = torchvision.models.resnet18(pretrained=False)
    classifier_model.fc = torch.nn.Linear(512,2)
    classifier_model.load_state_dict(torch.load('last_epoch_model.pth'))
    classifier_model = classifier_model.cuda()
    classifier_model.eval()

    target_embeds, weights = [], []

    clip_model = clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to('cuda')
    txt = 'A dog in a park'
    weight = 1.0
    target_embeds.append(clip_model.encode_text(clip.tokenize(txt).to('cuda')).float())
    weights.append(weight)


    target_embeds = torch.cat(target_embeds)
    weights = torch.tensor(weights, device='cuda')
    if weights.sum().abs() < 1e-3:
        raise RuntimeError('The weights must not sum to 0.')
    weights /= weights.sum().abs()

    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
    
    # Define the custom x axis metric
    wandb.define_metric("custom_step")

    # Define which metrics to plot against that x-axis
    wandb.define_metric("validation/*", step_metric='custom_step')


    # weights = FCN_ResNet50_Weights.DEFAULT
    # segm_model = fcn_resnet50(weights=weights)
    # segm_model.eval().cuda()
    # preprocess = weights.transforms()
    # class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}

    make_cutouts  = MakeCutouts(256, 16, 1)


    def cond_fn_segmentation(x, t, y=None):
        init = Image.open('./diffusion_samples_output/example_image.png')
        init = torchvision.transforms.ToTensor()(init).cuda().unsqueeze(0)
        batch_init = preprocess(init)
        prediction_init = segm_model(batch_init)["out"]
        normalized_masks_init = prediction_init.softmax(dim=1)
        mask_init = normalized_masks_init[0, class_to_idx["bird"]]

        with th.enable_grad():
            x = x.detach().requires_grad_()
            map_dif_steps_to_num_steps = {key : item.unsqueeze(0) for key, item in zip(diffusion.timestep_map, th.range(0, diffusion.num_timesteps).long())}
            out = diffusion.p_mean_variance(model, x, map_dif_steps_to_num_steps[t.item()].cuda(), clip_denoised=False, model_kwargs={})
            x_pred = out['pred_xstart']
            sample_temp = ((x_pred + 1) * 127.5).clamp(0, 255).to(th.uint8)
            x_pred = (x_pred + 1) / 2 
            samples_grid = torchvision.utils.make_grid(sample_temp, normalize=False)
            x_in = x_pred
            x_in_grad = torch.zeros_like(x_in)
            # x_in_normalized = normalize(x_in)
            batch_x_in = preprocess(x_in)
            prediction_x_in= segm_model(batch_x_in)["out"]
            normalized_masks_x_in = prediction_x_in.softmax(dim=1)
            mask_x_in = normalized_masks_x_in[0, class_to_idx["bird"]]
            losses = F.l1_loss(mask_init, mask_x_in)
            x_in_grad += torch.autograd.grad(losses.sum() * args.classifier_scale, x_in)[0]
            tv_losses = tv_loss(x_in)
            range_losses = range_loss(out['pred_xstart'])
            x_in_grad += torch.autograd.grad((tv_losses * 7500) + (range_losses * 1500), x_in)[0]
            log_dict = {
                "custom_step": wandb.run.step % 250,
                f"validation/pred_xstart {samples_to_generate}": wandb.Image(samples_grid),
                f"validation/loss {samples_to_generate}": losses.mean(),
                f"validation/mask_pred {samples_to_generate} ": wandb.Image(mask_x_in),
                f"validation/mask_target": wandb.Image(mask_init)

            }
            wandb.log(log_dict)
            return -th.autograd.grad(x_in, x, x_in_grad)[0]

    def cond_fn_clip(x, t, y=None):
        init = Image.open('./diffusion_samples_output/example_image.png')
        init = torchvision.transforms.ToTensor()(init).cuda().unsqueeze(0).mul(2).sub(1)
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            n = x.shape[0]
            map_dif_steps_to_num_steps = {key : item.unsqueeze(0) for key, item in zip(diffusion.timestep_map, th.range(0, diffusion.num_timesteps).long())}
            out = diffusion.p_mean_variance(model, x, map_dif_steps_to_num_steps[t.item()].cuda(), clip_denoised=False)
            fac = diffusion.sqrt_one_minus_alphas_cumprod[map_dif_steps_to_num_steps[t.item()].cuda()]
            x_in = out['pred_xstart']* fac + x * (1 - fac)
            sample_temp = ((out['pred_xstart'] + 1) * 127.5).clamp(0, 255).to(th.uint8)
            samples_grid = torchvision.utils.make_grid(sample_temp, normalize=False)
            x_in_norm = normalize((x_in + 1) / 2)

            x_in_grad = torch.zeros_like(x_in)

            clip_in = x_in_norm

            clip_in = torchvision.transforms.Resize(clip_model.visual.input_resolution)(clip_in)
            image_embeds = clip_model.encode_image(clip_in).float()
            dists = spherical_dist_loss(image_embeds.unsqueeze(1), target_embeds.unsqueeze(0))
            dists = dists.view([1, n, -1])
            losses = dists.mul(weights).sum(2).mean(0) * args.classifier_scale

            # loss_lpips = loss_fn_vgg(init, x_in)
            # x_in_grad += torch.autograd.grad(loss_lpips.sum() * args.classifier_scale, x_in)[0]

            tv_losses = tv_loss(x_in)
            range_losses = range_loss(out['pred_xstart'])            
            loss = (tv_losses.sum() * 50) + (range_losses.sum() * 50)
            all_losses = losses
            log_dict = {
                "custom_step": wandb.run.step % 250,
                f"validation/pred_xstart {samples_to_generate}": wandb.Image(samples_grid),
                f"validation/loss {samples_to_generate}": losses,
            }
            wandb.log(log_dict)
            x_in_grad += th.autograd.grad(all_losses, x_in)[0] # Calculated the gradient of L1Loss with respect to `pred_xstart`
            return -th.autograd.grad(x_in, x, x_in_grad)[0]
        
    def new_cond_fn_xpred(x, t, y=None):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            n = x.shape[0]
            map_dif_steps_to_num_steps = {key : item.unsqueeze(0) for key, item in zip(diffusion.timestep_map, th.range(0, diffusion.num_timesteps).long())}
            out = diffusion.p_mean_variance(model, x, map_dif_steps_to_num_steps[t.item()].cuda(), clip_denoised=False)
            fac = diffusion.sqrt_one_minus_alphas_cumprod[map_dif_steps_to_num_steps[t.item()].cuda()]
            x_in = out['pred_xstart']* fac + x * (1 - fac)
            sample_temp = ((out['pred_xstart'] + 1) * 127.5).clamp(0, 255).to(th.uint8)
            samples_grid = torchvision.utils.make_grid(sample_temp, normalize=False)
            x_in = normalize((x_in + 1) / 2)
            logits1 = classifier(x_in)
            logits2 = classifier_model(x_in)
            log_probs1 = F.log_softmax(logits1, dim=-1)
            log_probs2 = F.log_softmax(logits2, dim=-1)
            selected1 = log_probs1[range(len(logits1)), y.view(-1)]
            selected2 = log_probs2[range(len(logits2)), torch.tensor([1], device='cuda:0')]
            tv_losses = tv_loss(x_in)
            range_losses = range_loss(out['pred_xstart'])            
            loss = (tv_losses.sum() * 50) + (range_losses.sum() * 50)
            all_losses = (selected2.sum() * 50) + (selected1.sum() * 100) - loss
            log_dict = {
                "custom_step": wandb.run.step % 250,
                f"validation/pred_xstart {samples_to_generate}": wandb.Image(samples_grid),
                f"validation/loss {samples_to_generate}": F.softmax(logits1, dim=-1)[range(len(logits1)), y.view(-1)],
                f"validation/loss_artifi {samples_to_generate}": F.softmax(logits2, dim=-1)[range(len(logits2)), 1],


            }
            wandb.log(log_dict)
            x_in_grad = th.autograd.grad(all_losses, x_in)[0] # Calculated the gradient of L1Loss with respect to `pred_xstart`
            return th.autograd.grad(x_in, x, x_in_grad)[0]
    def cond_fn_xpred(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x = x.detach().requires_grad_()
            map_dif_steps_to_num_steps = {key : item.unsqueeze(0) for key, item in zip(diffusion.timestep_map, th.range(0, diffusion.num_timesteps).long())}
            out = diffusion.p_mean_variance(model, x, map_dif_steps_to_num_steps[t.item()].cuda(), clip_denoised=False, model_kwargs={})
            x_pred = out['pred_xstart']
            sample_temp = ((x_pred + 1) * 127.5).clamp(0, 255).to(th.uint8)
            samples_grid = torchvision.utils.make_grid(sample_temp, normalize=False)
            fac = diffusion.sqrt_one_minus_alphas_cumprod[map_dif_steps_to_num_steps[t.item()]]
            x_in = out['pred_xstart'] * fac + x * (1 - fac)
            x_in_norm = normalize((x_in + 1) / 2)
            x_in_grad = torch.zeros_like(x_in)
            batch = make_cutouts(x_in_norm)
            transforms = torchvision.transforms.Compose([
                                    torchvision.transforms.RandomResizedCrop(256),
                                    torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                    torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                    torchvision.transforms.RandomGrayscale(p=0.2)])
            batch = transforms(batch)
            logits2 = classifier(batch)
            log_probs2 = F.log_softmax(logits2, dim=-1)
            selected2 = log_probs2[range(len(logits2)), y.view(-1)]
            classifier_scale = args.classifier_scale
            x_in_grad += torch.autograd.grad(selected2.sum() * classifier_scale,  x_in)[0]
            
            tv_losses = tv_loss(x_in)
            range_losses = range_loss(out['pred_xstart'])
            mean = torch.mean(x_in, dim=(2, 3)).squeeze(0)
            std = torch.std(x_in, dim=(2, 3)).squeeze(0)
            mean_target=torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
            std_target=torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
            mse_mean = torch.nn.functional.mse_loss(mean, mean_target)
            mse_std = torch.nn.functional.mse_loss(std, std_target)
            mean_and_std_loss = mse_mean.sum() + mse_std.sum()
            x_in_grad -= torch.autograd.grad((tv_losses * 75) + (range_losses * 15) + (mean_and_std_loss * 100) ,  x_in)[0]            
            log_dict = {
                "custom_step": wandb.run.step % 250,
                f"validation/pred_xstart {samples_to_generate}": wandb.Image(samples_grid),
                f"validation/loss_classifier {samples_to_generate}": F.softmax(logits2, dim=-1)[range(len(logits2)), y.view(-1)].sum() / 16,
                f"validation/tv_loss {samples_to_generate}": tv_losses,
                f"validation/range_loss {samples_to_generate}": range_losses,
                f"validation/normalization_loss {samples_to_generate}": mean_and_std_loss,
            }
            wandb.log(log_dict)
            return th.autograd.grad(x_in, x, x_in_grad)[0]
            # return th.autograd.grad(x_in, x, x_in_grad)[0]


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

    def cond_fn_lpips(x, t, y=None):
        with th.enable_grad():
            x = x.detach().requires_grad_()
            map_dif_steps_to_num_steps = {key : item.unsqueeze(0) for key, item in zip(diffusion.timestep_map, th.range(0, diffusion.num_timesteps).long())}
            out = diffusion.p_mean_variance(model, x, map_dif_steps_to_num_steps[t.item()].cuda(), clip_denoised=False, model_kwargs={})
            x_pred = out['pred_xstart']
            sample_temp = ((x_pred + 1) * 127.5).clamp(0, 255).to(th.uint8)
            samples_grid = torchvision.utils.make_grid(sample_temp, normalize=False)
            fac = diffusion.sqrt_one_minus_alphas_cumprod[map_dif_steps_to_num_steps[t.item()]]
            x_in = out['pred_xstart'] * fac + x * (1 - fac)
            x_in_grad = torch.zeros_like(x_in)
            loss_lpips = loss_fn_vgg(init, x_in)
            x_in_grad += torch.autograd.grad(loss_lpips.sum() * args.classifier_scale, x_in)[0]
            # tv_losses = tv_loss(x_in)
            # range_losses = range_loss(out['pred_xstart'])
            # x_in_grad += torch.autograd.grad((tv_losses * 750) + (range_losses * 150), x_in)[0]
            log_dict = {
                "custom_step": wandb.run.step % 250,
                # f"validation/pred_xstart {samples_to_generate}": wandb.Image(samples_grid),
                f"validation/loss lpips {samples_to_generate}": loss_lpips.mean(),
                # f"validation/tv_loss {samples_to_generate}": tv_losses.mean(),
                # f"validation/range_loss {samples_to_generate}": range_losses.mean()
            }
            wandb.log(log_dict)
            return -torch.autograd.grad(x_in, x, x_in_grad)[0]
        
    def cond_fn_mse(x, t, y=None):
        init = Image.open('./diffusion_samples_output/example_image.png')
        init = torchvision.transforms.ToTensor()(init).cuda().unsqueeze(0).mul(2).sub(1)
        init = normalize(init)

        with th.enable_grad():
            x = x.detach().requires_grad_()
            map_dif_steps_to_num_steps = {key : item.unsqueeze(0) for key, item in zip(diffusion.timestep_map, th.range(0, diffusion.num_timesteps).long())}
            out = diffusion.p_mean_variance(model, x, map_dif_steps_to_num_steps[t.item()].cuda(), clip_denoised=False, model_kwargs={})
            x_pred = out['pred_xstart']
            sample_temp = ((x_pred + 1) * 127.5).clamp(0, 255).to(th.uint8)
            samples_grid = torchvision.utils.make_grid(sample_temp, normalize=False)
            x_in = x_pred
            x_in_grad = th.zeros_like(x_in)
            x_in_normalized = normalize(x_in)
            losses = th.nn.L1Loss()(x_in_normalized, init) # Calculated the loss          
            x_in_grad = th.autograd.grad(losses.sum() * args.classifier_scale, x_in)[0] # Calculated the gradient of L1Loss with respect to `pred_xstart`
            log_dict = {
                "custom_step": wandb.run.step % 250,
                f"validation/pred_xstart {samples_to_generate}": wandb.Image(samples_grid),
                f"validation/loss {samples_to_generate}": losses.mean()
            }
            wandb.log(log_dict)
            return -th.autograd.grad(x_in, x, x_in_grad)[0]

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)
    samples_to_generate = 0
    logger.log("sampling...")
    all_images = []
    all_labels = []
    classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

    
    for class_ in classes:
        for i, file in enumerate(glob.glob(f"small_stl10/train/{class_}/*.jpg")):
            init = Image.open(file)
            init = init.resize((256, 256), Image.Resampling.LANCZOS)
            init = torchvision.transforms.ToTensor()(init).cuda().unsqueeze(0).mul(2).sub(1)
            if i >= 100:
                break
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
                cond_fn=cond_fn_lpips ,
                device=th.device('cuda'),   
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            # sample = sample.contiguous()
            os.makedirs(f"small_stl10/valid/{class_}", exist_ok=True)
            name_of_file = file.split('\\')[-1]
            image = Image.fromarray(sample[0].cpu().numpy())
            image.save( f"./small_stl10/valid/{class_}/{name_of_file}")
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

def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow      

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        cutouts.append(input)
        return torch.cat(cutouts)

if __name__ == "__main__":
    main()
