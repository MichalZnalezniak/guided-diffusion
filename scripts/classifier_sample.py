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
from vgg_perceptual_loss import VGGPerceptualLoss

def main():
    args = create_argparser().parse_args()

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

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x['x']
            clean_image = x['pred_xstart']
            logits = classifier(clean_image, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            print(selected)
            k = th.autograd.grad(selected.sum(), x_in)[0]
            return k * args.classifier_scale

    def cond_fn_vgg(x, t, y=None):
        assert y is not None
        y = th.zeros(x['x'].shape) # GUIDE TO BLACK IMAGE
        y = (2*y) - 1
        y = y.cuda()
        with th.enable_grad():
            x_in = x['x']
            clean_image = x['pred_xstart']
            loss = th.nn.L1Loss(reduction='mean')(clean_image, y)
            print(loss)
            grads = th.autograd.grad(loss, x_in)[0]
            return grads * args.classifier_scale
        
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
            diffusion.p_sample_loop if not args.use_ddim else  diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn_vgg ,
            device=th.device('cuda'),
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        image = Image.fromarray(sample[0].cpu().numpy())
        image.show()

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
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
