import torch

from model import edsr
from model import light_weight
from option import args


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total parameters:     {total:,} ({total / 1e6:.3f} M)")
    print(f"Trainable parameters:  {trainable:,} ({trainable / 1e6:.3f} M)")


# Match train_edsr_x2_ucmerced.py
args.n_resblocks = 16
args.n_feats = 64
args.scale = [4]
args.n_colors = 3
args.rgb_range = 255

# Standard EDSR
edsr_model = edsr.make_model(args)

print("=== EDSR ===")
count_parameters(edsr_model)

# Lightweight EDSR: use the defaults from light_weight_model.py
args.n_resblocks = 8
args.n_feats = 32

lw_model = light_weight.make_model(args)

print("\n=== Lightweight EDSR ===")
count_parameters(lw_model)
