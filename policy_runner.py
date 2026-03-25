import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import argparse

from act_copy.detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed
from act_copy.detr.models.detr_vae import build


def get_default_args():
    """Returns hardcoded default arguments as a Namespace object."""
    args = argparse.Namespace()
    
    # Training parameters
    args.lr = 1e-4
    args.lr_backbone = 1e-5
    args.batch_size = 2
    args.weight_decay = 1e-4
    args.epochs = 300
    args.lr_drop = 200
    args.clip_max_norm = 0.1
    
    # Model parameters - Backbone
    args.backbone = 'resnet18'
    args.dilation = False
    args.position_embedding = 'sine'
    args.camera_names = ["image", "image2"]
    
    # Transformer parameters
    args.enc_layers = 4
    args.enc_decoder_layers = 4
    args.dec_layers = 7
    args.dim_feedforward = 3200
    args.hidden_dim = 512
    args.dropout = 0.1
    args.nheads = 8
    args.num_queries = 64
    args.pre_norm = False
    
    # Segmentation
    args.masks = False
    
    # Additional parameters (for compatibility)
    args.eval = True
    args.onscreen_render = False
    args.ckpt_dir = None
    args.policy_class = 'ACT'
    args.task_name = None
    args.seed = None
    args.num_epochs = None
    args.kl_weight = None
    args.chunk_size = None
    args.temporal_agg = True
    
    return args





class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        # Load default arguments and override with provided values
        args = get_default_args()
        
        for k, v in args_override.items():
            setattr(args, k, v)
        model = build(args)
        self.model = model # CVAE decoder

    def __call__(self, qpos, image, actions=None, is_pad=None, qvel=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, qvel=qvel, actions=actions, is_pad=is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state, qvel) # no action, sample from prior
            return a_hat


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
