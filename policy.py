import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

#from act_copy.detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import torch
import IPython
import numpy as np
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']


    def __call__(self, qpos, image, actions=None, is_pad=None, qvel=None, subtask_label=None, epoch = 0):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos, image, env_state, qvel=qvel, actions=actions, is_pad=is_pad, subtask_label=subtask_label
            )

            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

            loss_dict = {}

            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            # split actions
            """pos      = actions[..., 0:3]
            quat     = actions[..., 3:7]
            gripper  = actions[..., 7]

            pos_hat     = a_hat[..., 0:3]
            quat_hat    = a_hat[..., 3:7]
            gripper_hat = a_hat[..., 7]

            # padding mask
            mask = (~is_pad).float()  # [B, T]

            # --- position loss (Huber is best for BC) ---
            pos_l = F.smooth_l1_loss(pos_hat, pos, reduction='none')  # [B, T, 3]
            pos_l = (pos_l.sum(-1) * mask).sum() / mask.sum()

            # --- quaternion loss (angular distance) ---
            quat = F.normalize(quat, dim=-1)
            quat_hat = F.normalize(quat_hat, dim=-1)

            dot = torch.sum(quat * quat_hat, dim=-1).abs()  # [B, T]
            quat_l = (1.0 - dot) * mask
            quat_l = quat_l.sum() / mask.sum()

            # --- gripper loss ---
            grip_l = F.mse_loss(gripper_hat, gripper, reduction='none')  # [B, T]
            grip_l = (grip_l * mask).sum() / mask.sum()

            # --- weighted reconstruction loss ---
            all_l1 = (
                1.0 * pos_l +
                0.5 * quat_l +
                0.1 * grip_l
            )"""

            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]

            # ---- beta scheduling ----
            warmup_epochs = 0

            if epoch < warmup_epochs:
                beta = 0.0
            else:
                beta = min(
                    1.0,
                    1.0 / (1.0 + np.exp(-(epoch - 2500) / 500))
                )

            loss_dict['loss'] = l1 + beta * total_kld[0]
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state, qvel) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

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
