# https://github.com/openai/guided-diffusion/tree/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924
# import random

import matplotlib.pyplot as plt
import numpy as np

from helpers import *
from get_cam_score import make_cam_score


def get_beta_schedule(num_diffusion_steps, name="cosine"):
    betas = []
    if name == "cosine":
        max_beta = 0.999
        f = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        for i in range(num_diffusion_steps):
            t1 = i / num_diffusion_steps
            t2 = (i + 1) / num_diffusion_steps
            betas.append(min(1 - f(t2) / f(t1), max_beta))
        betas = np.array(betas)
    elif name == "linear":
        scale = 1000 / num_diffusion_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(beta_start, beta_end, num_diffusion_steps, dtype=np.float64)
    else:
        raise NotImplementedError(f"unknown beta schedule: {name}")
    return betas
      

def extract(arr, timesteps, broadcast_shape, device):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape).to(device)


def mean_flat(tensor):
    return torch.mean(tensor, dim=list(range(1, len(tensor.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL Divergence between two gaussians

    :param mean1:
    :param logvar1:
    :param mean2:
    :param logvar2:
    :return: KL Divergence between N(mean1,logvar1^2) & N(mean2,logvar2^2))
    """
    return 0.5 * (-1 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretised_gaussian_log_likelihood(x, means, log_scales):
    """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image.
        :param x: the target images. It is assumed that this was uint8 values,
                  rescaled to the range [-1, 1].
        :param means: the Gaussian mean Tensor.
        :param log_scales: the Gaussian log stddev Tensor.
        :return: a tensor like x of log probabilities (in nats).
        """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)

    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)

    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))

    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
            )
    assert log_probs.shape == x.shape
    return log_probs


def using_control_matrix(x, t):
    args_control_matrix = {
        "model": "densenet121_2048",
        "pt_path": "/home/seongwon/PycharmProjects/Test_bench/snuhe_dense2048_n_2.pt"
        }
    control_matrix = lambda x, t: np.ceil(make_cam_score(args_control_matrix, "gradcam", x) * t)
    noise = torch.randn_like(x)


class GaussianDiffusionModel:
    def __init__(
            self,
            img_size,
            betas,
            img_channels=1,
            loss_type="l2",  # l2,l1 hybrid
            loss_weight='none',  # prop t / uniform / None
            noise="gauss",
            ):
        super().__init__()
        if noise == "gauss":
            self.noise_fn = lambda x, t: torch.randn_like(x)

        self.img_size = img_size
        self.img_channels = img_channels
        self.loss_type = loss_type
        self.num_timesteps = len(betas)

        if loss_weight == 'prop-t':
            self.weights = np.arange(self.num_timesteps, 0, -1)
        elif loss_weight == "uniform":
            self.weights = np.ones(self.num_timesteps)

        self.loss_weight = loss_weight
        alphas = 1 - betas
        self.betas = betas
        self.sqrt_alphas = np.sqrt(alphas)
        self.sqrt_betas = np.sqrt(betas)

        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        # self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:],0.0)


        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
                np.append(self.posterior_variance[1], self.posterior_variance[1:])
                )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )
    

    def sample_t_with_weights(self, b_size, device):
        p = self.weights / np.sum(self.weights)
        indices_np = np.random.choice(len(p), size=b_size, p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / len(p) * p[indices_np]
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights

    def predict_x_0_from_eps(self, x_t, t, eps):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device) * eps)

    def predict_eps_from_x_0(self, x_t, t, pred_x_0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
                - pred_x_0) \
               / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device)

    def q_mean_variance(self, x_0, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
                extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_0.shape, x_0.device)
        log_variance = extract(
                self.log_one_minus_alphas_cumprod, t, x_0.shape, x_0.device
                )
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """

        # mu (x_t,x_0) = \frac{\sqrt{alphacumprod prev} betas}{1-alphacumprod} *x_0
        # + \frac{\sqrt{alphas}(1-alphacumprod prev)}{ 1- alphacumprod} * x_t
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape, x_t.device) * x_0
                          + extract(self.posterior_mean_coef2, t, x_t.shape, x_t.device) * x_t)

        # var = \frac{1-alphacumprod prev}{1-alphacumprod} * betas
        posterior_var = extract(self.posterior_variance, t, x_t.shape, x_t.device)
        posterior_log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape, x_t.device)
        return posterior_mean, posterior_var, posterior_log_var_clipped

    def p_mean_variance(self, model, x_t, t, estimate_noise=None):
        """
        Finds the mean & variance from N(x_{t-1}; mu_theta(x_t,t), sigma_theta (x_t,t))

        :param model:
        :param x_t:
        :param t:
        :return:
        """
        if estimate_noise == None:
            estimate_noise = model(x_t, t)

        # fixed model variance defined as \hat{\beta}_t - could add learned parameter
        model_var = np.append(self.posterior_variance[1], self.betas[1:])
        model_logvar = np.log(model_var)
        model_var = extract(model_var, t, x_t.shape, x_t.device)
        model_logvar = extract(model_logvar, t, x_t.shape, x_t.device)

        pred_x_0 = self.predict_x_0_from_eps(x_t, t, estimate_noise).clamp(-1, 1)
        model_mean, _, _ = self.q_posterior_mean_variance(
                pred_x_0, x_t, t
                )
        return {
            "mean":         model_mean,
            "variance":     model_var,
            "log_variance": model_logvar,
            "pred_x_0":     pred_x_0,
            }

    def sample_p(self, model, x_t, t, denoise_fn="gauss"):
        out = self.p_mean_variance(model, x_t, t)
        if type(denoise_fn) == str:
            if denoise_fn == "gauss":
                noise = torch.randn_like(x_t)
            elif denoise_fn == "noise_fn":
                noise = self.noise_fn(x_t, t).float()
        else:
            noise = denoise_fn(x_t, t)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_x_0": out["pred_x_0"]}

    def forward_backward(
            self, model, x, see_whole_sequence="half", t_distance=None, denoise_fn="gauss"
            ):
        assert see_whole_sequence == "whole" or see_whole_sequence == "half" or see_whole_sequence == None

        if t_distance == 0:
            return x.detach()

        if t_distance is None:
            t_distance = self.num_timesteps
        seq = [x.cpu().detach()]
        if see_whole_sequence == "whole":
            for t in range(int(t_distance)):
                t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                noise = self.noise_fn(x, t_batch).float()
                with torch.no_grad():
                    x = self.sample_q_gradual(x, t_batch, noise)
                seq.append(x.cpu().detach())
        else:
            t_tensor = torch.tensor([t_distance - 1], device=x.device).repeat(x.shape[0])
            x = self.sample_q(
                    x, t_tensor,
                    self.noise_fn(x, t_tensor).float()
                    )
            if see_whole_sequence == "half":
                seq.append(x.cpu().detach())
            seq.append(x.cpu().detach())

        for t in range(int(t_distance) - 1, -1, -1):
            t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
            with torch.no_grad():
                out = self.sample_p(model, x, t_batch, denoise_fn)
                x = out["sample"]
            # if see_whole_sequence:
                seq.append(x.cpu().detach())

        return seq

    def my_forward_backward(
            self, model, x, control_matrix, see_whole_sequence="half", t_distance=None, denoise_fn="gauss",
            ):
        assert see_whole_sequence == "whole" or see_whole_sequence == "half" or see_whole_sequence == None

        if t_distance == 0:
            return x.detach()

        if t_distance is None:
            t_distance = self.num_timesteps
        seq = [x.cpu().detach()]
        if see_whole_sequence == "whole":
            for t in range(int(t_distance)):
                t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                noise = self.noise_fn(x, t_batch).float()
                noise[control_matrix == 0] = 0
                control_matrix[control_matrix != 0] -= 1

                with torch.no_grad():
                    x = self.sample_q_gradual(x, t_batch, noise)

                seq.append(x.cpu().detach())
        else:
            t_tensor = torch.tensor([t_distance - 1], device=x.device).repeat(x.shape[0])
            x = self.sample_q_my(
                    x, t_tensor, control_matrix,
                    self.noise_fn(x, control_matrix).float()
                    )
            
            seq.append(x.cpu().detach())
            
            if see_whole_sequence == "half":
                seq.append(x.cpu().detach())

        for t in range(int(t_distance) - 1, -1, -1):
            t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
            with torch.no_grad():
                out = self.sample_p(model, x, t_batch, denoise_fn)
                x = out["sample"]
                seq.append(x.cpu().detach())

        return seq
    
    def sample_q_my(self, x_0, t_x, t_noise, noise):
        """
            modified q (x_t | x_0 )

            :param x_0:
            :param t_x: t for x term
            :param t_noise: t for noise term
            :param noise:
            :return:
        """
        return (extract(self.sqrt_alphas_cumprod, t_x, x_0.shape, x_0.device) * x_0 +
                extract(self.sqrt_one_minus_alphas_cumprod, t_noise, x_0.shape, x_0.device) * noise)
    
    def sample_q(self, x_0, t, noise):
        """
            q (x_t | x_0 )

            :param x_0:
            :param t:
            :param noise:
            :return:
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0 + 
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape, x_0.device) * noise
                )

    def sample_q_gradual(self, x_t, t, noise):
        """
        q (x_t | x_{t-1})
        :param x_t:
        :param t:
        :param noise:
        :return:
        """
        return (extract(self.sqrt_alphas, t, x_t.shape, x_t.device) * x_t +
                extract(self.sqrt_betas, t, x_t.shape, x_t.device) * noise)

    def calc_vlb_xt(self, model, x_0, x_t, t, estimate_noise=None):
        # find KL divergence at t
        true_mean, _, true_log_var = self.q_posterior_mean_variance(x_0, x_t, t)
        output = self.p_mean_variance(model, x_t, t, estimate_noise)
        kl = normal_kl(true_mean, true_log_var, output["mean"], output["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretised_gaussian_log_likelihood(
                x_0, output["mean"], log_scales=0.5 * output["log_variance"]
                )
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        nll = torch.where((t == 0), decoder_nll, kl)
        return {"output": nll, "pred_x_0": output["pred_x_0"]}

    def calc_loss(self, model, x_0, t):
        # noise = torch.randn_like(x)

        noise = self.noise_fn(x_0, t).float()

        x_t = self.sample_q(x_0, t, noise)
        estimate_noise = model(x_t, t)
        loss = {}
        if self.loss_type == "l1":
            loss["loss"] = mean_flat((estimate_noise - noise).abs())
        elif self.loss_type == "l2":
            loss["loss"] = mean_flat((estimate_noise - noise).square())
        elif self.loss_type == "hybrid":
            # add vlb term
            loss["vlb"] = self.calc_vlb_xt(model, x_0, x_t, t, estimate_noise)["output"]
            loss["loss"] = loss["vlb"] + mean_flat((estimate_noise - noise).square())
        else:
            loss["loss"] = mean_flat((estimate_noise - noise).square())
        return loss, x_t, estimate_noise

    def p_loss(self, model, x_0, args):
        self.x_0 = x_0
        if self.loss_weight == "none":
            if args["train_start"]:
                t = torch.randint(
                        0, min(args["sample_distance"], self.num_timesteps), (x_0.shape[0],),
                        device=x_0.device
                        )
            else:
                t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=x_0.device)
            weights = 1
        else:
            t, weights = self.sample_t_with_weights(x_0.shape[0], x_0.device)

        loss, x_t, eps_t = self.calc_loss(model, x_0, t)
        loss = ((loss["loss"] * weights).mean(), (loss, x_t, eps_t))
        return loss
