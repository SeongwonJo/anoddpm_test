import collections
import copy
import time
import argparse

import matplotlib.pyplot as plt
import numpy as np
from torch import optim

import dataset
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from helpers import *
from UNet import UNetModel, update_ema_params


def ArgumentParse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-y', '--yml_num', required=True, help="number of yml file contains options")
    parser.add_argument('-i', '--image_path', required=True ,help="need train image folder path")
    # parser.add_argument('-t', '--test_path' ,help="need test image folder path")
    parser.add_argument('-d', '--device', default="cuda:0")
    parser.add_argument('-r', '--resume', default=None, help="input 'auto' or 'pt file path' ")
    parser.add_argument('--is_rgb', default=False)

    args = parser.parse_args()
    return args


def train(training_dataset, args, resume, device):
    """

    :param training_dataset_loader: cycle(dataloader) instance for training
    :param testing_dataset_loader:  cycle(dataloader) instance for testing
    :param args: dictionary of parameters
    :param resume: dictionary of parameters if continuing training from checkpoint
    :return: Trained model and tested
    """


    training_dataset_loader = dataset.init_dataset_loader(training_dataset, args)


    model = UNetModel(
            args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], dropout=args[
                "dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
            in_channels=args['in_channels']
            )

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diffusion = GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=args['in_channels']
            )

    if resume: # loaded model
        if "unet" in resume:
            model.load_state_dict(resume["unet"])
        else:
            model.load_state_dict(resume["ema"])

        ema = UNetModel(
                args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'],
                dropout=args["dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
                in_channels=args['in_channels']
                )
        ema.load_state_dict(resume["ema"])
        start_epoch = resume['n_epoch']

    else:
        start_epoch = 1
        ema = copy.deepcopy(model)

    tqdm_epoch = range(start_epoch, args['EPOCHS'] + 1)
    model.to(device)
    ema.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(args['lr']), weight_decay=args['weight_decay'], betas=(0.9, 0.999))
    if resume:
        optimizer.load_state_dict(resume["optimizer_state_dict"])

    del resume
    start_time = time.time()
    losses = []
    iters = range(len(training_dataset) // args['Batch_Size'])
    # iters = range(10) # for test
    # 1349 장 기준으로 에폭당 1~2시간 소요

    # dataset loop
    for epoch in tqdm_epoch:
        mean_loss = []

        for i in iters:
            data = next(training_dataset_loader)
            if args["dataset"] == "custom":
                x = data["image"]
                x = x.to(device)

            loss, estimates = diffusion.p_loss(model, x, args)

            noisy, est = estimates[1], estimates[2]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            update_ema_params(ema, model)
            mean_loss.append(loss.data.cpu())

        losses.append(np.mean(mean_loss))
        
        _time = time.time()
        time_taken = _time - start_time
        start_time = time.time()
        remaining_epochs = args['EPOCHS'] - epoch
        est_hours = remaining_epochs * time_taken / 3600
        est_mins = (est_hours % 1) * 60
        est_hours = int(est_hours)

        print_terminal_width_line()
        print(
                f"|    Epoch: {epoch}    |    Trained Images: {(i + 1) * args['Batch_Size'] + epoch * len(iters) * args['Batch_Size']}"
                f"\n|    Last 20 epoch mean loss: {np.mean(losses[-20:]):.4f}    "
                f"|    Last 100 epoch mean loss: {np.mean(losses[-100:]) if len(losses) > 0 else 0:.4f}"
                f"\n|    Time taken {time_taken:.2f}s    " 
                f"|    Time elapsed {int(time_taken / 3600)}: {((time_taken / 3600) % 1) * 60:02.0f}    " 
                f"|    Estimate time remaining: {est_hours}:{est_mins:02.0f}"
                )
        print_terminal_width_line()


        if epoch % 5 == 0:
            print("\nSampling test progressing ...")
            row_size = min(8, args['Batch_Size'])
            training_outputs(
                    diffusion, x, est, noisy, epoch, row_size, save_imgs=args['save_imgs'], model=ema, args=args
                    )
        
            _time = time.time()
            time_taken = _time - start_time
            start_time = time.time()
            print(f"\nSampling image saved... time taken {time_taken:.2f}s\n")
        

        if epoch % 10 == 0 and epoch >= 0:
            save(unet=model, args=args, optimizer=optimizer, final=False, ema=ema, epoch=epoch)

    save(unet=model, args=args, optimizer=optimizer, ema=ema, final=True)


def save(final, unet, optimizer, args, ema, loss=0, epoch=0):
    """
    Save model final or checkpoint
    :param final: bool for final vs checkpoint
    :param unet: unet instance
    :param optimizer: optimizer instance
    :param args: model parameters
    :param ema: ema instance
    :param loss: loss for checkpoint
    :param epoch: epoch for checkpoint
    :return: saved model
    """
    ROOT_DIR = "./"


    if final:
        torch.save(
                {
                    'n_epoch':              args["EPOCHS"],
                    'model_state_dict':     unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    "ema":                  ema.state_dict(),
                    "args":                 args
                    # 'loss': LOSS,
                    }, f'{ROOT_DIR}model/diff-params-ARGS={args["arg_num"]}/params-final.pt'
                )
    else:
        torch.save(
                {
                    'n_epoch':              epoch,
                    'model_state_dict':     unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    "args":                 args,
                    "ema":                  ema.state_dict(),
                    'loss':                 loss,
                    }, f'{ROOT_DIR}model/diff-params-ARGS={args["arg_num"]}/checkpoint/diff_epoch={epoch}.pt'
                )


def training_outputs(diffusion, x, est, noisy, epoch, row_size, model, args, save_imgs=False):
    """
    Saves images based on args info
    :param diffusion: diffusion model instance
    :param x: x_0 real data value
    :param est: estimate of the noise at x_t (output of the model)
    :param noisy: x_t
    :param epoch:
    :param model:
    :param row_size: rows for outputs into torchvision.utils.make_grid
    :param save_imgs: bool for saving imgs
    :return:
    """

    try:
        os.makedirs(f'./diffusion-training-images/ARGS={args["arg_num"]}')
    except OSError:
        pass

    if save_imgs:
        img = torch.randn_like(x, device=x.device)
        for t in range(int(args["sample_distance"]) - 1, -1, -1):
            t_temp = torch.tensor([t], device=x.device).repeat(x.shape[0])
            with torch.no_grad():
                out = diffusion.sample_p(model, img, t_temp)
                img = out["sample"]

        plt.title(f'{epoch}epoch trained model sampling')
        plt.rcParams['figure.dpi'] = 150
        plt.grid(False)
        plt.imshow(gridify_output(img, row_size), cmap='gray')
        plt.axis('off')

        plt.savefig(f'./diffusion-training-images/ARGS={args["arg_num"]}/EPOCH={epoch}.png')
    
    plt.close('all')


def main():
    """
        Load arguments, run training and testing functions, then remove checkpoint directory
    :return:
    """
    # make directories
    for i in ['./model/', './diffusion-training-images/']:
        try:
            os.makedirs(i)
        except OSError:
            pass
    torch.cuda.empty_cache()

    argparse = ArgumentParse()
    device = torch.device(argparse.device if torch.cuda.is_available() else 'cpu')

    args = load_parameters(argparse.yml_num)
    args["arg_num"] = argparse.yml_num
    print_terminal_width_line()
    print(args)
    print_terminal_width_line()

    # make arg specific directories
    for i in [f'./model/diff-params-ARGS={argparse.yml_num}',
              f'./model/diff-params-ARGS={argparse.yml_num}/checkpoint',
              f'./diffusion-training-images/ARGS={argparse.yml_num}']:
        try:
            os.makedirs(i)
        except OSError:
            pass

    if argparse.is_rgb:
        args['in_channels'] = 3
    else :
        args['in_channels'] = 1

    training_dataset = dataset.custom(
            argparse.image_path, img_size=args["img_size"], rgb=argparse.is_rgb
            )
    print(f'\n{len(training_dataset)} images loaded.\n')

    # if resuming, loaded model is attached to the dictionary
    loaded_model = {}
    if argparse.resume:
        if argparse.resume == "auto":
            checkpoints = os.listdir(f'./model/diff-params-ARGS={argparse.yml_num}/checkpoint')
            checkpoints.sort(reverse=True)
            for i in checkpoints:
                try:
                    file_dir = f"./model/diff-params-ARGS={argparse.yml_num}/checkpoint/{i}"
                    loaded_model = torch.load(file_dir, map_location=device)

                    print(f"\n'./model/diff-params-ARGS={argparse.yml_num}/checkpoint/{i}' loaded.")
                    break
                except RuntimeError:
                    continue
        else:
            file_dir = argparse.resume
            loaded_model = torch.load(file_dir, map_location=device)
            print(f"\n'{argparse.resume}' loaded.")


    # load, pass args
    train(training_dataset, args, loaded_model, device=device)

    # remove checkpoints after final_param is saved (due to storage requirements)
    # for file_remove in os.listdir(f'./model/diff-params-ARGS={argparse.yml_num}/checkpoint'):
    #     os.remove(os.path.join(f'./model/diff-params-ARGS={argparse.yml_num}/checkpoint', file_remove))
    # os.removedirs(f'./model/diff-params-ARGS={argparse.yml_num}/checkpoint')


if __name__ == '__main__':
    start = time.time()

    main()

    print(f"training script done: {time.time() - start:.2f}s")