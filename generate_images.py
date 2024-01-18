import os

import argparse

from matplotlib import font_manager
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
import numpy as np

import dataset
from helpers import gridify_output, load_parameters, load_checkpoint, print_terminal_width_line
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from get_cam_score import make_cam_score
from UNet import UNetModel

import time

def ArgumentParse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-y', '--yml_num', required=True, help="number of yml file contains options")
    parser.add_argument('-i', '--image_path', required=True ,help="need image folder path")
    parser.add_argument('-d', '--device', default="cuda:0")
    parser.add_argument('-p', '--pt_path' ,help="need pt file path")
    parser.add_argument('-m', '--model' ,help="model name of pt file")
    parser.add_argument('-l', '--lambda_list', required=True ,help="input list of lambda values\nExample : -l 100,200,300", type=str)

    parser.add_argument('--num_iter', default=0, type=int)
    parser.add_argument('--is_rgb', default=False)
    parser.add_argument('--seq_setting', required=False, default=None)
    parser.add_argument('--use_checkpoint', required=False)
    parser.add_argument('--use_control_matrix', required=False)

    args = parser.parse_args()
    return args


def main():
    argparse = ArgumentParse()
    device = torch.device(argparse.device if torch.cuda.is_available() else 'cpu')

    plt.set_cmap('gray')
    plt.rcParams['figure.dpi'] = 400
    # scale_img = lambda img: ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # add times new roman to mpl fonts
    font_path = "./times new roman.ttf"
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = prop.get_name()

    """
    generate videos for dataset based on input arguments
    :return: selection of videos for dataset of trained model
    """
    # load parameters
    args = load_parameters(argparse.yml_num)
    output = load_checkpoint(argparse.yml_num, argparse.use_checkpoint, argparse.device)

    args["Batch_Size"] = 1

    if argparse.is_rgb:
        in_channels = 3
    else :
        in_channels = 1

    # init model, betas and diffusion classes
    unet = UNetModel(
            args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], in_channels=in_channels
            )

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diff = GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=in_channels
            )
    
    print(args)
    # load checkpoint
    unet.load_state_dict(output["ema"])
    unet.to(device)
    unet.eval()
    
    d_set = dataset.custom(
        argparse.image_path, img_size=args["img_size"], rgb=False)

    loader = dataset.init_dataset_loader(d_set, args)

    if argparse.use_control_matrix:
        args_control_matrix = {
            "model": argparse.model,
            "pt_path": argparse.pt_path
            }

    seq_setting = argparse.seq_setting # defalut : None / see all sequance to input ' --seq_setting "whole" '


    t_distance_list = [int(item) for item in argparse.lambda_list.split(',')]
    
    # snu_he : 2164  /  kaggle : 390    || snuhe_sample : 9   /   kaggle_sample : 5
    if argparse.num_iter == 0:
        num_iter = len(d_set)
    else :
        num_iter = argparse.num_iter

    for t_distance in t_distance_list:
        print_terminal_width_line()
        print("\nt_distance set: ", t_distance)
        print_terminal_width_line()
        # make directories
        for i in [f'./final-outputs/', f'./final-outputs/ARGS={args["arg_num"]}', f'./final-outputs/ARGS={args["arg_num"]}/images_{t_distance}/']:
            if not os.path.exists(i):
                os.makedirs(i)

        # 임시 : 생성한 이미지 나누려고 만듦
        for i in [f'./final-outputs/ARGS={args["arg_num"]}/images_{t_distance}/in/',
                  f'./final-outputs/ARGS={args["arg_num"]}/images_{t_distance}/out/']:
            if not os.path.exists(i):
                os.makedirs(i)

        # generate 20 videos
        for i in range(num_iter):
            # print("\r"+"generate images : ", f"{i+1} / {num_iter}", end="")
            print_terminal_width_line()
            print("generate images : ", f"{i+1} / {num_iter}")

            new = next(loader)
            img = new["image"].to(device)

            print("\nload img",img.shape)
            
            if argparse.use_control_matrix:
                print("\nget cam score ...")
                control_matrix = make_cam_score(args_control_matrix, "gradcam", img) # args, method, target_img
                print("\n control matrix",control_matrix.shape)
                control_matrix = np.floor(control_matrix * t_distance)

                print("\nperform diffusion ...")
                # perform diffusion
                output = diff.my_forward_backward(
                        unet, img, control_matrix=control_matrix.type(torch.int64),
                        see_whole_sequence=seq_setting,
                        t_distance=t_distance, denoise_fn=args["noise_fn"]
                        )
            else:
                print("\nperform diffusion ...")
                # perform diffusion
                output = diff.forward_backward(
                        unet, img,
                        see_whole_sequence=seq_setting,
                        t_distance=t_distance, denoise_fn=args["noise_fn"]
                        )

            # savedir set
            temp2 = os.listdir(
                    f'./final-outputs/ARGS={args["arg_num"]}/images_{t_distance}/in/'
                    )
            
            output_name2 = f'./final-outputs/ARGS={args["arg_num"]}/images_{t_distance}'
            output_name3 = f'/attempt={len(temp2)}'

            # save gif (when --seq_setting "whole")
            # print("\nsaving gif ... \n")

            # use every image
            # img_list = []
            # for out in output:
            #     img_list.append(gridify_output(out, 1))
            # imageio.mimsave(output_name2 + output_name3 + ".gif", img_list, 'GIF', duration=0.1)


            # save image
            print("\nsaving input & output image ... \n")
            
            plt.imsave(output_name2 + '/in/' + output_name3 + "-in0.png", np.uint8(gridify_output(output[0], 1)))

            plt.imsave(output_name2 + '/out/' + output_name3 + "-out0.png", np.uint8(gridify_output(output[-1:][0], 1)))
            
            if seq_setting == "whole":
                plt.imsave(f'./final-outputs' + output_name3 + f"-test_{t_distance}.png", np.uint8(gridify_output(output[t_distance + 1], 1)))
            else:
                plt.imsave(f'./final-outputs' + output_name3 + f"-test_{t_distance}.png", np.uint8(gridify_output(output[1], 1)))

            plt.close('all')




if __name__ == '__main__':
    start_time = time.time()

    main()

    print(f"image generated done: {time.time() - start_time:.2f}s")
