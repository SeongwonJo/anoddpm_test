"""
Output varying figures for paper
"""

# import random
# import imageio
# import sys
import os

from matplotlib import animation, font_manager
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
import numpy as np

import dataset
from helpers import gridify_output, load_parameters
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from get_cam_score import make_cam_score
from UNet import UNetModel

import time



def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    plt.set_cmap('gray')

    plt.rcParams['figure.dpi'] = 600
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
    args, output = load_parameters(device)
    in_channels = 1
    if args["dataset"].lower() == "leather":
        in_channels = 3

    # init model, betas and diffusion classes
    unet = UNetModel(
            args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], in_channels=in_channels
            )

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diff = GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=in_channels
            )
    
    args["Batch_Size"] = 1
    print(args)
    # load checkpoint
    unet.load_state_dict(output["ema"])
    unet.to(device)
    unet.eval()
    if args["dataset"].lower() == "carpet":
        d_set = dataset.DAGM("./DATASETS/CARPET/Class1", True)
    elif args["dataset"].lower() == "leather":
        d_set = dataset.MVTec(
                "../dataset/mvtec/leather", anomalous=True, img_size=args["img_size"],
                rgb=True, include_good=False
                )
    elif args["dataset"].lower() == "custom":
        d_set = dataset.custom(
                "../data/chest_xray/test/PNEUMONIA/", anomalous=False, img_size=args["img_size"],
                rgb=False, e_aug = False
                )
        # d_set = dataset.custom(
        #         "./temp/cxr_sample/", anomalous=False, img_size=args["img_size"],
        #         rgb=False, e_aug = False
        #         )
    elif args["dataset"].lower() == "snu":
        d_set = dataset.custom(
                "../data/snu_xray-resize/PNEUMONIA/", anomalous=False, img_size=args["img_size"],
                rgb=False,
                )
    elif args["dataset"].lower() == "snu_he":
        d_set = dataset.custom(
                "../data/snu_he/PNEUMONIA/", anomalous=False, img_size=args["img_size"],
                rgb=False,
                )
        # d_set = dataset.custom(
        #         "./temp/snuhe_sample/", anomalous=False, img_size=args["img_size"],
        #         rgb=False,
        #         )
    elif args["dataset"].lower() == "snu_he_p":
        # d_set = dataset.custom(
        #         "../data/snu_he_PNEUMONIA/", anomalous=False, img_size=args["img_size"],
        #         rgb=False,
        #         )
        d_set = dataset.custom(
                "./temp/snuhe_normal_sample/", anomalous=False, img_size=args["img_size"],
                rgb=False,
                )
    elif args["dataset"].lower() == "snu_invert":
        d_set = dataset.custom(
                "../data/snu_invert_PNEUMONIA/", anomalous=False, img_size=args["img_size"],
                rgb=False,
                )

    loader = dataset.init_dataset_loader(d_set, args)
    plt.rcParams['figure.dpi'] = 600

    # t_distance = 200
    args_control_matrix = {
        "model": "resnet152",
        "pt_path": "/home/seongwon/PycharmProjects/Test_bench/kaggle_resnet152_gamma.pt"
        }

    ##############################
    use_control_matrix = True
    seq_setting = None
    ##############################


    t_distance_list = [300]
    num_iter = 100  # snu_he : 2164  /  kaggle : 390    || snuhe_sample : 9   /   kaggle_sample : 5
    for t_distance in t_distance_list:
        print("\n","="*60, "\n")
        print("\nt_distance set: ", t_distance)
        print("\n","="*60, "\n")
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
            print("\r"+"generate images : ", f"{i+1} / {num_iter}", end="")

            new = next(loader)
            img = new["image"].to(device)

            print("\n load img",img.shape)
            
            if use_control_matrix:
                print("\nget cam score ...")
                control_matrix = make_cam_score(args_control_matrix, "gradcam", img) # args, method, target_img
                print("\n control matrix",control_matrix.shape)
                # print("\n", control_matrix, "\n")
                # print(torch.max(control_matrix))
                control_matrix = np.floor(control_matrix * t_distance)
                # control_matrix = torch.where(control_matrix < t_distance / 2, t_distance / 2, control_matrix )
                # print(torch.max(control_matrix))
                # print(torch.min(control_matrix))
                # test = torch.where(control_matrix == 0, 9999, control_matrix)
                # print(torch.max(test))
                # print(torch.min(test))
                # print(test)
                # print(torch.bincount(control_matrix)[0])
                # print(control_matrix)


                print("\nperform diffusion ...")
                # perform diffusion
                output = diff.my_forward_backward(
                        unet, img, control_matrix=control_matrix.type(torch.int64),
                        see_whole_sequence=seq_setting,
                        # t_distance=5, denoise_fn=args["noise_fn"]
                        t_distance=t_distance, denoise_fn=args["noise_fn"]
                        )
            else:
                print("\nperform diffusion ...")
                # perform diffusion
                output = diff.forward_backward(
                        unet, img,
                        see_whole_sequence=seq_setting,
                        # t_distance=5, denoise_fn=args["noise_fn"]
                        t_distance=t_distance, denoise_fn=args["noise_fn"]
                        )

            
            
            # plot, animate and save diffusion process
            # fig, ax = plt.subplots()
            # plt.axis('off')
            # imgs = [[ax.imshow(gridify_output(output[x], 1), animated=True)] for x in range(0, len(output), 2)]
            # ani = animation.ArtistAnimation(
            #         fig, imgs, interval=25, blit=True,
            #         repeat_delay=1000
            #         )
            # temp = os.listdir(
            #         f'./final-outputs/ARGS={args["arg_num"]}'
            #         )

            # output_name = f'./final-outputs/ARGS={args["arg_num"]}/attempt={len(temp) + 1}-sequence.mp4'
            # ani.save(output_name)

            # plt.close('all')
            # print(len(output))


            # savedir set
            temp2 = os.listdir(
                    f'./final-outputs/ARGS={args["arg_num"]}/images_{t_distance}/in/'
                    )
            
            output_name2 = f'./final-outputs/ARGS={args["arg_num"]}/images_{t_distance}'
            output_name3 = f'/attempt={len(temp2)}'

            # save gif
            # print("\n saving gif ... \n")

            # from list
            # _save_list = [19, 57, 91, 201, 282] # kaggle
            # if len(temp2) in _save_list:
            #     img_list = []
            #     for out in output:
            #         img_list.append(gridify_output(out, 1))
            #     imageio.mimsave(output_name2 + output_name3 + ".gif", img_list, 'GIF', duration=0.1)

            # use every image
            # img_list = []
            # for out in output:
            #     img_list.append(gridify_output(out, 1))
            # imageio.mimsave(output_name2 + output_name3 + ".gif", img_list, 'GIF', duration=0.1)


            # just image
            # print("\n saving final image ... \n")
            # print(output)
            
            
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
    print("image generated")

    print(f"script done: {time.time() - start_time:.2f}s")
