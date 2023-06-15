"""
Output varying figures for paper
"""

import random
import argparse

import os
import cv2
from matplotlib import animation
from torchvision import transforms
import numpy as np
import pandas as pd

import torch
from PIL import Image

import dataset
from helpers import gridify_output, load_parameters

from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp

from IQA_pytorch import VIF, SSIM, utils


def img_similarity_result(path1, path2, name):

    imgs_input = os.listdir(path1)
    imgs_output = os.listdir(path2)


    sum_mse = 0
    sum_psnr = 0
    sum_vif = 0
    sum_ssim = 0

    mse_list = []
    vif_list = []
    psnr_list = []
    ssim_list = []

    for idx in range(len(imgs_input)):
        print("\r"+"processing ... ", idx, end="")
        img_in = cv2.imread(path1 + imgs_input[idx], cv2.IMREAD_GRAYSCALE)
        img_out = cv2.imread(path2 + imgs_output[idx], cv2.IMREAD_GRAYSCALE)

        # print("======================")
        # print("MSE: ", mse(img_in, img_out))
        # print("RMSE: ", rmse(img_in, img_out))
        # print("PSNR: ", psnr(img_in, img_out))
        # print("SSIM: ", ssim(img_in, img_out))
        # print("UQI: ", uqi(img_in, img_out))
        # print("MSSSIM: ", msssim(img_in, img_out))
        # print("ERGAS: ", ergas(img_in, img_out))
        # print("SCC: ", scc(img_in, img_out))
        # print("RASE: ", rase(img_in, img_out))
        # print("SAM: ", sam(img_in, img_out))
        # print("VIF: ", vifp(img_in, img_out))
        # print("=======================")

        _mse = mse(img_in, img_out)
        _vifp = vifp(img_in, img_out)
        _psnr = psnr(img_in, img_out)
        _ssim = ssim(img_in, img_out)[0]

        mse_list.append(_mse)
        vif_list.append(_vifp)
        psnr_list.append(_psnr)
        ssim_list.append(_ssim)


        sum_mse += _mse
        sum_vif += _vifp
        sum_psnr += _psnr
        sum_ssim += _ssim


    r_dict = {"img":imgs_input, 
              "mse":mse_list, 
            #   "psnr":psnr_list, 
              "ssim":ssim_list,
              "vif":vif_list}

    vif_list = np.array(vif_list)
    
    vif_var = np.var(vif_list)

    re_vif_list = vif_list / vif_var

    r_dict['relative vif'] = re_vif_list

    re_vif_list_exp = np.exp(re_vif_list) / np.sum(np.exp(re_vif_list))
    r_dict['relative vif with exp'] = re_vif_list_exp
    
    path = f'./similarity_result_{name}.csv'
    df = pd.DataFrame(r_dict)
    # df = df.transpose()
    # df.columns = ['mse', 'vif', 'psnr']
    if not os.path.exists(path):
        df.to_csv(path, mode="w")
    else:
        df.to_csv(path, mode="a", header=False)

    print()
    print("MSE: ", sum_mse / len(imgs_input))
    print("VIF: ", sum_vif / len(imgs_input))
    print("SSIM: ", sum_psnr / len(imgs_input))
    print("PSNR: ", sum_ssim / len(imgs_input))


def img_similarity_result_torch(path1, path2, name):

    imgs_input = os.listdir(path1)
    imgs_output = os.listdir(path2)


    sum_mse = 0
    sum_psnr = 0
    sum_vif = 0
    sum_ssim = 0

    mse_list = []
    vif_list = []
    psnr_list = []
    ssim_list = []

    for idx in range(len(imgs_input)):
        print("\r"+"processing ... ", idx, end="")
        img_in = utils.prepare_image(Image.open(path1 + imgs_input[idx]).convert("L")).to(device)
        img_out = utils.prepare_image(Image.open(path2 + imgs_output[idx]).convert("L")).to(device)

        model_ssim = SSIM(channels=1)
        model_vif = VIF(channels=1)

        _ssim = model_ssim(img_out, img_in, as_loss=False)
        _vifp = model_vif(img_out, img_in, as_loss=False)
        print(_vifp)

        # mse_list.append(_mse)
        vif_list.append(_vifp.cpu())
        # psnr_list.append(_psnr)
        ssim_list.append(_ssim)


        # sum_mse += _mse
        sum_vif += _vifp
        # sum_psnr += _psnr
        sum_ssim += _ssim


    r_dict = {"img":imgs_input, 
            #   "mse":mse_list, 
            #   "psnr":psnr_list, 
              "ssim":ssim_list,
              "vif":vif_list}

    vif_list = np.array(vif_list)
    
    vif_var = np.var(vif_list)

    re_vif_list = vif_list / vif_var

    r_dict['relative vif'] = re_vif_list

    re_vif_list_exp = np.exp(re_vif_list) / np.sum(np.exp(re_vif_list))
    r_dict['relative vif with exp'] = re_vif_list_exp
    
    path = f'./similarity_result_{name}.csv'
    df = pd.DataFrame(r_dict)
    # df = df.transpose()
    # df.columns = ['mse', 'vif', 'psnr']
    if not os.path.exists(path):
        df.to_csv(path, mode="w")
    else:
        df.to_csv(path, mode="a", header=False)

    print()
    # print("MSE: ", sum_mse / len(imgs_input))
    print("VIF: ", sum_vif / len(imgs_input))
    print("SSIM: ", sum_psnr / len(imgs_input))
    # print("PSNR: ", sum_ssim / len(imgs_input))


def make_saliency_map(path1, path2):

    for i in [f'./heatmap_images/']:
        if not os.path.exists(i):
            os.makedirs(i)
    
    n = 1
    while os.path.exists(f'./heatmap_images/exp{n}'):
        n += 1
    os.makedirs(f'./heatmap_images/exp{n}')


    imgs_input = os.listdir(path1)
    imgs_output = os.listdir(path2)

    for idx in range(len(imgs_input)):
        print("\r"+"processing ... ", idx, end="")
        img_in = cv2.imread(path1 + imgs_input[idx], 0)
        img_out = cv2.imread(path2 + imgs_output[idx], 0)

        # compute difference
        # difference = cv2.subtract(img_in, img_out)
        difference = cv2.absdiff(img_in, img_out)

        # img_in = img_in.astype(np.float32)
        # img_out = img_out.astype(np.float32)
        
        # img_in_norm = ((img_in - img_in.min()) * (1) / (img_in.max() - img_in.min()))
        # img_out_norm = ((img_out - img_out.min()) * (1) / (img_out.max() - img_out.min()))
        
        # difference = (img_in_norm - img_out_norm) ** 2

        # print("norm 1",difference)
        # img_norm = img_norm.astype(np.uint8)
        # mask = np.where(difference > 0.5, 1, 0)
        # print(mask)

        # img_norm = ((img_f - img_f.min()) * (255) / (img_f.max() - img_f.min()))
        # print("norm 255",img_norm)
        # difference = img_norm.astype(np.uint8)

        # color the mask red
        # Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

        # ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
        # difference[mask != 255] = [0, 0, 255]

        # threshold = 1
        # i_mask = difference > threshold

        # mask = np.zeros_like(img_in, np.uint8)
        # mask[i_mask] = img_out[i_mask]
        _, mask = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY)

        cv2.imwrite(f'./heatmap_images/exp{n}/'+ imgs_input[idx][:-8] + '_diff.png', mask)

    print("\n"+"="*20)
    print(f"\nresult saved 'exp{n}'")
    print("\n"+"="*20)


        # 구조화 요소 커널, 사각형 생성 ---①
        # k = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        # 열림 연산 적용 ---②
        # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        # cv2.imwrite(f'./heatmap_images/exp{n}/'+ imgs_input[idx][:-8] + '_diff.png', opening)

        # result = cv2.subtract(cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY), opening)

        # cv2.imwrite(f'./heatmap_images/exp{n}/'+ imgs_input[idx][:-8] + '_diff.png', result)


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="check similarity")
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("-t", "--task", default="no")
    parser.add_argument("-n", "--args_num")
    parser.add_argument("-o", "--order")

    arguments = parser.parse_args()
    
    device = torch.device(arguments.device if torch.cuda.is_available() else 'cpu')

    # path_in = path_main + "/images_450/in/"
    # path_out = path_main + "/images_500/out/"

    # img_similarity_result(path_in, path_out)

    # path_in = 

    if arguments.task == "test":
        print("test mode\n")
        path_main = "/home/swcho/AnoDDPM/test/"
        path_in = path_main + "/in/"
        path_out = path_main + "/out_500/"
        if arguments.order == "sim":
            img_similarity_result_torch(path_in, path_out, "test")
        elif arguments.order == "map":
            make_saliency_map(path_in, path_out)
    elif arguments.task == "no":
        path_main = f"/home/swcho/AnoDDPM/final-outputs/ARGS={arguments.args_num}/"

        for i in [500]:
            print(f"step {i} result :")
            path_in = path_main + f"images_{i}/in/"
            path_out = path_main + f"images_{i}/out/"
            if arguments.order == "sim":
                img_similarity_result(path_in, path_out, i)
            elif arguments.order == "map":
                make_saliency_map(path_in, path_out)
            print("\n"+"="*20)
    else:
        print("what?")