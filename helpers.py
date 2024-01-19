import os
from collections import defaultdict
import yaml
import shutil
import matplotlib.pyplot as plt
import glob
import cv2

import torch
import torchvision.utils


def print_terminal_width_line():
    terminal_width, _ = shutil.get_terminal_size()
    line = '=' * terminal_width
    print(line)


def gridify_output(img, row_size=-1):
    scale_img = lambda img: ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    return torchvision.utils.make_grid(scale_img(img), nrow=row_size, pad_value=-1).cpu().data.permute(
            0, 2,
            1
            ).contiguous().permute(
            2, 1, 0
            )

# 딕셔너리 기본값처리 : 모든 키에 대해 값이 없는 경우 자동으로 기본값 할당 
# 여기서는 str 로 해놓아서 공백 ""이 기본값으로 들어감
def defaultdict_from_dict(dict): 
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(dict)
    return dd


def load_checkpoint(param, use_checkpoint, device):
    """
    loads the most recent (non-corrupted) checkpoint or the final model
    :param param: args number
    :param use_checkpoint: checkpointed or final model
    :return:
    """
    if not use_checkpoint:
        return torch.load(f'./model/diff-params-ARGS={param}/params-final.pt', map_location=device)
    else:
        checkpoints = os.listdir(f'./model/diff-params-ARGS={param}/checkpoint')
        checkpoints.sort(reverse=True)
        for i in checkpoints:
            try:
                file_dir = f"./model/diff-params-ARGS={param}/checkpoint/{i}"
                loaded_model = torch.load(file_dir, map_location=device)
                break
            except RuntimeError:
                continue
        return loaded_model


def yml_to_dict(args_num):
    with open(f'./test_args/args{args_num}.yaml') as f:
        taskdict = yaml.load(f, Loader=yaml.FullLoader)
    taskdict['arg_num'] = args_num
    taskdict = defaultdict_from_dict(taskdict)
    return taskdict


def load_parameters(args_num):
    """
    Loads the trained parameters for the detection model
    :return:
    """
    try:
        args = yml_to_dict(args_num=args_num)
    except FileNotFoundError:
        raise ValueError(f"args{args_num} doesn't exist")

    if "noise_fn" not in args:
        args["noise_fn"] = "gauss"

    print(f'\nargs{args_num} parameters loaded.\n')

    return args


def make_table(lamb_list, path):
    xlabels = ["Original"]
    folders = [f'{path}/in/']

    for l in lamb_list:
        xlabels.append(f"λ={l}")
        folders.append(f'{path}/images_{l}/out/')
    # xlabels = ["Original", "λ=200", "λ=300", "λ=400", "λ=500", "λ=600"]
    # folders = ['ARGS=33/images_200/in/',
    #            'ARGS=33/images_200/out/',
    #            'ARGS=33/images_300/out/',
    #            'ARGS=33/images_400/out/',
    #            'ARGS=33/images_500/out/',
    #            'ARGS=33/images_600/out/']

    rows = len(xlabels)
    cols = len(folders)
    num_images = len(glob.glob(folders[0] + "*.png"))

    i = 1
    j = 0
    fig = plt.figure(figsize=(6, 8))  # rows*cols 행렬의 i번째 subplot 생성
    for folder in folders:
        # print(folder)
        for filename in sorted(glob.glob(folder + "*.png")):
            img = cv2.imread(filename)
            # print(filename)
            ax = fig.add_subplot(rows, cols, i)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if i % cols == 1:
                ax.set_ylabel(xlabels[j])
                j += 1
            ax.set_xticks([]), ax.set_yticks([])
            i += 1

    plt.savefig(f'{path}/result.png', dpi=200)
    plt.clf()
    plt.close()
        

# if __name__ == '__main__':
