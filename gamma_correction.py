import numpy as np
import cv2
import os
import argparse

# img = cv2.imread("../NORMAL-6433176-0001.jpeg")

#
# path = "D:/chest-xray/chest_xray2/train/(resize)/test/"
# save_path = "D:/chest-xray/chest_xray2/train/(resize)/noise_test/"
# img_list = os.listdir(path)


def make_img_list(path):
    img_list = os.listdir(path)
    return img_list


def gamma_correction(image, gamma=0.5):
    f_img = np.float32(image)

    normalize = f_img / 255

    apply_gamma_c = normalize ** (1 / gamma)

    result = np.uint8(apply_gamma_c * 255)

    return result

# result = gamma_correction(img)

# cv2.imshow("gamma correction", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Image Classification')
    parser.add_argument('-p', "--path", required=True, default="./temp/")
    parser.add_argument('-s', "--save_path", required=True, default="./temp/")
    parser.add_argument('-g', "--gamma", default=1.0, type=float)
    args = parser.parse_args()

    img_list = make_img_list(args.path)

    for img in img_list:
        x = cv2.imread(args.path + img)
        x_g = gamma_correction(x, args.gamma)

        cv2.imwrite(args.save_path + img, x_g)



