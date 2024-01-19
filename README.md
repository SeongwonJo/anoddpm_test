## About Project

Pneumonia Treatment Progress Prediction using Diffusion Model 

<br><br><br>

## Usage

examples
<br><br>


### Training

```
python diffusion_training.py -y <num> -i <image folder path> -d <device> -r <resume option>
```

- 모델의 파라미터 관련 옵션은 test_args 폴더에 args1.yaml 과 같이 yaml 파일로 저장하여 사용

- -y  :  args\<num>.yaml 의 숫자를 입력

- -i  :  학습용 이미지 폴더의 경로 (이 프로젝트에서는 학습시 정상 폐의 방사선 이미지만 사용)

- -d  :  사용할 gpu 설정 (기본값 = cuda:0)

- -r  :  "auto" 입력하여 사용하는 yaml로 학습한 모델이 있다면 이어서 학습 또는 pt file 의 경로를 입력하여 이어서 학습

- 예시

  - ```
    python diffusion_training.py -y 1 -i ../data/chest_xray/train/NORMAL/
    ```

<br><br>   

## Generate image

```
python generate_images.py -y <num> -i <image folder path> -d <device> -l <lambda list> -p <pt file path> -m <model name> --use_control_matrix
```

- 모델의 파라미터 관련 옵션은 test_args 폴더에 args1.yaml 과 같이 yaml 파일로 저장하여 사용

- -y  :  args\<num>.yaml 의 숫자를 입력

- -i  :  사용할 이미지 폴더의 경로

- -d  :  사용할 gpu 설정 (기본값 = cuda:0)

- -l  :  사용할 λ 값(노이즈 스텝) 입력

  - 입력예시

    - ```
      -l 100
      ```

    - ```
      -l 100,200,300
      ```

      __!! 여러 값 입력시 띄어쓰기 x !!__

- --use_control_matrix  :  제어행렬 사용 , -m, -p 파라미터 입력 필수

  - -p  :  CAM 값 구하는 데 사용할 pt 파일 경로
  - -m  :  모델 이름
    - -m 옵션으로 들어가는 모델명 목록
      - resnet50
      - resnet101
      - resnet152
      - densenet121
      - densenet201
      - densenet121_2048
        - fc layer 길이를 2048 로 늘린 모델

- 예시

  - ```
    python generate_images.py -y 1 -i ../data/chest_xray/test/PNEUMONIA/ -l 200,300,400
    ```
    
  - ```
    python generate_images.py -y 1 -i ../data/chest_xray/test/PNEUMONIA/ -l 200,300,400 -p ./resnet152.pt -m resnet152 --use_control_matrix
    ```

<br><br>



## 생성결과 정리용 코드 업데이트 예정 ..




<br><br>


### Reference

```
https://github.com/Julian-Wyatt/AnoDDPM
```
