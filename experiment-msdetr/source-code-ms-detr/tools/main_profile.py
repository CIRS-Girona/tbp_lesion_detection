import os
import warnings
warnings.filterwarnings('ignore')
import torch
from ultralytics import RTDETR

#Print model parameters

if __name__ == '__main__':

    # select model yaml file
    YAML_BASE_PATH = '/home/wqg/ObjectDetection/MSDETR/ultralytics/cfg/models/rt-detr'
    yaml_filename = 'MSDETR.yaml'

    model_path = os.path.join(YAML_BASE_PATH, yaml_filename)
    # load model
    model = RTDETR(model_path)
    # model.model.eval()
    # model.info(detailed=True)
    try:
        model.profile(imgsz=[640, 640])
    except Exception as e:
        print(e)
        pass
    print('after fuse:', end='')
    model.fuse()
