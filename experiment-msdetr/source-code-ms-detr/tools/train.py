import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR('/home/wqg/ObjectDetection/MSDETR/ultralytics/cfg/models/rt-detr/MSDETR.yaml') #path to model cfg file
    # model.load('') # loading pretrain weights(optional)
    model.train(data='/home/wqg/ObjectDetection/datasets/VisDrone/VisDrone.yaml', #path to dataset yaml file
                cache=False,
                imgsz=640, #image size
                epochs=300, #train epochs
                batch=4, #batch size
                workers=4, #num workers
                project='runs/train/VisDrone', #result save path
                name='MSDETR', #result save name
                #resume='/home/wqg/ObjectDetection/rtdetr/runs/train/VisDrone/MSDETR/weights/last.pt', #resume last pt training
                )
