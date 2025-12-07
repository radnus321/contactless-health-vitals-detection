# import cv2
# import numpy as np
# from dataloader import DatasetLoader
# from preprocessing.roi_extractor import ROIExtractor
# from preprocessing.pos_wang import apply_pos_wang
# from preprocessing.chrome_dehaan import apply_chrome_dehaan
# from preprocessing.green import apply_green
# from preprocessing.signal_extractor import extract_signals
# from preprocessing.signal_extractor import SignalExtractor

from data.dataloader import DatasetLoader
from data.phys_net_dataset import PhysNetDataset
from preprocessing.signal_extractor import SignalExtractor
from supervised_models.model_factory import build_model
from supervised_models.model_trainer import ModelTrainer

from torch.utils.data import DataLoader
import torch

dataloader = DatasetLoader("small_data/", "small_data/ToHealth_RepeatDataSet.xlsx")
dataloader.create_dataset(num_frames=30)

signal_extractor = SignalExtractor()
signal_extractor.extract_signals(dataloader.video_list, 30)

dataset = PhysNetDataset(
    target_list=dataloader.target_list,
    signal_extractor=signal_extractor
)

train_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    drop_last=True
)

model = build_model()

trainer = ModelTrainer(
    model=model
)

trainer.fit(
    train_loader=train_loader,
    epochs=10,
)

# features, target = dataset[0]["features"], dataset[0]["target"]
# print(features.shape)
# print(target)
# print(signal_extractor.pos_wang_signal)
# print(signal_extractor.chrome_dehaan_signal)
# print(signal_extractor.green_signal)
# extract_signals(dataloader.video_list, 30)

# roi_extractor = ROIExtractor("forehead")
# roi_video = []
# for frame in dataloader.video_list[0]:
#     roi_video.append(roi_extractor.extract_roi(frame))
# roi_video = np.array(roi_video)
# print("shape is: ", roi_video.shape)
# green_signal = apply_green(roi_video)
# print(green_signal)
# chrome_signal = apply_chrome_dehaan(roi_video, 30)
# print(chrome_signal)
# pos_signal = apply_pos_wang(roi_video, 30)
# print(pos_signal)

# ret, frame = cap.read()
# print("ret:", ret)
# print("frame:", None if frame is None else frame.shape)
#
# roi = roi_extractor.extract_roi(frame)
# roi = cv2.resize(roi, (64, 64))
#
# cv2.imwrite("roi.png", roi)
#
# if cv2.waitKey(1):
#     print("Released")
#
# cap.release()
# roi_extractor.release()
# cv2.destroyAllWindows()
#
# from models.phys_net import PhysNet
#
# phys_net = PhysNet()

# dataloader = DatasetLoader("small_data/", "small_data/ToHealth_RepeatDataSet.xlsx")
# dataloader.create_dataset(num_frames=10)
#
# signal_extractor = SignalExtractor()
# signal_extractor.extract_signals()
