from data.dataloader import DatasetLoader
from data.phys_net_dataset import PhysNetDataset
from preprocessing.signal_extractor import SignalExtractor
from evaluation.features.signal_dataset import SignalDataset
from supervised_models.model_factory import build_model
from supervised_models.supervised_model_trainer import SupervisedModelTrainer 
from evaluation.features.raw_extractor import RawExtractor
from evaluation.model_trainer import ModelTrainer
from evaluation.models.cnn_regressor_model import CNNRegressor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

dataloader = DatasetLoader("small_data/", "small_data/ToHealth_RepeatDataSet.xlsx")
dataloader.create_dataset(num_frames=100)

signal_extractor = SignalExtractor()
signal_extractor.extract_signals(dataloader.video_list, 30)

chrome_signal = signal_extractor.chrome_dehaan_signal
pos_signal = signal_extractor.pos_wang_signal

model = CNNRegressor(T=len(dataloader.target_list))

model_trainer = ModelTrainer(pos_signal[0], dataloader.target_list, model)
model_trainer.train_model(50, 16, 1e-3, 0.2, True)

model_trainer.save_model("cnn_regressor_model.pth")

# dataset = PhysNetDataset(
#     target_list=dataloader.target_list,
#     signal_extractor=signal_extractor
# )
#
# train_loader = DataLoader(
#     dataset,
#     batch_size=1,
#     shuffle=True,
#     drop_last=True
# )
#
# model = build_model()
#
# trainer = SupervisedModelTrainer(
#     model=model
# )
#
# trainer.fit(
#     train_loader=train_loader,
#     epochs=10,
# )

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
