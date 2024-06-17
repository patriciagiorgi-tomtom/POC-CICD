from mmengine.config import Config
import os

def get_classifier_classes_from_config(config_path) -> dict:
    cls_config = Config.fromfile(config_path)
    cls_config = cls_config.to_dict()
    ann_path = cls_config["val_dataloader"]["dataset"]["ann_file"]
    if not os.path.exists(ann_path):
        ann_path = os.path.join(
            cls_config["val_dataloader"]["dataset"]["data_root"], ann_path
        )
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Annotation file {ann_path} not found")
    # read ann_path which is a txt file
    with open(ann_path, "r") as file:
        data = file.readlines()
    class_id_to_name_mapping = dict()
    for line in data:
        img_path, class_id = line.strip().split(" ")
        class_name = img_path.split("/")[0]
        class_id_to_name_mapping[class_id] = class_name

    return class_id_to_name_mapping

def get_config_as_dict(config_path) -> dict:
    config = Config.fromfile(config_path)
    config = config.to_dict()
    return config

def get_path_to_the_final_checkpoint(folder: str) -> str:
    # get all folders that start with model_
    model_folders = [x for x in os.listdir(folder) if x.startswith("model_")]
    # sort first by epoch and then by iteration
    model_folders.sort(key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))
    # return the last folder
    return os.path.join(folder, model_folders[-1], "state_dict.pth")