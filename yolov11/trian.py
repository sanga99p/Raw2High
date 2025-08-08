from ultralytics import YOLO


model = YOLO("yolov11/pretrained_yolo/yolo11n.pt")

# Train the model
train_results = model.train(
    data="configs/dataset.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=600,  # training image size
    # device="device=0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)



# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("/media/EXT0/sanga/studio_project/zombie_cat_detection_640/export/dataset/val/images/20.png")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model

# Train the model
results = model.train(data="/media/EXT0/sanga/studio_project/zombie_cat_detect_augment_640/zombie_cat_aug_640.yaml", epochs=100, imgsz=416)