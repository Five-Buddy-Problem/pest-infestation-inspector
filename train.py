from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolo11n.pt")

results = model.train(data="C:\\Users\\Atabay\\PycharmProjects\\datasets\\eupoecilia_ambiguella\\data.yaml", epochs=50)

# Evaluate the model's performance
results = model.val()

# Perform object detection on an image
result = model("species_images/eupoecilia_ambiguella/eupoecilia_ambiguella_128.jpg")
result[0].show()

# Export the model to ONNX format
success = model.export(format="onnx")
model.export(format="torchscript")
model.export(format="tensorflow")