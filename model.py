from ultralytics import YOLO

class MaskModel(YOLO):
    def __int__(self, model):
        super().__init__()
        self.model_name = model

    def train(self, checkpoint_path, data_path, epochs, imgsize, batch, device):
        model = YOLO(self.model_name)
        model = YOLO(checkpoint_path)
        model.train(data=data_path, epochs=epochs, imgsz=imgsize, batch=batch, device=batch)
        model.export(format="onnx")

    def val(checkpoint_path):
        model = YOLO(checkpoint_path)
        metrics = model.val()
        print(f"MAP : {metrics.box.map}")
        print(f"MAP-50 : {metrics.box.map50}")
        print("Mean precision:", metrics.box.mp)
        print("Mean recall:", metrics.box.mr)
        print("Fitness:", metrics.box.fitness())