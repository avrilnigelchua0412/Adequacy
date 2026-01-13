import onnxruntime as ort
import numpy as np
import json, yaml

class TileInferencer:
    def __init__(self, model_path, input_spec_path, preprocessing_path, provider="CPUExecutionProvider"):
        # Load configs
        with open(input_spec_path) as f:
            self.input_spec = json.load(f)

        with open(preprocessing_path) as f:
            self.pre_cfg = yaml.safe_load(f)

        # Init ONNX session
        self.session = ort.InferenceSession(model_path, providers=[provider])
        self.input_name = self.session.get_inputs()[0].name

        # Cache input shape
        self.input_h = self.input_spec["input_size"][2]
        self.input_w = self.input_spec["input_size"][3]

    
    def preprocess(self, tile_img):
            """
            tile_img: np.ndarray (H, W, 3), uint8, RGB
            """
            assert tile_img.shape[:2] == (self.input_h, self.input_w), \
                f"Expected {(self.input_h, self.input_w)}, got {tile_img.shape[:2]}"

            img = tile_img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)  # HWC → CHW
            img = np.expand_dims(img, axis=0)  # CHW → BCHW
            return img
        
    def infer(self, tile_img):
        inp = self.preprocess(tile_img)
        outputs = self.session.run(None, {self.input_name: inp})
        return outputs  # raw predictions only