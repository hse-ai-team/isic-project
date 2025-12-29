import numpy as np
import torch
from .image_utils import prepare_for_inference
from .simple_unet import SimpleUNet
import logging

logger = logging.getLogger(__name__)


class SegmentationModel:
    def __init__(self, model_path=None, device="cpu"):
        self.model = None
        self.target_size = (256, 256)
        self.device = torch.device(device)

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        try:
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )

            self.model = SimpleUNet()
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Модель загружена из {model_path} на устройство {self.device}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise

    def predict_segmentation(self, image_input):
        try:
            prepared_image = prepare_for_inference(image_input, self.target_size)

            if isinstance(prepared_image, np.ndarray):
                if prepared_image.shape[-1] == 3:
                    prepared_image = np.transpose(prepared_image, (0, 3, 1, 2))
                image_tensor = torch.from_numpy(prepared_image).float()
            else:
                image_tensor = prepared_image

            image_tensor = image_tensor.to(self.device)

            with torch.no_grad():
                prediction = self.model(image_tensor)

            prediction = prediction.cpu().numpy()

            if len(prediction.shape) == 4:
                prediction = prediction[0, 0]
            elif len(prediction.shape) == 3:
                prediction = prediction[0]

            binary_mask = (prediction > 0.5).astype(np.uint8) * 255

            return binary_mask

        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            raise
