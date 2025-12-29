import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class ImageProcessingError(Exception):
    pass


def load_image(image_path, target_size):
    try:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Файл не найден: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(
                f"Не удалось загрузить изображение. Возможно, файл поврежден: {image_path}"
            )

        if image.size == 0:
            raise ValueError(f"Изображение пустое: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

        return image_resized

    except FileNotFoundError as e:
        raise ImageProcessingError(f"Файл изображения не найден: {image_path}") from e
    except Exception as e:
        raise ImageProcessingError(f"Ошибка обработки изображения: {e}") from e


def load_image_from_bytes(image_bytes, target_size):
    try:
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != "RGB":
            image = image.convert("RGB")

        image_np = np.array(image)

        image_resized = cv2.resize(
            image_np, target_size, interpolation=cv2.INTER_LINEAR
        )

        return image_resized

    except Exception as e:
        raise ImageProcessingError(
            f"Ошибка обработки изображения из байтов: {e}"
        ) from e


def load_mask(mask_path, target_size):
    try:
        mask_path = Path(mask_path)

        if not mask_path.exists():
            raise FileNotFoundError(f"Файл маски не найден: {mask_path}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise ValueError(f"Не удалось загрузить маску: {mask_path}")

        if mask.size == 0:
            raise ValueError(f"Маска пустая: {mask_path}")

        mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask_resized > 127).astype(np.float32)

        return mask_binary

    except FileNotFoundError as e:
        raise ImageProcessingError(f"Файл маски не найден: {mask_path}") from e
    except Exception as e:
        raise ImageProcessingError(f"Ошибка загрузки маски: {e}") from e


def normalize_image(image):
    return image.astype(np.float32) / 255.0


def prepare_for_inference(image_input, target_size):
    if isinstance(image_input, (str, Path)):
        image = load_image(image_input, target_size)
    elif isinstance(image_input, bytes):
        image = load_image_from_bytes(image_input, target_size)
    else:
        raise ValueError(f"Unsupported image_input type: {type(image_input)}")

    image_normalized = normalize_image(image)
    image_batch = np.expand_dims(image_normalized, axis=0)

    return image_batch
