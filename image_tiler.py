import cv2
import numpy as np
from typing import Iterator, Tuple


class ImageTiler:
    """
    Splits a large image into overlapping tiles and tracks spatial metadata.

    Each yielded tile is:
      - Padded to (tile_size, tile_size)
      - Associated with its (x0, y0) origin in the original image
      - Assigned a stable tile_id
    """

    def __init__(self, tile_size: int = 512, overlap: float = 0.25):
        assert 0.0 <= overlap < 1.0, "Overlap must be in [0, 1)"
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = int(tile_size * (1 - overlap))

    @staticmethod
    def pad_image(image: np.ndarray, desired: int = 512) -> np.ndarray:
        """
        Pads image to desired size (bottom-right padding).
        """
        h, w, _ = image.shape
        pad_bottom = max(desired - h, 0)
        pad_right = max(desired - w, 0)

        if pad_bottom > 0 or pad_right > 0:
            image = cv2.copyMakeBorder(
                image,
                0, pad_bottom,
                0, pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=[255, 255, 255]
            )
        return image

    def tile(
        self, original_image: np.ndarray
    ) -> Iterator[Tuple[np.ndarray, int, int, str]]:
        """
        Generator over image tiles.

        Yields:
            tile      : np.ndarray (tile_size, tile_size, 3)
            x0, y0    : top-left coordinate in original image
            tile_id   : stable string id
        """
        img_h, img_w, _ = original_image.shape

        for row_idx, y0 in enumerate(range(0, img_h, self.stride)):
            for col_idx, x0 in enumerate(range(0, img_w, self.stride)):
                tile_id = f"{row_idx}_{col_idx}"

                x1 = min(x0 + self.tile_size, img_w)
                y1 = min(y0 + self.tile_size, img_h)

                tile = original_image[y0:y1, x0:x1]
                tile = self.pad_image(tile, self.tile_size)

                yield tile, x0, y0, tile_id