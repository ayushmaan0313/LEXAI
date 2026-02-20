"""Cell segmentation for GNN graph construction.

Uses threshold-based segmentation in HSV color space combined with
morphological operations to isolate white blood cells (WBCs) from
blood smear images.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CellInfo:
    """Information about a single detected cell."""
    centroid: Tuple[int, int]       # (x, y) center coordinates
    bbox: Tuple[int, int, int, int] # (x, y, w, h)
    area: float                      # pixel area
    contour: np.ndarray              # OpenCV contour
    mask: Optional[np.ndarray] = None  # Binary mask for this cell


class CellSegmenter:
    """
    Threshold-based white blood cell segmentation.

    Uses HSV color space thresholding + morphological operations
    to detect WBC nuclei (which appear dark purple/blue in
    Wright-Giemsa or H&E stained smears).
    """

    def __init__(
        self,
        min_cell_area: int = 200,
        max_cell_area: int = 50000,
        # HSV thresholds for WBC nuclei (purple/dark blue)
        hsv_lower: Tuple[int, int, int] = (100, 40, 40),
        hsv_upper: Tuple[int, int, int] = (170, 255, 255),
        morph_kernel_size: int = 5,
    ):
        self.min_cell_area = min_cell_area
        self.max_cell_area = max_cell_area
        self.hsv_lower = np.array(hsv_lower)
        self.hsv_upper = np.array(hsv_upper)
        self.morph_kernel_size = morph_kernel_size

    def segment(
        self, image: np.ndarray
    ) -> Tuple[List[CellInfo], np.ndarray]:
        """
        Segment WBCs from a blood smear image.

        Args:
            image: (H, W, 3) uint8 BGR image

        Returns:
            cells: List of CellInfo for each detected cell
            mask: Full binary segmentation mask (H, W)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Threshold for WBC nuclei
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Fill holes
        mask_filled = mask.copy()
        h, w = mask.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(mask_filled, flood_mask, (0, 0), 255)
        mask_filled = cv2.bitwise_not(mask_filled)
        mask = mask | mask_filled

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter and extract cell info
        cells = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_cell_area or area > self.max_cell_area:
                continue

            # Centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Bounding box
            x, y, bw, bh = cv2.boundingRect(contour)

            # Per-cell mask
            cell_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(cell_mask, [contour], 0, 255, -1)

            cells.append(CellInfo(
                centroid=(cx, cy),
                bbox=(x, y, bw, bh),
                area=area,
                contour=contour,
                mask=cell_mask,
            ))

        return cells, mask

    def extract_cell_features(
        self, image: np.ndarray, cells: List[CellInfo], feature_dim: int = 64
    ) -> np.ndarray:
        """
        Extract simple visual features for each detected cell.

        Features include color histograms, shape descriptors, and 
        intensity statistics.

        Args:
            image: (H, W, 3) BGR image
            cells: List of detected CellInfo
            feature_dim: Target feature dimension

        Returns:
            features: (N, feature_dim) array of cell features
        """
        if not cells:
            return np.zeros((0, feature_dim), dtype=np.float32)

        features = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        for cell in cells:
            feat = []
            cell_mask = cell.mask

            # Color histogram features (per channel in HSV, 8 bins each = 24)
            for ch in range(3):
                hist = cv2.calcHist(
                    [hsv], [ch], cell_mask,
                    [8], [0, 256]
                )
                hist = hist.flatten() / (hist.sum() + 1e-6)
                feat.extend(hist)

            # Intensity statistics (4 features)
            masked_gray = gray[cell_mask > 0]
            if len(masked_gray) > 0:
                feat.extend([
                    masked_gray.mean() / 255.0,
                    masked_gray.std() / 255.0,
                    np.percentile(masked_gray, 25) / 255.0,
                    np.percentile(masked_gray, 75) / 255.0,
                ])
            else:
                feat.extend([0.0, 0.0, 0.0, 0.0])

            # Shape features (8 features)
            area = cell.area
            perimeter = cv2.arcLength(cell.contour, True) + 1e-6
            circularity = 4 * np.pi * area / (perimeter ** 2)
            x, y, bw, bh = cell.bbox
            aspect_ratio = bw / (bh + 1e-6)
            extent = area / (bw * bh + 1e-6)

            # Hu moments (7 values)
            moments = cv2.moments(cell.contour)
            hu = cv2.HuMoments(moments).flatten()
            # Log-transform Hu moments for better scale
            hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

            feat.extend([
                area / 10000.0,  # normalized area
                circularity,
                aspect_ratio,
                extent,
            ])
            feat.extend(hu[:4].tolist())  # First 4 Hu moments

            features.append(feat)

        features = np.array(features, dtype=np.float32)

        # Pad or truncate to feature_dim
        current_dim = features.shape[1]
        if current_dim < feature_dim:
            padding = np.zeros(
                (features.shape[0], feature_dim - current_dim),
                dtype=np.float32
            )
            features = np.concatenate([features, padding], axis=1)
        elif current_dim > feature_dim:
            features = features[:, :feature_dim]

        return features

    def compute_blast_percentage(self, cells: List[CellInfo]) -> float:
        """
        Estimate blast cell percentage based on morphological features.

        Blast cells tend to be larger with higher nucleus-to-cytoplasm ratio.
        This is a simplified heuristic — in production, a trained classifier
        would be used.

        Returns:
            Estimated blast percentage (0-100)
        """
        if not cells:
            return 0.0

        # Heuristic: cells larger than median area are potential blasts
        areas = [c.area for c in cells]
        median_area = np.median(areas)
        threshold = median_area * 1.3

        blast_count = sum(1 for a in areas if a > threshold)
        return (blast_count / len(cells)) * 100.0
