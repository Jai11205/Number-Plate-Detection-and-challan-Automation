import cv2
import numpy as np

class PerspectiveTransformer:
    def __init__(self, source_points: np.ndarray, real_world_width: float, real_world_length: float):
        # Define the target rectangle in meters or scaled units
        target_points = np.array([
            [0, 0],
            [real_world_width, 0],
            [real_world_width, real_world_length],
            [0, real_world_length]
        ], dtype=np.float32)

        # Calculate the 3x3 Homography Matrix M
        self.M = cv2.getPerspectiveTransform(source_points.astype(np.float32), target_points)

    def transform_point(self, point: tuple) -> tuple:
        # Create the homogeneous coordinate [x, y, 1]
        p = np.array([point[0], point[1], 1.0])

        # Apply matrix multiplication
        transformed = np.dot(self.M, p)

        # Normalize to get the final real-world coordinate
        x_real = transformed[0] / transformed[2]
        y_real = transformed[1] / transformed[2]

        return (x_real, y_real)