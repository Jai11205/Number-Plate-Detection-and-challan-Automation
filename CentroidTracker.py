import numpy as np
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, max_disappeared=30):
        # Dictionary to keep track of object IDs and their current centroids
        self.objects = OrderedDict()
        # Dictionary to track how many consecutive frames an object has been "lost"
        self.disappeared = OrderedDict()
        self.next_object_id = 0
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        # Assign a new ID to a new centroid
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        # Remove an object if it has been lost for too long
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        # rects: list of bounding boxes from YOLO in format (startX, startY, endX, endY)

        if len(rects) == 0:
            # If no detections, increment disappeared count for all existing objects
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Initialize an array of input centroids for the current frame
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        # If no objects are currently being tracked, register all new inputs
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Pure NumPy computation of the distance matrix between existing and new centroids
            # We use broadcasting to calculate distances efficiently without loops
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

            # Find the smallest value in each row and sort row indexes based on their min values
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            # Loop over the combinations of (row, column) index tuples
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                # Update the object's centroid and reset its disappeared count
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Compute row and col indices we haven't examined yet
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # Handle existing objects that were lost in this frame
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Register any brand new input centroids
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects