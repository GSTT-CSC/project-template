# Wrapper for Monai Transforms so failures are logged instead of run ending

import os
import csv
import torch

class SafeWrapperTransform:
    def __init__(self, transform, image_size, log_file="transform_failures.csv"):
        self.transform = transform
        self.image_size = image_size
        self.log_file = log_file
        self.logged_ids = set()

        # Create the log file
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["subject_id", "data", "error_message"])

    def subject_already_logged(self, subject_id):
        if not os.path.exists(self.log_file):
            return False
        with open(self.log_file, "r") as f:
            return any(subject_id == row.split(",")[0] for row in f.readlines()[1:])

    def __call__(self, data):
        try:
            result = self.transform(data)
            result["valid"] = True
            return result
        except Exception as e:
            subject_id = data.get("subject_id", "UNKNOWN")

            # Check if already logged
            if not self.subject_already_logged(subject_id):
                subject_data = data.get("data", "N/A")
                with open(self.log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([subject_id, subject_data, str(e)])

            # Return dummy image, label, and data
            keys =['subject_id']
            result = {k: v for k, v in data.items() if k in keys}
            result["image"] = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
            result["label"] = torch.tensor(-1)
            result["valid"] = False
            return result
