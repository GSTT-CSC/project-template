import os
import csv
import torch

class SafeWrapperTransform:
    """
    Wrapper for MONAI transforms that logs failures and returns dummy images/labels
    instead of crashing.
    """

    def __init__(self, transform, image_size, log_file="transform_failures.csv"):
        self.transform = transform
        self.image_size = image_size
        self.log_file = log_file

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["subject_id", "error_message"])

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

            if not self.subject_already_logged(subject_id):
                with open(self.log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([subject_id, str(e)])

            # Return dummy 3D image tensor and label
            result = {
                "subject_id": subject_id,
                "image": torch.zeros((1, self.image_size, self.image_size, self.image_size), dtype=torch.float32),
                "label": torch.tensor(-1),
                "valid": False
            }
            return result
