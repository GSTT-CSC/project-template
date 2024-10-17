import os
import yaml
from deepdiff import DeepDiff

def perform_audit():
    check_qms_templates()
    return

def check_qms_templates():

    # get qms template manifest

    qms_manifest_filename = ""
    current_project_manifest_filename = ""

    # compare manifest
    with (open(qms_manifest_filename, 'r') as qms_manifest,
          open(current_project_manifest_filename, 'r') as current_project_manifest):

        qms_manifest_yaml = yaml.safe_load(qms_manifest)
        current_project_manifest_yaml = yaml.safe_load(current_project_manifest)
        diff = DeepDiff(qms_manifest_yaml, current_project_manifest_yaml, ignore_order=True)
        # create output text file
        with open('qms_template_manifest_diff.txt', 'w') as f:
                f.write(diff)

    return


if __name__ == "__main__":
    perform_audit()
