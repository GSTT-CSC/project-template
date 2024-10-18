import yaml
from deepdiff import DeepDiff


def perform_audit():
    """ Performs a series of documentation checks for iso13485 compliance"""
    check_qms_templates()
    return


def kpi_check_num_of_hazards():
    return


def kpi_check_average_ticket_ages():
    return


def check_stakeholders():
    return


def verification_test_coverage():
    return

def validation_tests_completion():
    return

def classification_applied():
    return

def 

def check_qms_templates():
    """"""
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
    documentation_dir = ""
    data_files_dir = ""
    templates_dir = ""
    perform_audit()

