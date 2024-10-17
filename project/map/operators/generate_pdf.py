"""This is an example operator to implement PDF generator as a way to visualise segmentation and classification result overlaid on a DICOM image"""

import os.path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import monai.deploy.core as md
from monai.deploy.core import (
    DataPath,
    ExecutionContext,
    InputContext,
    IOType,
    Operator,
    OutputContext,
)
from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries

import PIL as pil
from collections import UserDict
from typing import List
import reportlab.platypus as pl
from reportlab.platypus import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import toColor
from reportlab.platypus import Frame, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


from reportlab.lib.fonts import tt2ps
from reportlab.rl_config import canvas_basefontname as _baseFontName

_baseFontNameB = tt2ps(_baseFontName, 1, 0)
_baseFontNameI = tt2ps(_baseFontName, 0, 1)
_baseFontNameBI = tt2ps(_baseFontName, 1, 1)

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Frame, Image, Paragraph


@md.input("study_selected_series_list", List[StudySelectedSeries], IOType.IN_MEMORY)
@md.input("output_udict", UserDict, IOType.DISK)
@md.output("pdf_file", DataPath, IOType.DISK)
@md.env(pip_packages=["monai"])
class GeneratePDFOperator(Operator):
    """
    Creates a PDF of the results from the image classifier.
    Results are input for the DICOM PDF Encapsulator
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(
        self,
        op_input: InputContext,
        op_output: OutputContext,
        context: ExecutionContext,
    ):
        """
        Reads in original selected_series, and results from classifier, to generate a PDF report.
        Sets operator output to path of generated dicom.

        """

        # Read inputs
        original_image = op_input.get("study_selected_series_list")
        if not original_image or len(original_image) < 1:
            raise ValueError("Missing input, list of 'StudySelectedSeries'.")

        # Extract selected image sop_instance
        study_selected_series = original_image[0]
        selected_series = study_selected_series.selected_series
        sop_instance = (
            selected_series[0].series.get_sop_instances()[0].get_native_sop_instance()
        )

        # Read classifier results
        result_dict = op_input.get("output_udict")
        if not result_dict or len(result_dict) < 3:
            raise ValueError("Missing input, Dict of results.")

        # Get output directory
        output_folder = op_output.get_group_path()

        # Return pdf of results
        pdf_path = self.generate_report_pdf(sop_instance, result_dict, output_folder)

        op_output.set(pdf_path, "pdf_file")

    # Generates the PDF
    def generate_report_pdf(self, sop_instance, results_dict, output_folder):

        pdf_filename = "report_pdf.pdf"

        canv = Canvas(pdf_filename, pagesize=A4)

        # get dicom data from study
        ds_meta = sop_instance
        patient_name = str(ds_meta["PatientName"].value)
        # patient_name = "John DOnt
        dob = ds_meta["PatientBirthDate"].value
        # dob = "01/01/2022"
        pat_id = ds_meta["PatientID"].value
        # pat_id = "1234567A"
        # gender = ds_meta['PatientSex'].value
        # consultant = ds_meta['ReferringPhysicianName'].value
        study_description = ds_meta["StudyDescription"].value
        # series_description = ds_meta['SeriesDescription'].value
        # series_uid = ds_meta['SeriesInstanceUID'].value
        accession_number = ds_meta["AccessionNumber"].value
        xray_date = ds_meta["SeriesDate"].value

        if ds_meta["StudyTime"].value == "":
            xray_time = "Time not available"
        else:
            xray_time = ds_meta["StudyTime"].value

        # protocol_name = ds_meta['ProtocolName']

        # Get application data
        result = results_dict["result"]
        advice = results_dict["advice"]
        error_message = results_dict["error_message"]

        b_box = results_dict["bounding_box"]
        # list of [x1, y1, x2, y2, confidence]

        # reformat time
        reversed_date = (
            xray_date[6:] + "/" + xray_date[4] + xray_date[5] + "/" + xray_date[:4]
        )
        reversed_dob = dob[6:] + "/" + dob[4] + dob[5] + "/" + dob[:4]
        xray_date_and_time = str(reversed_date) + " " + str(xray_time)

        # set variables for sizes of things
        a4_height = A4[1]
        a4_width = A4[0]
        side_margins = 10
        banner_height = 60
        banner_frames_y = a4_height - banner_height
        description_height = 25
        patient_name_frame_height = 120
        patient_dob_frame_height = 60
        patient_info_width = a4_width * 0.48
        study_frame_height = 60
        xray_frame_height = a4_height * 0.45
        xray_frame_width = (a4_width * 0.5) - side_margins
        disclaimer_frame_height = a4_height / 8

        # set text styles
        styles = getSampleStyleSheet()
        style_h1 = styles["Heading1"]
        n_style = ParagraphStyle(
            name="nameStyle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=30,
            textColor="#bdd7ee",
            wordWrap="LTR",
            leading=24 * 1.2,
        )
        pt_style = ParagraphStyle(
            name="patientStyle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=24,
            textColor="#bdd7ee",
            leading=24,
        )
        study_details_style = ParagraphStyle(
            name="studyDetailsStyle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=20,
            textColor="#bdd7ee",
            leading=20,
        )
        model_description_style = ParagraphStyle(
            name="modelStyle",
            parent=styles["Title"],
            fontName="Helvetica",
            fontSize=12,
            textColor="#bdd7ee",
            leading=0,
        )
        disclaimer_style = ParagraphStyle(
            name="disclaimerStyle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            textColor="#ffc000",
        )
        misc_style = ParagraphStyle(
            name="miscStyle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=12,
            textColor="#bdd7ee",
            # leading = 14,
            spaceAfter=8,
        )
        # changing the patientNameFrameHeight/text size based on name length
        pt_name_length = len(patient_name)

        if pt_name_length <= 18:
            patient_name_frame_height = 40
            patient_name = [Paragraph(patient_name, style=n_style)]
        elif pt_name_length <= 37:
            patient_name_frame_height = 80
            patient_name = [Paragraph(patient_name, style=n_style)]
        elif pt_name_length <= 47:
            patient_name_frame_height = 90
            patient_name = [Paragraph(patient_name, style=pt_style)]
        else:
            patient_name_frame_height = 100
            patient_name = [Paragraph(patient_name, style=pt_style)]

        # set banner frames (height of page 842 and width 595 in pixels)
        banner_frame = ColorFrame(
            1,
            banner_frames_y,
            198,
            banner_height,
            showBoundary=0,
            background="white",
            bottomPadding=1,
            topPadding=1,
        )

        banner_frame_2 = ColorFrame(
            198,
            banner_frames_y,
            200,
            banner_height,
            showBoundary=0,
            background="white",
            bottomPadding=0,
            topPadding=10,
        )

        banner_frame_3 = ColorFrame(
            397,
            banner_frames_y,
            198,
            banner_height,
            showBoundary=0,
            background="white",
            bottomPadding=1,
            topPadding=1,
            rightPadding=10,
        )

        description_frame = Frame(
            side_margins,
            (banner_frames_y - 25),
            (a4_width - (side_margins * 2)),
            description_height,
            showBoundary=0,
        )

        # set results colorFrame
        result_frame = ColorFrame(
            side_margins,
            a4_height / 4,
            a4_width - (side_margins * 2),
            a4_height / 6,
            showBoundary=1,
            background="#bdd7ee",
        )

        # set patient details and X-ray image Frames
        patient_name_frame = Frame(
            side_margins,
            (banner_frames_y - (patient_name_frame_height + description_height)),
            patient_info_width,
            (patient_name_frame_height),
            showBoundary=0,
            topPadding=0,
            leftPadding=0,
            rightPadding=0,
            bottomPadding=0,
        )

        patient_dob_frame = Frame(
            side_margins,
            (
                banner_frames_y
                - (
                    patient_name_frame_height
                    + description_height
                    + patient_dob_frame_height
                )
            ),
            patient_info_width,
            patient_dob_frame_height,
            showBoundary=0,
            leftPadding=0,
        )

        patient_id_frame = Frame(
            side_margins,
            banner_frames_y
            - (
                patient_dob_frame_height
                + description_height
                + (patient_dob_frame_height * 2)
            ),
            patient_info_width,
            patient_dob_frame_height,
            showBoundary=0,
            leftPadding=0,
        )

        study_des_frame = Frame(
            side_margins,
            banner_frames_y
            + (study_frame_height * 2)
            - (xray_frame_height + description_height),
            patient_info_width,
            study_frame_height,
            showBoundary=0,
            leftPadding=0,
        )

        study_day_frame = Frame(
            side_margins,
            (
                banner_frames_y
                + study_frame_height
                - (xray_frame_height + description_height)
            ),
            patient_info_width,
            study_frame_height,
            showBoundary=0,
            leftPadding=0,
        )

        study_id_frame = Frame(
            side_margins,
            (banner_frames_y - (xray_frame_height + description_height)),
            patient_info_width,
            study_frame_height,
            showBoundary=0,
            leftPadding=0,
        )

        xray_frame = Frame(
            a4_width * 0.5,
            (banner_frames_y - (xray_frame_height + description_height)),
            (xray_frame_width),
            xray_frame_height,
            showBoundary=0,
            rightPadding=0,
            leftPadding=0,
            topPadding=0,
            bottomPadding=0,
        )

        # set disclaimer and misc frames
        disclaimer_frame = Frame(
            side_margins,
            (a4_height / 4 - disclaimer_frame_height),
            a4_width - (side_margins * 2),
            disclaimer_frame_height,
            showBoundary=0,
        )

        model_info_frame = Frame(
            side_margins, 0, a4_width * 0.35, 100, showBoundary=0, leftPadding=0
        )

        further_info_frame = Frame(
            a4_width * 0.65, 0, a4_width * 0.35, 100, showBoundary=0, rightPadding=10
        )

        # set background color
        canv.setFillColor("#1f4e79")
        canv.rect(0, 0, a4_width, a4_height, fill=1)

        # content
        description = [
            Paragraph(
                "ScaphX determines the presence of a Scaphoid fracture \
                on AP/PA views of the scaphoid",
                style=model_description_style,
            )
        ]

        csc_logo = [
            Image(
                os.path.dirname(__file__) + "/images/cscLogo.png", width=100, height=40
            )
        ]

        gstt_logo = [
            Image(
                os.path.dirname(__file__) + "/images/gsttLogo.png",
                width=140,
                height=58,
                hAlign="RIGHT",
            )
        ]

        patient_dob = [
            Paragraph("<strong>DOB:</strong>", style=study_details_style),
            Paragraph(reversed_dob, style=study_details_style),
        ]

        patient_id = [
            Paragraph("<strong>Patient ID:</strong>", style=study_details_style),
            Paragraph(pat_id, style=study_details_style),
        ]

        study_des = [
            Paragraph("<strong>Study Description:</strong>", style=study_details_style),
            Paragraph(study_description, style=study_details_style),
        ]

        study_day = [
            Paragraph("<strong>Study Date:</strong>", style=study_details_style),
            Paragraph(xray_date_and_time, style=study_details_style),
        ]

        study_id = [
            Paragraph("<strong>Accession Number:</strong>", style=study_details_style),
            Paragraph(accession_number, style=study_details_style),
        ]

        results = [
            Paragraph(f"Result: {result}", style=style_h1),
            Paragraph(f"Action: {advice}", style=style_h1),
            Paragraph(f"Error message: {error_message}", style=style_h1),
        ]

        disclaimer = [
            Paragraph(
                "These are preliminary results only. \
                Please await a finalised report. \
                For any diagnostic queries, \
                    please discuss with an MSK Radiologist via the Radiology department.",
                style=disclaimer_style,
            )
        ]

        further_info = [
            Paragraph(
                "For further information on how to use this tool \
                                  or to report a problem:",
                style=misc_style,
            ),
        ]

        model_info = [
            Paragraph("ScaphX model version: v1.0.0", style=misc_style),
            Paragraph("ScaphX app version: v1.0.0", style=misc_style),
        ]

        xray = self.create_xray_jpeg(
            sop_instance.pixel_array, b_box, xray_frame_width, xray_frame_height
        )

        # add Flowables to page

        banner_frame_2.addFromList(csc_logo, canv)
        banner_frame_3.addFromList(gstt_logo, canv)
        description_frame.addFromList(description, canv)
        result_frame.addFromList(results, canv)
        patient_name_frame.addFromList(patient_name, canv)
        patient_dob_frame.addFromList(patient_dob, canv)
        patient_id_frame.addFromList(patient_id, canv)
        study_des_frame.addFromList(study_des, canv)
        study_day_frame.addFromList(study_day, canv)
        study_id_frame.addFromList(study_id, canv)
        xray_frame.addFromList(xray, canv)
        disclaimer_frame.addFromList(disclaimer, canv)
        model_info_frame.addFromList(model_info, canv)
        further_info_frame.addFromList(further_info, canv)

        # create page

        canv.showPage()
        canv.save()

        return DataPath(os.path.join(os.getcwd(), pdf_filename))

    def create_xray_jpeg(self, pixel_data, b_box, xray_frame_width, xray_frame_height):
        """generate the jpeg of X-ray with bounding box overlay"""

        image = pil.Image.fromarray(pixel_data)
        fig, ax = plt.subplots()
        ax.imshow(image, cmap="gray")
        rect = patches.Rectangle(
            (b_box[0], b_box[1]),
            b_box[2] - b_box[0],
            b_box[3] - b_box[1],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        plt.axis("off")
        fig.savefig("dummy_im.png", bbox_inches="tight", pad_inches=0)

        return [
            pl.flowables.Image(
                "dummy_im.png", width=xray_frame_width, height=xray_frame_height
            )
        ]


class ColorFrame(Frame):
    """Extends the reportlab Frame with a background color."""

    def __init__(self, *args, **kwargs):
        self.background = kwargs.pop("background")
        super().__init__(*args, **kwargs)

    def draw_background(self, canv):
        color = toColor(self.background)
        canv.saveState()
        canv.setFillColor(color)
        canv.rect(
            self._x1,
            self._y1,
            self._x2 - self._x1,
            self._y2 - self._y1,
            stroke=0,
            fill=1,
        )
        canv.restoreState()

    def addFromList(self, drawlist, canv):
        if self.background:
            self.draw_background(canv)
        Frame.addFromList(self, drawlist, canv)

    # back to main script
