---
documentid: srs-v1.0
version: 1.0

---

# System Requirements Specification - PROJECT


|             |           |
|-------------|-----------|
| Document Id | srs-v1.0  |
| Version     | 1.0       |
| Author      |           |
|             |           |


### Purpose 
This purpose of this document is to describe what the {{device.name}} application must do. 

This document is meant to be read and agreed-upon by the project owners and the CSC team during design and development.

The document also provides traceability for software requirements throughout the project. 


### Scope

This document applies to {{device.name}} release {{device.version}}.

### Definitions

| Term    | Definition                                                                                   |
|---------|----------------------------------------------------------------------------------------------|
| SRS     | Software Requirements Spec                                                                   |
| SDS     | Software Design Spec                                                                         |


### Roles and Responsibilities
 
| Role             | Name(s) | Responsibilities                                                                                 |
|------------------|---------|--------------------------------------------------------------------------------------------------|
| Development lead |         | Completing documentation <br>  Gathering requirements <br >Organising meetings with stakeholders |
| Clinical lead    |         | Organising meetings with stakeholders <br>  Providing requirements                               |
| ML lead          |         | Providing requirements for ML activities.                                                        |

##### Stakeholders

The following stakeholders contributed to the requirements gathering process. 

| Name | Role |
|------|------|
|      |      |



### Introduction


### Users

####Clinicians

#### Trust IT


### Use Environments

[EG:
#### MRI scanner control room
Once the MRI scan is completed of the prostate radiotherapy patient consented for MR only treatment planning, the radiographer in 
the MRI control area will reconstruct the images and send all series in the study to PACS. At this point their work instruction
would require them to send to the AI platform destination where the relevant series in the study would be processed for
SyntheticCT generation. ]


### Use Cases 

#### Use Case #1

#### Use Case #2

### Considerations

The following list of considerations has been compiled from relevant suggestions from ISO 13485, BSI 62304 and BSI 62366:

- input data format to be processed.
- output format of the application
- destination of the output data
- the expected application/service uptime 
- the workload e.g. number of CTs per week.
- maximum acceptable turnaround time
- number and frequency of users to be supported
- user access 
- platform for delivery
- past complaints, failure reports, adverse events of similar products
- usability and maintenance
- security controls
- workflow integration
- local population demographics, to be reflected in training data.
- minimum sensitivity, specificity, true/false positive/negative rates.
- turnaround time for inference
- expected frequency of retraining.
- training data bias to acknowledge
- quality objectives for the project - e.g. quality metrics, ranges and thresholds
- contents of surveillance plan including metric to monitor once the application is deployed
- applicable regulatory requirements
- training requirements, such as workshops, training, documents etc.
- safety considerations for patients and staff
- information derived from similar/previous designs
- functional and capability requirements
- interfaces between systems
- software driven alarms, warnings and operator messages
- security requirements
- data definitions and database requirements
- installation and acceptance requirements of the delivered software at operation and maintenance site (* although this 
is GSTT only for projects built under this QMS)
- requirements related to methods of operation and maintenance
- requirements of IT network, Trust IT integration
- user maintenance requirements
- software update requirements

### User requirements

| Reference | Requirement title                                                                                           | Requirements Description                                                                                                                                                                        |                                                                                                                                           
|-----------|-------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| SRS-001   | Shall accept MRI DICOM images.                                                                              | Shall accept only MRI DICOM images from set list of MR protocols. <br> Accepts DICOM Images from certain machines with radiotherapy treatment couch.<br> Accepts a full MRI full DICOM Study.   | 


| ID     | Title   | Description   |
|--------|---------|---------------|
{%- for requirement in requirements.requirements %}
| {{ requirement.id }} | {{ requirement.title }} | {{requirement.description}} |
{%- endfor %}
