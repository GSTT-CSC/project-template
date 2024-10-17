---
qms_version: 2.2.0
sop_id: CSC PR.001
sop_version: 2.0.1
template_id: CSC F.018
template_version: 2.0.1
record_version:
record_id: MDR-001
title: Medical Device Regulation Classification
---

# Medical Device Classification 

## General 

|                           |               |
|---------------------------|---------------|
| **Template ID**           | CSC F.018     | 
| **Template Version**      | 2.0.1         |
| **QMS Version**           | 2.2.0         |
| **SOP ID**                | CSC PR.001    |
| **SOP Version**           | 2.0.1         |
| **Regulatory References** |               |



|              |              |
|--------------|--------------|
| **Author**   |              |
| **Approval** |              |
## Purpose

This document describes the medical device classification for {{device.name}} release {{device.version}}. 

## Scope

This document applies to {{device.name}} release {{device.version}}.

## Definitions

| Item | Definition |
|------|------------|
| EU MEDDEV     | European Union Medical Device Directives            |
| ESAPI 	| Eclipse Scripting Application Interface	|
| TPS	| Treatment Planning System |
| MHRA | Medicines and Healthcare products Regulatory Agency |

## Roles and Responsibilities

| Role | Responsibilities |
|------|------------------|
|Clinical Lead      | Advise on safe use and potential misuse of the application                 |
|Development Lead      | Complete the classification of the application                 |
|Clinical Safety Officer      | Complete the classification of the application                  |

## MHRA Classification and rationale

| MHRA Classification | {{device.mhra_class}} |
|---------------------|-------------------------------|

## Medical software classification and rationale - BS EN 62304:2020

| 62304 Classification | {{device.BS62304_class}} |
|----------------------|-------------|
- The PlanCheck script meets the definition of ‘software’ and ‘stand-alone software’ from EU MEDDEV 2.1/6 (European Commission, 2016):
  - Software: a set of instructions that processes input data and creates output data.
  - Stand alone software: software which is not incorporated in a medical device at the time of its placing on the market or its making available. Although the PlanCheck script uses ESAPI, which is part of the Eclipse TPS (a medical device), the PlanCheck script itself does not just use ESAPI functions, and was not incorporated at the time of Eclipse being placed on the market.
  - Following the MHRA guidance on qualification of software as a medical device (MHRA, 2023):
  - The software is a computer program.
  - The software does not have a medical purpose. It performs an automated check of aspects of a radiotherapy treatment plan, but does not modify the plan itself, and replaces no operator actions in the treatment pathway, effectively acting as an additional check.
  - The software works in combination with one or more devices (Eclipse).
  - The software does not enable the function of the device (is not an accessory). Treatment plans can be produced in Eclipse without the use of the PlanCheck script, and no workflows need to be modified if the script cannot be used for a particular treatment plan.
The PlanCheck script is NOT a medical device.
It will be developed under the Radiotherapy Physics quality system for in-house software development.


