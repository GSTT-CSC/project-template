---
documentid: v&v-v1.0
version: 1.0

---

# Verification and Validation


|             |          |
|-------------|----------|
| Document Id | v&v-v1.0 |
| Version     | 1.0      |
| Author      | n        |
|             |          |


### Purpose 

This document describes a set of activities which will be used for the verification and validation of the software

### Scope

This document applies to {{device.name}} release {{device.version}}.

### Definitions

| Term    | Definition                                        |
|---------|---------------------------------------------------|
| SRS     | Software Requirements Spec                        |
| SDS     | Software Design Spec                              |


### Roles and Responsibilities

| Role             | Name | Responsibilities                                                                      |
|------------------|------|---------------------------------------------------------------------------------------|
| Development lead |      | Completing documentation <br>  Generating Unit tests <br >Performing validation tests |
| ML Lead          |      | Reviewing ML activities                                                               |
| Clinical Lead    |      | Review Verification and validation test                                               |
| CSO              |      | Ensure Risks controls and verified/validated                                          |


### Related Documents

The following documents are related to design and development activities and contains records 

| Document id                               | Purpose                                                                                                                                                                                                                                                                                                                                                      |
|-------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Software Requirements Specification (SRS) | Describes what the software needs to accomplish.  It is largely written by the project lead during the requirements gathering and analyiss stage, and is reviewed by the project lead during the release.  Software developers may clarify and extend the document slightly during the unit implementation and testing activity                              |
| Clinical Risk Management plan (CRMP)      |                                                                                                                                                                                                                                                                                                                                                              |
| Software Design Specification (SDS)       | Describes how the software accomplishes what the SRS requires.  A significant portion is written by the project lead during the architectural design, but many details and specifics are added by software developers during the  unit implementation and testing activity.  It is reviewed for consistency by the project lead during the release activity. |
| Device Classification                     |                                                                                                                                                                                                                                                                                                                                                              |
| Test Record                               | Describes a set of tests which were run, when, and by who.  It also must include enough details to reproduce the testing setup.                                                                                                                                                                                                                              |
| Release Record                            | Describes the verifications steps performed by the project lead during the release.                                                                                                                                                                                                                                                                          |


### 1. Verification plan

#### 1.1 Unit tests overview 

The main method of verifying the code has been written correctly is to generate and perform unit tests. 

- Unit tests will be created to cover >90% of the code. 
- Tests will be written for each item in the design specification. 
- Unit tests will be reviewed at key stages of the development process.
- Each Pull Request should only be merged where all unit tests pass. 


#### 1.2 Unit tests

| Design spec item | Unit Tests                                                                                                                                            | Expected Results | Results |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|---------|


### 2. Validation plan

#### 2.1 Validation test Overview

Validation tests are manual checks and tasks performed to ensure the correct application has been built.
- Validation tests are performed by the software developer to check that the software meets all items identified in the 
System requirements specification. 
- In addition, validation tests can also capture design specification items that cannot be verified with automated unit
tests. 

#### 2.2 Validation tests

| Design Spec Item | Test(s)                                                                                                                                      | Performed by     | Expected result     | Result |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------|------------------|---------------------|--------|



| Requirements spec item | Test(s)                                                                     | Performed by     | Expected result | Result |
|------------------------|-----------------------------------------------------------------------------|------------------|-----------------|--------|
