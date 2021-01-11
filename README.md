# TFX implementation of Centernet.

### Summary
This repository contains a TFX pipeline for training a Centernet model on
the CCPD2019 license plate dataset.

### Features
- AI platform training with GPU
- Dataflow processing
- Training with custom containers
- Object Detection

### Code Structure
The repo is structured as follows:

Core:
- Contains the code related to centernet implementation.

Data:
- Contains the code to generate TFRecords

Pipeline:
- Create the TFX pipeline + utility function for schema generation.

Trainer:
- Code for tensorflow transform and trainer components.