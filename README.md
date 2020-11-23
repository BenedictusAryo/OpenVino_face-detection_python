# OpenVino_face-detection_python
Tutorial on how to aply Face Detection in webcam using OpenVino with python. This code tested on Windows 10 and Ubuntu.

## What is OpenVino
OpenVino (OpenVisual Inferencing and Neural Network Optimization) is toolkit to develop Deep Learning Application especially for Computer Vision by Intel. OpenVino Enables deep learning inference at the edge and supports heterogeneous execution across computer vision accelerators—CPU, GPU, Intel® Movidius™ Neural Compute Stick, and FPGA—using a common API. [read more](https://docs.openvinotoolkit.org/)

## Download OpenVino
https://software.intel.com/en-us/openvino-toolkit/choose-download

## Hardware Requirement
*CPU:*
* Minimum Intel gen6 processors
* Some Ryzen series support but not all

*GPU:*
* Intel HD Graphics

## OpenVINO Installation Guide
* Windows 10: https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html
* Linux (Ubuntu, Centos, Yocto): https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html
* MacOS: https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_macos.html

## How to use:
* Ubuntu: add `source /opt/intel/openvino/bin/setupvars.sh` to the `.bashrc` <br>
then run `python3 face_detection.py`

* Windows: run OpenVINO `setupvars.bat` first before run the script. _Example_: <br>
`"C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"` <br>
(_with " because it has space on the folder path_) then run `python face_detection.py`

## Latest version tested:
* openvino_2021.1.110