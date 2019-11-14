# -*- coding: utf-8 -*-
"""
Created on Wed Oct  16 10:14:14 2019

@author: benedict.aryo

"""
#######################################################################
######################  Library Initialization  #########################
#  Import Library being used in program
import platform
import argparse
import time

try:
    from openvino.inference_engine import IENetwork, IEPlugin
    import cv2 as cv
except:
    raise Exception("""
OpenVINO not found in your environment.

in Windows: run OPENVINO_DIR/bin/setupvars.bat before run this script.
in Ubuntu: put this on your .bashrc files: 
    source /opt/intel/openvino/bin/setupvars.sh
           then run the script again.""")


# OPENVINO_DIR = 'C:\\Program Files (x86)\\IntelSWTools\\openvino'
# if platform.os.environ.get('PATH').find('openvino') != -1:
#     platform.subprocess.run('setupvars.bat')
# os.environ['PYTHONPATH'] = 'C:\\Program Files (x86)\\IntelSWTools\\openvino\\deployment_tools\\open_model_zoo\\tools\\accuracy_checker;C:\\Program Files (x86)\\IntelSWTools\\openvino\\python\\python3.7;C:\\Program Files (x86)\\IntelSWTools\\openvino\\python\\python3;C:\\Program Files (x86)\\IntelSWTools\\openvino\\deployment_tools\\open_model_zoo\\tools\\accuracy_checker;C:\\Program Files (x86)\\IntelSWTools\\openvino\\python\\python3.7;C:\\Program Files (x86)\\IntelSWTools\\openvino\\python\\python3;'
# os.environ['INTEL_OPENVINO_DIR'] = 'C:\\Program Files (x86)\\IntelSWTools\\openvino'
# os.environ['INTEL_CVSDK_DIR'] = 'C:\\Program Files (x86)\\IntelSWTools\\openvino'
# os.environ['OpenCV_DIR'] = 'C:\\Program Files (x86)\\IntelSWTools\\openvino\\opencv\\cmake'


# # platform.sys.path.append('C:\\Program Files (x86)\\IntelSWTools\\openvino\\deployment_tools\\open_model_zoo\\tools\\accuracy_checker')
# if platform.os.environ.get('PATH').find('openvino') != -1:
#     import cv2 as cv
#     # import openvino
#     from openvino.inference_engine import IENetwork, IEPlugin
# else:
#     print('OpenVINO Setupvars is not in your path')

#####################  Argument Parser  ################################
parser = argparse.ArgumentParser(description="OpenVINO Face Detection")
parser.add_argument("-d", "--device", metavar='', default='CPU',
                    help="Device to run inference: GPU, CPU or MYRIAD", type=str)
parser.add_argument("-c", "--camera", metavar='', default=0,
                    help="Camera Device, default 0 for Webcam", type=int)
parser.add_argument("-s", "--sample", default=False,
                    action='store_true', help="Inference using sample video")

args = parser.parse_args()

#######################  Device  Initialization  ########################
#  Plugin initialization for specified device and load extensions library if specified

device = args.device.upper()

# Device Options = "CPU", "GPU", "MYRIAD"
plugin = IEPlugin(device=device)

# DETECT OS WINDOWS / UBUNTU  TO USE EXTENSION LIBRARY
# Plugin UBUNTU :
LINUX_CPU_PLUGIN = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so"
# Plugin Windows
WINDOWS_CPU_PLUGIN = r"C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll"

if platform.system() == 'Windows':
    cpu_plugin = WINDOWS_CPU_PLUGIN
else:
    cpu_plugin = LINUX_CPU_PLUGIN

# Add Extension to Device Plugin
if device == "CPU":
    plugin.add_cpu_extension(cpu_plugin)

#################### no need for GPU or MYRIAD ########################
#######################################################################

#######################  MODEL INITIALIZATION  ########################
#  Prepare and load the models

# Model 1: Face Detection
# FACEDETECT_XML = "models/face-detection-retail-0004.xml"
# FACEDETECT_BIN = "models/face-detection-retail-0004.bin"
FACEDETECT_XML = "models/face-detection-adas-0001.xml"
FACEDETECT_BIN = "models/face-detection-adas-0001.bin"
# Model 2: Age Gender Recognition
AGEGENDER_XML = "models/age-gender-recognition-retail-0013_FP32.xml"
AGEGENDER_BIN = "models/age-gender-recognition-retail-0013_FP32.bin"

################  Create PostProcessing Inferece Function  ################


def gender_class(gender):
    """
    PostProcessing & Classify Output Gender into Male & Female
    """
    GENDER_LIST = ['Female', 'Male']
    if gender[0, 1, 0, 0] >= 0.60:
        return GENDER_LIST[1]
    else:
        return GENDER_LIST[0]


def age_class(age):
    """
    Classify Age whether it's below or above 30
    """
    return 'Below 30' if age <= 30 else 'Above 30'
