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
OPENVINO_DIR = 'C:\\Program Files (x86)\\IntelSWTools\\openvino'
# if platform.os.environ.get('PATH').find('openvino') != -1:
#     platform.subprocess.run('setupvars.bat')
platform.sys.path.append(
    OPENVINO_DIR+'\\python\\python3.7')
platform.sys.path.append(
    OPENVINO_DIR+'\\python\\python3.7\\openvino\\inference_engine')
platform.sys.path.append(
    OPENVINO_DIR+'\\python\\python3')
platform.sys.path.append(
    OPENVINO_DIR+'\\opencv\\cmake')
platform.sys.path.append(
    OPENVINO_DIR+'\\openvx')
platform.sys.path.append(
    OPENVINO_DIR+'\\deployment_tools\\inference_engine\\share')
platform.sys.path.append(
    OPENVINO_DIR+'\\deployment_tools\\inference_engine\\external\\hddl')
platform.sys.path.append(
    OPENVINO_DIR+'\\deployment_tools\\inference_engine\\external\\hddl\\bin')
platform.sys.path.append(
    OPENVINO_DIR+'\\deployment_tools\\inference_engine\\bin\\intel64\\Release')
platform.sys.path.append(
    OPENVINO_DIR+'\\deployment_tools\\inference_engine\\bin\\intel64\\Debug')

# platform.sys.path.append('C:\\Program Files (x86)\\IntelSWTools\\openvino\\deployment_tools\\open_model_zoo\\tools\\accuracy_checker')
if platform.os.environ.get('PATH').find('openvino') != -1:
    import cv2 as cv
    # import openvino
    from openvino.inference_engine import IENetwork, IEPlugin
else:
    print('OpenVINO Setupvars is not in your path')

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
