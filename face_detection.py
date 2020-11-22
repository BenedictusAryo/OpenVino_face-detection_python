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
    from openvino.inference_engine import IENetwork, IECore
    import cv2 as cv
except:
    raise Exception("""
OpenVINO not found in your environment.

After install OpenVINO:
in Windows: run OPENVINO_DIR/bin/setupvars.bat before run this script.
    eg: "C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"
in Ubuntu: put this on your .bashrc files: 
    source /opt/intel/openvino/bin/setupvars.sh
           then run the script again.""")

#####################  Argument Parser  ################################
parser = argparse.ArgumentParser(description="OpenVINO Face Detection")
parser.add_argument("-d", "--device", metavar='', default='CPU',
                    help="Device to run inference: GPU, CPU or MYRIAD", type=str)
parser.add_argument("-c", "--camera", metavar='', default=0,
                    help="Camera Device, default 0 for Webcam", type=int)
parser.add_argument("-s", "--sample", default=False,
                    action='store_true', help="Inference using sample video")

args = parser.parse_args()

#######################  DEVICE INITIALIZATION  ########################
#  Plugin initialization for specified device and load extensions library if specified
device = args.device.upper()

# Device Options = "CPU", "GPU", "MYRIAD"
plugin = IECore()

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
# if device == "CPU":
#     plugin.add_extension(cpu_plugin, device)

#################### no need for GPU or MYRIAD ########################
#######################################################################

#######################  MODEL INITIALIZATION  ########################
#  Prepare and load the models

# Model : Face Detection
FACEDETECT_XML = "models/face-detection-adas-0001.xml"
FACEDETECT_BIN = "models/face-detection-adas-0001.bin"


#######################  IMAGE PREPROCESSING  ########################
# Input Image Preprocessing
def image_preprocessing(image, n, c, h, w):
    """
    Image Preprocessing steps, to match image 
    with Input Neural nets

    Image,
    N, Channel, Height, Width
    """
    blob = cv.resize(image, (w, h))  # Resize width & height
    blob = blob.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    blob = blob.reshape((n, c, h, w))
    return blob

#########################  LOAD NEURAL NETWORK  ########################


def load_model(plugin, model, weights, device):
    """
    Load OpenVino IR Models

    Input:
    Plugin = Hardware Accelerator
    Model = model_xml file 
    Weights = model_bin file

    Output:
    execution network (exec_net)
    """
    #  Read in Graph file (IR) to create network
    net = plugin.read_network(model, weights)
    # Load the Network using Plugin Device
    exec_net = plugin.load_network(network=net, device_name=device)
    return net, exec_net


####################  CREATE EXECUTION NETWORK  #######################
net_facedetect, exec_facedetect = load_model(
    plugin, FACEDETECT_XML, FACEDETECT_BIN, device)

#################  OBTAIN INPUT & OUTPUT TENSOR  ######################
# Face Detection Model
#  Define Input&Output Network dict keys
FACEDETECT_INPUTKEYS = 'data'
FACEDETECT_OUTPUTKEYS = 'detection_out'
#  Obtain image_count, channels, height and width
n_facedetect, c_facedetect, h_facedetect, w_facedetect = net_facedetect.input_info[FACEDETECT_INPUTKEYS].input_data.shape


#########################  READ VIDEO CAPTURE  ########################
#  Using OpenCV to read Video/Camera
#  Use 0 for Webcam, 1 for External Camera, or string with filepath for video
if args.sample:
    input_stream = 'face-demographics-walking-and-pause.mp4'
else:
    input_stream = args.camera

cap = cv.VideoCapture(input_stream)

#  If Video File, slow down the video playback based on FPS
if type(input_stream) is str:
    time.sleep(1/cap.get(cv.CAP_PROP_FPS))

while cv.waitKey(1) != ord('q'):
    if cap:
        hasFrame, image = cap.read()

    if not hasFrame:
        break

    ###################  Start  Inference Face Detection  ###################
    #  Start asynchronous inference and get inference result
    blob = image_preprocessing(
        image, n_facedetect, c_facedetect, h_facedetect, w_facedetect)
    req_handle = exec_facedetect.start_async(
        request_id=0, inputs={FACEDETECT_INPUTKEYS: blob})

    ######################## Get Inference Result  #########################
    status = req_handle.wait()
    res = req_handle.outputs[FACEDETECT_OUTPUTKEYS]

    # Get Bounding Box Result
    for detection in res[0][0]:
        confidence = float(detection[2])  # Face detection Confidence
        # Obtain Bounding box coordinate, +-10 just for padding
        xmin = int(detection[3] * image.shape[1] - 10)
        ymin = int(detection[4] * image.shape[0] - 10)
        xmax = int(detection[5] * image.shape[1] + 10)
        ymax = int(detection[6] * image.shape[0] + 10)

        # OpenCV Drawing Set Up
        font = cv.FONT_HERSHEY_SIMPLEX
        fontColor = (0, 0, 255)
        bottomLeftCornerOfText = (xmin, ymin-10)
        fontScale = 0.6
        lineType = 1

        cv.putText(image, "press 'q' to exit", (4, 20),
                   font, fontScale, fontColor, lineType)

        # Crop Face which having confidence > 90%
        if confidence > 0.9:
            # Draw Boundingbox
            cv.rectangle(image, (xmin, ymin), (xmax, ymax), fontColor)

    cv.namedWindow('OpenVINO Face Detection', cv.WINDOW_NORMAL)
    cv.moveWindow('OpenVINO Face Detection', 0, 0)
    cv.resizeWindow('OpenVINO Face Detection', 700, 700)
    cv.imshow('OpenVINO Face Detection', image)
