# -*- coding: utf-8 -*-
"""
Created on Wed Oct  16 10:14:14 2019

@author: benedict.aryo
"""
#######################################################################
######################  Library Initialization  #########################
#  Import Library being used in program
from openvino.inference_engine import IENetwork, IEPlugin
import cv2 as cv
import platform
import time
import argparse
