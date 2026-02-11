
import pandas as pd
import os
import glob
import numpy as np
import seaborn as sns
import json
import matplotlib.pyplot as plt
import re
import sys
from tqdm import tqdm

sys.path.append('../')
from utils import *
from gatherData import _parse_ncu_report, roofline_results_to_df, calc_roofline_data

from tqdm.contrib.concurrent import process_map
from functools import partial
from os import path
import csv
from io import StringIO

sm_80_sass_files = glob.glob('../collected-data/scraped-sass/*sm_80*.sass')
sm_86_sass_files = glob.glob('../collected-data/scraped-sass/*sm_86*.sass')
sm_90_sass_files = glob.glob('../collected-data/scraped-sass/*sm_90*.sass')


def extract_text_SASS_sections_from_sass_file(sass_file):
    # TODO: Maybe move this into a class as a static function?
    with open(sass_file, 'r') as f:
        content = f.read()

    # Each .text CUDA kernel section looks something like this:
    # //--------------------- .text._ZN3cub17CUB_300001_SM_8006detail11EmptyKernelIvEEvv --------------------------
    # //--------------------- .text.__cuda_reduxsync_s32_add --------------------------
    # //--------------------- .text._Z9compute_iiPKfS0_Pi --------------------------

    return sass_sections


def sass_results_to_df():
    # TODO
    return None


    


