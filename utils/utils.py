import os
import argparse
import json
import random
import re
import logging

import numpy as np
import av

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

from mamba_ssm import Mamba
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import deepspeed

from tqdm import tqdm

def parse_time(time):
    try:
        return float(time)
    except ValueError:
        return None

def interpret_time(time):
   starts_pattern = re.search(r'starts? at (\d+\.?\d*)', time.lower())
   ends_pattern = re.search(r'ends? at (\d+\.?\d*)', time.lower())

   if starts_pattern and ends_pattern:
       try:
           start = float(starts_pattern.group(1))
           end = float(ends_pattern.group(1))
           return start, end
       except ValueError:
           return None, None

   return None, None

def calculate_iou (pred_start, pred_end, gt_start, gt_end):
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return intersection / union if union > 0 else 0