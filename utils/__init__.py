import os
import sys
import inspect

current_path = !pwd
parent_path = os.path.dirname(current_path[0])
if parent_path not in sys.path: sys.path.append(parent_path)
santi_path = "/".join(parent_path.split('/')[:-1])+"/santi-iso-response"
if santi_path not in sys.path: sys.path.append(santi_path)
santi_etc_path = os.path.join(santi_path, 'etc')
sys.path.append(santi_etc_path)

from models.cnn_sys_ident.data import Dataset, MonkeyDataset
from models.cnn_sys_ident.cnn import ConvNet