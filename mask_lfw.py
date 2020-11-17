import os
import subprocess

import shutil
from pathlib import Path
from PIL import Image

base_path = "C:\\Users\\david\\Documents\\masked-face-recognition\\lfw_complete"
lfw_orig_path = os.path.join(base_path, "lfw")
# mask_the_face_path = "C:\\Users\\david\\Documents\\MaskTheFace"
# entries = os.listdir(lfw_orig_path)
# for entry in entries:
#     folder_path = os.path.join(lfw_orig_path, entry)    
#     command = 'cd C:\\Users\\david\\Documents\\MaskTheFace && python mask_the_face.py --path {} --mask_type random --verbose --write_original_image'.format(folder_path)
#     print(command)
#     subprocess.call(command, shell=True)

lfw_cropped = os.path.join(base_path, "lfw_cropped")
entries = os.listdir(lfw_cropped)
for entry in entries:
    if "_masked" not in entry:
        masked = entry + "_masked"
        if masked in entries:
            masked_path = os.path.join(lfw_cropped, masked)
            orig_path = os.path.join(lfw_cropped, entry)
            shutil.copytree(masked_path, orig_path, dirs_exist_ok = True)
            shutil.rmtree(masked_path)
        