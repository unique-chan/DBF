import os
from shutil import copy2
from time import sleep

root = '../../' + 'mmdetection/mmdet/models/detectors'
candidates = [f'single_stage.py', f'two_stage.py']
for candidate in candidates:
    if not os.path.exists(f'{root}/{candidate[:-3]} (original).py'):
        os.rename(f'{root}/{candidate}', f'{root}/{candidate[:-3]} (original).py')
        print(f'* Renamed: from "{root}/{candidate}" -> to "{root}/{candidate[:-3]} (original).py"')
        sleep(0.1)
        copy2(f'stages/{candidate}', f'{root}/{candidate}')
        print(f'* Copied: from "stages/{candidate}" -> to "{root}/{candidate}"')
    else:
        print(f'* Already overwritten: "{candidate}" at "{root}"')
