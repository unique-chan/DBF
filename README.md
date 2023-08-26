<h1 align="center" style="font-weight: 500; line-height: 1.4;">
  Dynamic Backbone Freezing 
</h1>

<p align="center">
  <a href="#"><img alt="Python3.7+" src="https://img.shields.io/badge/Python-3.7+-blue?logo=python&logoColor=white"></a>
  <a href="#"><img alt="PyTorch1.5+" src="https://img.shields.io/badge/PyTorch-1.5+-orange?logo=pytorch&logoColor=white"></a>
  <a href="#"><img alt="MMDetection2.28.2" src="https://img.shields.io/badge/MMDetection-2.28.2-red?logo=mmdetection&logoColor=white"></a>
  <a href="#"><img alt="MIT" src="https://img.shields.io/badge/License-MIT-green?logo=MIT"></a>
</p>

<p align="center">
  <b><a href="https://github.com/unique-chan">Yechan Kim</a></b>
</p>


### This repo includes:
- Official implementation of our proposed approach

### Overview:
- This work presents a training strategy coined `Dynamic Backbone Freezing` that aims to achieve two distinct goals in remote-sensing object detection: **resource-saving** and **robust prediction**.
![Overview_Figure](./my_src/Overview_DBF.png)
  - Specifically, this work implements and utilizes **Freezing Scheduler** to dynamically control the update of backbone features during training.

### Preliminaries
- Install all necessary packages listed in the `requirements.txt`. 
- Modify your base detector code as follows:
  - Declare an attribute named `bool_freeze_backbone` (boolean variable)
  - Modify `extract_feat()` to be dynamically locked / unlocked by `bool_freeze_backbone` 
~~~
# example: mmdetection 2.x ➡️ mmdet/models/detectors/single_stage.py

class SingleStageDetector(...):

  def __init__(...):
    self.bool_freeze_backbone = False
    ...
  
  def extract_feat(...):
    x = self.backbone(img)
    if self.with_neck:
      if self.bool_freeze_backbone:  x = self.neck(tuple([_.detach() for _ in x]))
      else:                          x = self.neck(x)
    else:
      if self.bool_freeze_backbone:  x = x.detach()
    return x
  
  ...
~~~
- For experiments on your own dataset and detection model, prepare your own configuration file in `my_src/my_cfg`. (See `README.md` in `my_src/my_cfg` for details.)

### Announcement:
- Code under construction
