# License-Plate-Recognition
1. License plate detection using **YOLOv4** trained on custom data. </br>
  mAP :- 88.25%(IoU threshold = 50%) </br>
  avg IoU :- 62.87%(conf_threshold = 0.25) </br>
  avg fps :- 16 </br>
2. License plate text detection and recognition using keras-ocr. </br>
3. Rejecting false positives by matching pattern with Indian license plates. </br>

### Demo:
&nbsp; Download pretrained model from [here](https://drive.google.com/drive/folders/1MEyGB-ogHhqnyT5cW5-Vk3wwHowghOCK?usp=sharing) and copy it inside "data" folder. </br>

#Note - 
1. Command line argument 'size' must be a multiple of 32. Increasing 'size' increases the accuracy of license plate detection but requires more memory. Reduce the size if your gpu runs out of memory.
2. If the gpu-ram is 4 GB or less, Reduce memory-limit in this [line](https://github.com/keshavoct98/License-Plate-Recognition/blob/be64d2e01e2d1d9ef54fc36dd931a3f9b74a6c82/demo_video.py#L17) to a value less than your gpu-ram.
3. If the gpu-ram is 2 GB or less, Reduce memory-limit [here](https://github.com/keshavoct98/License-Plate-Recognition/blob/be64d2e01e2d1d9ef54fc36dd931a3f9b74a6c82/demo.py#L17) and [here](https://github.com/keshavoct98/License-Plate-Recognition/blob/be64d2e01e2d1d9ef54fc36dd931a3f9b74a6c82/demo_video.py#L17) to a value less than your gpu-ram.
4. Since there are 3 models running in a sequence(yolov4 for license plate detection, keras-ocr CRAFT text detection and keras-ocr CRNN text recognition), memory usage is high and fps is low. This solution gives an average fps of 2.5 on gtx 1660 gpu.
Fps and memory usage can be improved by training a single YOLOv4 model for both license plate detection and text recognition.

### Results:
![results/output1.jpg](https://github.com/keshavoct98/License-Plate-Recognition/blob/master/results/output1.jpg)
![results/output2.jpg](https://github.com/keshavoct98/License-Plate-Recognition/blob/master/results/output2.jpg)
![results/output3.jpg](https://github.com/keshavoct98/License-Plate-Recognition/blob/master/results/output3.jpg)
![results/output4.jpg](https://github.com/keshavoct98/License-Plate-Recognition/blob/master/results/output4.jpg)
![results/output1.gif](https://github.com/keshavoct98/License-Plate-Recognition/blob/master/results/output1.gif)
![results/output2.gif](https://github.com/keshavoct98/License-Plate-Recognition/blob/master/results/output2.gif)

### How to train:
1. Download dataset zip from this link - https://www.kaggle.com/tustunkok/license-plate-detection/data
2. Create a folder named 'dataset' inside 'data' and extract the content of zip into 'dataset' folder.
3. Go to 'data' folder and run this command on cmd - "python augmentation.py". After augmentation total number of images will be 2154.
4. Now to train darknet YOLOv4 on the dataset, follow the steps given here - https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects

### References
1. https://www.kaggle.com/tustunkok/license-plate-detection/data </br>
2. https://github.com/AlexeyAB/darknet </br>
3. https://github.com/hunglc007/tensorflow-yolov4-tflite </br>
4. http://www.rto.org.in/vehicle-registration-plates-of-india/format-of-number-plates.htm

THANK YOU :)
