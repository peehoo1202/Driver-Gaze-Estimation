# Driver-Gaze-Estimation
Driver Gaze estimation using Convolutional Neural Networks.
# Steps
1. Create a google-colab notebook.
2. Clone this repository
3. Install all packages specified in requirements.txt
4. Upload dataset on the notebook
5. Train and Test the model

# Datasets
## LISA Gaze Dataset v0
Download the complete RGB dataset for driver gaze classification using this [link](https://drive.google.com/file/d/1Ez-pHW0v-5bRdz8NjTLlzWZPT0GS2rYT/view).
### Training (v0 RGB data)
The SqueezeNet gaze classifier can be trained using the following command:
```
python gazenet.py --dataset-root-path=/path/to/lisat_gaze_data_v0/ --version=1_1 --snapshot=./weights/squeezenet1_1_imagenet.pth --random-transforms
```

### Inference (v0 RGB data)
Inference can be carried out using the command as follows:
```
python infer.py --dataset-root-path=/path/to/lisat_gaze_data_v0/ --split=val --version=1_1 --snapshot-dir=/path/to/trained/rgb-model/directory/ --save-viz
```

## LISA Gaze Dataset v1
Download the complete RGB dataset for driver gaze classification using this [link](https://drive.google.com/file/d/1YvFzqfDkC2NLX8s0YX0XiMi8SOp_eINx/view).
### Training (v1 RGB data)
The SqueezeNet gaze classifier can be trained using the following command:
```
python gazenet.py --dataset-root-path=/path/to/lisat_gaze_data_v1/ --version=1_1 --snapshot=./weights/squeezenet1_1_imagenet.pth --random-transforms
```
### Inference (v1 RGB data)
Inference can be carried out using the command as follows:
```
python infer.py --dataset-root-path=/path/to/lisat_gaze_data_v1/ --split=val --version=1_1 --snapshot-dir=/path/to/trained/rgb-model/directory/ --save-viz
```

#### Config files, logs, results, snapshots, and visualizations from running the above scripts will be stored in the "Driver-Gaze-Estimation/experiments" folder by default.

# References

[On Generalizing Driver Gaze Zone Estimation using Convolutional Neural Networks," IEEE Intelligent Vehicles Symposium, 2017](http://cvrr.ucsd.edu/publications/2017/IV2017-VoraTrivedi-OnGeneralizingGazeZone.pdf)
[Driver Gaze Zone Estimation using Convolutional Neural Networks: A General Framework and Ablative Analysis," IEEE Transactions on Intelligent Vehicles, 2018] (http://cvrr.ucsd.edu/publications/2018/sourabh_gaze_zone.pdf)
