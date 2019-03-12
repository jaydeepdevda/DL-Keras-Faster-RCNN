# Keras-FasterRCNN
This implementations is detect car and swimming pool from the satellite images of 224x224 dimensions.<br/>
Dataset is hidden due to privacy issue.<br/>
But you can refer similar Problem [https://www.kaggle.com/c/airbus-ship-detection](https://www.kaggle.com/c/airbus-ship-detection)<br/>

Keras implementation of Faster R-CNN<br/>
cloned from [https://github.com/rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)<br/>

## USAGE:
- Both theano and tensorflow backends are supported. However compile times are very high in theano, and tensorflow is highly recommended.
- `train_frcnn.py` can be used to train a model. To train on Pascal VOC data, simply do:
`python train_frcnn.py -p /path/to/pascalvoc/`. 
- the Pascal VOC data set (images and annotations for bounding boxes around the classified objects) can be obtained from: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
- simple_parser.py provides an alternative way to input data, using a text file. Simply provide a text file, with each
line containing:

    `filepath,x1,y1,x2,y2,class_name`

    For example:

    `dataset/training_data/images/000000000.jpg,58.47,152.31,69.58,163.43,Car`</br>
    `dataset/training_data/images/000000103.jpg,221.63,162.44,224.00,171.55,Pool`

    The classes will be inferred from the file. To use the simple parser instead of the default pascal voc style parser,
    use the command line option `-o simple`. For example `python train_frcnn.py -o simple -p input.txt`.

- Running `train_frcnn.py` will write weights to disk to an hdf5 file, as well as all the setting of the training run to a `pickle` file. These
settings can then be loaded by `test_frcnn.py` for any testing.

- test_frcnn.py can be used to perform inference, given pretrained weights and a config file. Specify a path to the folder containing
images:
    `python test_frcnn.py -p /path/to/test_data/`
- Data augmentation can be applied by specifying `--hf` for horizontal flips, `--vf` for vertical flips and `--rot` for 90 degree rotations



## NOTES:
- config.py contains all settings for the train or test run. The default settings match those in the original Faster-RCNN
paper. The anchor box sizes are [16, 32, 64] and the ratios are [1:1, 1:2, 2:1].
- The theano backend by default uses a 7x7 pooling region, instead of 14x14 as in the frcnn paper. This cuts down compiling time slightly.
- The tensorflow backend performs a resize on the pooling region, instead of max pooling. This is much more efficient and has little impact on results.
- Keras 2.0.3 Version is Supported
- upload code with data and run command `!python train_frcnn.py` in google colab

## Example output:

Input             |  Output
:-------------------------:|:-------------------------:
![](images/1.jpg?raw=true "Input")  |  ![](results_imgs/1.png?raw=true "Output")
![](images/2.jpg?raw=true "Input")  |  ![](results_imgs/2.png?raw=true "Output")


## ISSUES:

- If you run out of memory, try reducing the number of ROIs that are processed simultaneously. Try passing a lower `-n` to `train_frcnn.py`. Alternatively, try reducing the image size from the default value of 224 (as image dimension is 224 x 224) (this setting is found in `config.py`).

## Reference
[1] [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, 2015](https://arxiv.org/pdf/1506.01497.pdf) <br/>
[2] [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning, 2016](https://arxiv.org/pdf/1602.07261.pdf) <br/>
[3] [https://github.com/yhenon/keras-frcnn/](https://github.com/yhenon/keras-frcnn/)<br/>
[4] [https://github.com/you359/Keras-FasterRCNN](https://github.com/you359/Keras-FasterRCNN)<br/>
[5] [https://github.com/jinfagang/keras_frcnn](https://github.com/jinfagang/keras_frcnn)
