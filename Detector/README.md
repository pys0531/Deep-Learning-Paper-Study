### Directory structure of the data
You need to follow direction structure of the data as below </br>
Download PASCALVOC data **[VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html) [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)**
```
${CLASSIFIER_ROOT}
|-- data
|   |-- PASCALVOC
|   |   |-- VOC2007
|   |   |   |-- Annotations
|   |   |   |-- ImageSets
|   |   |   |-- JPEGImages
|   |   |   |-- SegmentationClass
|   |   |   |-- SegmentationObject
|   |   |-- VOC2012
|   |   |   |-- Annotations
|   |   |   |-- ImageSets
|   |   |   |-- JPEGImages
|   |   |   |-- SegmentationClass
|   |   |   |-- SegmentationObject
|   |   |-- PASCALVOC.py
```


### Networks structure
```
${CLASSIFIER_ROOT}
|-- common
|   |-- networks
|   |   |-- MobileNetV2.py
|   |   |-- VGG.py
```


### Directory structure

```
${CLASSIFIER_ROOT}
|-- common
|   |-- networks
|   |   |-- MobileNetV2.py
|   |   |-- VGG.py
|   |-- networks
|   |   |-- networks
|   |   |   |-- MobileNetV2.py
|   |   |   |-- VGG.py
|   |   |-- SSD.py
|   |   |-- modules.py
|   |-- utils
|   |   |-- dir_utils.py
|   |   |-- preprocessing.py
|   |   |-- torch.utils.py
|   |-- base.py
|-- main
|   |-- configs
|   |   |-- config of networks..
|   |-- config.py
|   |-- detect.py
|   |-- function.py
|   |-- losses.py
|   |-- model.py
|   |-- onnx_convertor.py
|   |-- test.py
|   |-- train.py
|-- model_dump
|   |-- snapshot_0.pth.tar
|   |-- snapshot_1.pth.tar
.
.
.
```


### Train
change the config file to the model befor training
```
cd main
python train.py
```

### Test
```
cd main
python test.py --test_epoch {epoch_num}
```

### Onnx Convert
```
cd main
python onnx_convertor.py
```