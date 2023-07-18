### Directory structure of the data

```
${CLASSIFIER_ROOT}
|-- data
|   |-- cifar10
|   |-- stl10
|   |-- etc..
```


### Directory structure

```
${CLASSIFIER_ROOT}
|-- common
|   |-- networks
|   |   |-- CNN.py
|   |   |-- Resnet.py
|   |   |-- etc..
|   |-- base.py
|-- main
|   |-- config.py
|   |-- model.py
|   |-- onnx_convertor.py
|   |-- test.py
|   |-- test_itr.py
|   |-- train.py
|-- model_dump
|   |-- snapshot_0.pth.tar
|   |-- snapshot_1.pth.tar
.
.
.
```


### Run
