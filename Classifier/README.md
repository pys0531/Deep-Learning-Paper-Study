### Directory structure of the data

```
${CLASSIFIER_ROOT}
|-- data
|   |-- cifar10
|   |-- stl10
|   |-- etc..
```


### Networks structure
```
${CLASSIFIER_ROOT}
|-- common
|   |-- networks
|   |   |-- CNN.py
|   |   |-- Resnet.py
|   |   |-- etc..
.
.
.
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
