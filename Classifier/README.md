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
|   |   |-- MobileNetV2.py
|   |   |-- Resnet.py
|   |   |-- VGG.py
|   |   |-- ViT.py (Vision Transformer)
|   |   |-- etc..
```


### Directory structure

```
${CLASSIFIER_ROOT}
|-- common
|   |-- networks
|   |   |-- CNN.py
|   |   |-- MobileNetV2.py
|   |   |-- Resnet.py
|   |   |-- VGG.py
|   |   |-- ViT.py (Vision Transformer)
|   |   |-- etc..
|   |-- networks
|   |   |-- dir_utils.py
|   |   |-- torch_utils.py
|   |-- base.py
|-- main
|   |-- configs
|   |   |-- config of networks..
|   |-- config.py
|   |-- model.py
|   |-- onnx_convertor.py
|   |-- test.py
|   |-- test_itr.py
|   |-- train.py
|   |-- vis.py
|-- vis
|   |-- attention_score.py
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