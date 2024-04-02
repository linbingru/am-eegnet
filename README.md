# AM-EEGNet: An advanced multi-input deep learning framework for classifying stroke patient EEG task states
## Introduction
This EEG classification deep learning model was from the Lab of Intelligent and Bio-mimetic Machinery, Department of Mechanical Engineering, Tsinghua University, Beijing, China.
## Requirements
For the deep learning model
- Python == 3.7 or 3.8
- Keras == 2.6.0
- Tensorflow == 2.3 (Both for CPU and GPU)

For model explanation
- Shap == 0.39.0

For EEG signal preprocessing
- Matlab == R2021b
- EEGLAB == v2021.1

## Usage
To use this package, place the AM_EEGModels.ipynb file in your project folder. Then, one can simply use this model as
```
from AM_EEGModels import AM_EEGNet

model = AM_EEGNet(classes = ..., psd_frq = ..., psd_chans = ..., fc_conn = ..., fc_band = ...)
```
Compile the model with the associated loss function and optimizer (in our case, the categorical cross-entropy and Adam optimizer, respectively). Then fit the model and predict on new test data.
```
# Classes determination
if classes == 2:
    model.compile(loss = 'BinaryCrossentropy',optimizer = 'adam')
else:
    model.compile(loss = 'categorical_crossentropy',optimizer = 'adam')

# Model fit and predict   
model.fit([your_psd_train_data, your_fc_train_data], your_target_label, epochs = ..., batch_size = ..., verbose = 1)
model.predict([your_psd_test_data, your_fc_test_data])
```

## Paper Citation
if you use this model in your research and found it helpful, please cite the following paper:
```
@article{Lin2024,
  author={Ping-Ju Lin and Wei Li and Xiaoxue Zhai and Jingyao Sun and Yu Pan and Linhong Ji and Chong Li},
  title={AM-EEGNet: An advanced multi-input deep learning framework for classifying stroke patient EEG task states},
  journal={Neurocomputing},
  volume={585},
  number={127622},
  url={https://doi.org/10.1016/j.neucom.2024.127622},
  year={2024}
}
```
