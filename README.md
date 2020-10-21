# Text classification

## Model description

### Model
This model is mainly composed of:
- Embeddings layer (possibility to use pre-trained GloVe embeddings)
- Convolution layer
- LSTM layer
- Attention mechanism
- Dense layers 
- Softmax as the last layer

Its complexity varies depending on the embeddings dimension, and is roughly
around 430,000 parameters.

Loss is categorical cross-entropy, minimised with the Adam algorithm.

Example with embeddings of dimension 10:
```text
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            [(None, 10)]         0                                            
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 10, 10)       180         input_2[0][0]                    
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 10, 32)       992         embedding_1[0][0]                
__________________________________________________________________________________________________
max_pooling1d_1 (MaxPooling1D)  (None, 5, 32)        0           conv1d_1[0][0]                   
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 5, 32)        128         max_pooling1d_1[0][0]            
__________________________________________________________________________________________________
lstm_1 (LSTM)                   (None, 5, 100)       53200       batch_normalization_3[0][0]      
__________________________________________________________________________________________________
dense_6 (Dense)                 (None, 5, 1)         101         lstm_1[0][0]                     
__________________________________________________________________________________________________
flatten_2 (Flatten)             (None, 5)            0           dense_6[0][0]                    
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 5)            0           flatten_2[0][0]                  
__________________________________________________________________________________________________
repeat_vector_1 (RepeatVector)  (None, 100, 5)       0           activation_1[0][0]               
__________________________________________________________________________________________________
permute_1 (Permute)             (None, 5, 100)       0           repeat_vector_1[0][0]            
__________________________________________________________________________________________________
multiply_1 (Multiply)           (None, 5, 100)       0           lstm_1[0][0]                     
                                                                 permute_1[0][0]                  
__________________________________________________________________________________________________
flatten_3 (Flatten)             (None, 500)          0           multiply_1[0][0]                 
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 500)          250500      flatten_3[0][0]                  
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 500)          0           dense_7[0][0]                    
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 500)          2000        dropout_3[0][0]                  
__________________________________________________________________________________________________
dense_8 (Dense)                 (None, 200)          100200      batch_normalization_4[0][0]      
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 200)          0           dense_8[0][0]                    
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 200)          800         dropout_4[0][0]                  
__________________________________________________________________________________________________
dense_9 (Dense)                 (None, 100)          20100       batch_normalization_5[0][0]      
__________________________________________________________________________________________________
dropout_5 (Dropout)             (None, 100)          0           dense_9[0][0]                    
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 10)           1010        dropout_5[0][0]                  
__________________________________________________________________________________________________
dense_11 (Dense)                (None, 1)            11          dense_10[0][0]                   
==================================================================================================
Total params: 429,222
Trainable params: 427,758
Non-trainable params: 1,464
__________________________________________________________________________________________________
```
Training was using 400% CPU and 13G RAM on my laptop.

#### Results
These results were computed using cross-validation (4 stratified splits).
- Accuracy: 12.15% 
- Loss (categorical cross-entropy): 4.96
The model was still improving steadily after 20 epochs, but it was taking quite a 
while to compute (over 3 hours of training) so I limited the test.

### Preprocessing
Words are tokenized using Keras' tokenizer. The sequence length is fixed and defined
in the model parameters. Any tokenized sequence that has more words that the 
fixed length will be truncated, and the shorter ones will be zero-padded.

Labels are one-hot encoded with scikit-learn's LabelBinarizer.

### Evaluation
Evaluation is performed with Keras' function "evaluate" and reports loss and accuracy 
across all classes. The model is evaluated on the training set with cross validation.

## Architecture
##### classifier package
This is the main python package. It contains:
- model.py: wrapper that builds the keras model
- preprocessing.py: contains a class for text preprocessing (tokenizing and zero-padding)
- pipeline.py: chains the preprocessing and the model, and formats the labels
- utils: functions for data loading and reading pre-trained GloVe embeddings.

##### Tests
Contains one test file per module in the classifier package. These
tests are designed to run using pytest.

##### data
Folder I use to store glove embeddings and training/test data. 

##### models
Where I initially wanted to store the trained models. This functionality is
not yet supported.

## Usage
This code can be used in a python script or from the command line. The script main.py shows 
an example of how to create a text classification pipeline, cross-validate the model
and generate predictions.

### Command line
the repository can be run from outside this folder. All arguments start with "--"
```shell script
python adthena_test --training_data adthena_test/data/trainSet.csv
```
Arguments:
- training_data: path to training data
- prediction_data: path to prediction data
- output_predictions: path where to output the predictions
- sequence_length: maximum number of words taken into account by the model
- embeddings_path: path to pretrained glove embeddings.
- embeddings_dim: dimension of the embeddings. If the embeddings path is given, 
this number must match the dimension of the embeddings.
- batch_size: batch size when training the model. Defaults to 32.
- epochs: number of epochs to train the model. Defaults to 1.

### Improvements
The model could be improved with the following:
- Increasing the number of epochs (predictions are generated with a model trained on 20)
- Tuning hyperparameters (no tuning was done, I adapted structures that have worked for
me in the past)
- Using higher-dimension word embeddings / different word embeddings. I chose the smallest 
GloVe vectors I had, but from experience FastText works slightly better (but they
worked slower when I used them in the past)
- Saving / reloading the model. The label encoder and text formatter can be saved 
with pickle, while the keras model has to be saved with Keras' save model function.
This requires coding a specific function.
- Keras and Tensorflow have changed since I used it last. For example, it contains
a built in attention layer. The model could be better / clearer if it uses it.


### Dependencies
Built on:
- Tensorflow 2.3.1
- pandas 1.1.3
- scikit-learn 0.23.2

### Running tests
Using pytest (my version: 6.6.1), go to the repository's directory and execute:
```shell script
python -m pytest tests
```
