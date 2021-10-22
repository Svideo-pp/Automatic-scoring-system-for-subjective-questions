import matplotlib
import pandas as pd
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaModel, RobertaConfig
from matplotlib import pyplot as plt
import numpy as np


def Sentence_tokens_length_distribution(input_string, tokenizer):
    """""""""
    A function that first tokenize the input sentences 
    and Count the number of tokens in each sentence and plot the result

    Input:
        - tokenizer:   Tokenizer object from the PreTrainedTokenizer Class
        - input_string: The string list you want to calculate and plot
    Output:
        A window illustrates the result
    """""""""

    input_tokenized = tokenizer(input_string)

    row_length = []
    for i in range(len(input_tokenized['input_ids'])):
        row_length.append(len(input_tokenized['input_ids'][i]))

    plt.figure(figsize=(10, 5))
    res_tuple = plt.hist(row_length, bins=50, color='steelblue')
    plt.title('Tokens Length Distribution')
    plt.show()


def Label_analysis(input_labels):
    """""""""
    A function to analysis whether a certain dataset's labels are distributed uniformly.
    (i.e. 50% correct label with 50% incorrect label)

    Input:
        - input_labels: The label list you want to calculate and plot
    Output:
        A window illustrates the result
    """""""""
    plt.figure(figsize=(5, 5))
    res_tuple = plt.hist(input_labels, bins=3, color='steelblue')
    plt.title('Labels Distribution')
    plt.show()


# hyperparameters
model_name = 'C:/Users/svideo/Desktop/HuggingFace Model/roberta-base'
sequence_length = 64
batchsz = 32
ROBERTA_DROPOUT = 0.2
ROBERTA_ATT_DROPOUT = 0.2
LAYER_DROPOUT = 0.2
LEARNING_RATE = 5e-5
RANDOM_STATE = 42
EPOCHS = 5


# read data from csv file
dataset = pd.read_csv('Dataset.csv', usecols=[1, 2, 3, 4, 5])

# print some rows of the whole dataset and the length of the whole dataset
print(dataset)

# tokenize all input strings and sketch the result
tokenizer = RobertaTokenizer.from_pretrained(model_name)
Sentence_tokens_length_distribution(dataset['studentAnswer'].tolist(), tokenizer)

# Split the data set, obtain the train, val and test dataset
"""
train_set 80% data
val_set   10% data
test_set  10% data
"""
dataset = dataset.sample(frac=1.0)
dataset = dataset.reset_index()

train_features = dataset.sample(frac=0.8)
dataset = dataset[~dataset.index.isin(train_features.index)]

val_features = dataset.sample(frac=0.5)
test_features = dataset[~dataset.index.isin(val_features.index)]

# train_features.pop('index')
# val_features.pop('index')
# test_features.pop('index')

# construct label set for each dataset
train_labels = train_features.pop('accuracy')
val_labels = val_features.pop('accuracy')
test_labels = test_features.pop('accuracy')

temp = []
for i in train_labels:
    if(i == 'correct'): temp.append(1)
    else: temp.append(0)
train_labels = pd.DataFrame(temp, columns=['accuracy'])

temp = []
for i in val_labels:
    if(i == 'correct'): temp.append(1)
    else: temp.append(0)
val_labels = pd.DataFrame(temp, columns=['accuracy'])

temp = []
for i in test_labels:
    if(i == 'correct'): temp.append(1)
    else: temp.append(0)
test_labels = pd.DataFrame(temp, columns=['accuracy'])

# output train, val, and test set to check
print(train_features), print(val_features), print(test_features)
print(train_labels), print(val_labels), print(test_labels)

# label distribution analysis
temp = []
temp.extend(train_labels['accuracy'].tolist())
temp.extend(val_labels['accuracy'].tolist())
temp.extend(test_labels['accuracy'].tolist())
Label_analysis(temp)


# Define function to encode text data in batches
def batch_encode(tokenizer, texts, batch_size=1, max_length=sequence_length):
    """""""""
    A function that encodes a batch of texts and returns the texts'
    corresponding encodings and attention masks that are ready to be fed
    into a pre-trained transformer model.

    Input:
        - tokenizer:   Tokenizer object from the PreTrainedTokenizer Class
        - texts:       List of strings where each string represents a text
        - batch_size:  Integer controlling number of texts in a batch
        - max_length:  Integer controlling max number of words to tokenize in a given text
    Output:
        - input_ids:       sequence of texts encoded as a tf.Tensor object
        - attention_mask:  the texts' attention mask encoded as a tf.Tensor object
    """""""""

    input_ids = []
    attention_mask = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer.batch_encode_plus(batch,
                                             max_length=max_length,
                                             padding='max_length',  # implements dynamic padding
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_token_type_ids=False)
        input_ids.extend(inputs['input_ids'])
        attention_mask.extend(inputs['attention_mask'])

    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)
    # return input_ids, attention_mask


# Encode Train_features
train_features_1_ids, train_features_1_attention = batch_encode(tokenizer, train_features['studentAnswer'].tolist())
train_features_ref_ids, train_features_ref_attention = batch_encode(tokenizer, train_features['referenceAnswer'].tolist())
train_features_2_ids, train_features_2_attention = batch_encode(tokenizer, train_features['ref_1'].tolist())
train_features_3_ids, train_features_3_attention = batch_encode(tokenizer, train_features['ref_2'].tolist())

# Encode Valid_features
val_features_1_ids, val_features_1_attention = batch_encode(tokenizer, val_features['studentAnswer'].tolist())
val_features_ref_ids, val_features_ref_attention = batch_encode(tokenizer, val_features['referenceAnswer'].tolist())
val_features_2_ids, val_features_2_attention = batch_encode(tokenizer, val_features['ref_1'].tolist())
val_features_3_ids, val_features_3_attention = batch_encode(tokenizer, val_features['ref_2'].tolist())

# Encode Test_features
test_features_1_ids, test_features_1_attention = batch_encode(tokenizer, test_features['studentAnswer'].tolist())
test_features_ref_ids, test_features_ref_attention = batch_encode(tokenizer, test_features['referenceAnswer'].tolist())
test_features_2_ids, test_features_2_attention = batch_encode(tokenizer, test_features['ref_1'].tolist())
test_features_3_ids, test_features_3_attention = batch_encode(tokenizer, test_features['ref_2'].tolist())

# shape of train_features_ids is (3940, 64)
# shape of train_features_attention (3940, 64)


# Configure Roberta's initialization
# config = RobertaConfig(dropout=ROBERTA_DROPOUT,
#                        attention_dropout=ROBERTA_ATT_DROPOUT,
#                        output_hidden_states=True,
#                        output_attentions=True)

# The bare, pre-trained Roberta transformer model outputting raw hidden-states and without any specific head on top.
Roberta = TFRobertaModel.from_pretrained(model_name)

# Make RobertaModel layers trainable
for layer in Roberta.layers:
    layer.trainable = True


def build_model(transformer, max_length=sequence_length):
    """""""""
    Template for building a model of the BERT or Roberta architecture
    for a binary classification task.

    Input:
      - transformer:  a base Hugging Face transformer model object (BERT or Roberta)
                      with no added classification head attached.
      - max_length:   integer controlling the maximum number of encoded tokens
                      in a given sequence.

    Output:
      - model:        a compiled tf.keras.Model with added classification layers
                      on top of the base pre-trained model architecture.
    """""""""

    # Define weight initializer with a random seed to ensure reproducibility
    weight_initializer = tf.keras.initializers.GlorotNormal(seed=RANDOM_STATE)

    # Define input layers
    input_ids_layer = tf.keras.layers.Input(shape=(max_length,),
                                            name='input_ids',
                                            dtype='int32')
    input_attention_layer = tf.keras.layers.Input(shape=(max_length,),
                                                  name='input_attention',
                                                  dtype='int32')

    ref_ids_layer = tf.keras.layers.Input(shape=(max_length,),
                                          name='ref_ids',
                                          dtype='int32')
    ref_attention_layer = tf.keras.layers.Input(shape=(max_length,),
                                                name='ref_attention',
                                                dtype='int32')

    input_ids_layer2 = tf.keras.layers.Input(shape=(max_length,),
                                            name='input_ids2',
                                            dtype='int32')
    input_attention_layer2 = tf.keras.layers.Input(shape=(max_length,),
                                                  name='input_attention2',
                                                  dtype='int32')

    input_ids_layer3 = tf.keras.layers.Input(shape=(max_length,),
                                            name='input_ids3',
                                            dtype='int32')
    input_attention_layer3 = tf.keras.layers.Input(shape=(max_length,),
                                                  name='input_attention3',
                                                  dtype='int32')

    # Roberta outputs a tuple where the first element at index 0
    # represents the hidden-state at the output of the model's last layer.
    # It is a tf.Tensor of shape (batch_size, sequence_length, hidden_size=768).
    last_hidden_state0 = transformer([input_ids_layer, input_attention_layer])[0]
    last_hidden_state1 = transformer([ref_ids_layer, ref_attention_layer])[0]
    last_hidden_state2 = transformer([input_ids_layer2, input_attention_layer2])[0]
    last_hidden_state3 = transformer([input_ids_layer3, input_attention_layer3])[0]

    # We only care about Roberta's output for the [CLS] token,
    # which is located at index 0 of every encoded sequence.
    # Splicing out the [CLS] tokens gives us 2D data.
    cls_token0 = last_hidden_state0[:, 0, :]
    cls_token1 = last_hidden_state1[:, 0, :]
    cls_token2 = last_hidden_state2[:, 0, :]
    cls_token3 = last_hidden_state3[:, 0, :]

    ##                                                 ##
    ## Define additional dropout and dense layers here ##
    ##                                                 ##

    # Define a single node that makes up the output layer (for binary classification)
    # output = tf.keras.layers.Dense(1,
    #                                activation='sigmoid',
    #                                kernel_initializer=weight_initializer,
    #                                kernel_constraint=None,
    #                                bias_initializer='zeros')(cls_token)

    prediction1 = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='sigmoid', kernel_initializer=weight_initializer,
        kernel_constraint=None, bias_initializer='zeros'),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=weight_initializer,
        kernel_constraint=None, bias_initializer='zeros')
    ])(cls_token0)

    prediction2 = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='sigmoid', kernel_initializer=weight_initializer,
        kernel_constraint=None, bias_initializer='zeros'),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=weight_initializer,
        kernel_constraint=None, bias_initializer='zeros')
    ])(cls_token1)

    prediction3 = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='sigmoid', kernel_initializer=weight_initializer,
                              kernel_constraint=None, bias_initializer='zeros'),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=weight_initializer,
                              kernel_constraint=None, bias_initializer='zeros')
    ])(cls_token2)

    prediction4 = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='sigmoid', kernel_initializer=weight_initializer,
                              kernel_constraint=None, bias_initializer='zeros'),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=weight_initializer,
                              kernel_constraint=None, bias_initializer='zeros')
    ])(cls_token3)

    prediction = tf.keras.layers.concatenate([prediction1, prediction2, prediction3, prediction4])

    output = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=weight_initializer,
                                   kernel_constraint=None, bias_initializer='zeros')(prediction)

    # Define the model
    model = tf.keras.Model([input_ids_layer, input_attention_layer, ref_ids_layer, ref_attention_layer,
                            input_ids_layer2, input_attention_layer2, input_ids_layer3, input_attention_layer3], output)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=tf.metrics.BinaryAccuracy(threshold=0.5))

    return model


# build and train the model
model = build_model(Roberta)

model.fit(x=[train_features_1_ids, train_features_1_attention, train_features_ref_ids, train_features_ref_attention,
             train_features_2_ids, train_features_2_attention, train_features_3_ids, train_features_3_attention],
          y=train_labels['accuracy'].to_numpy(),
          epochs=EPOCHS,
          batch_size=batchsz,
          validation_data=([val_features_1_ids, val_features_1_attention, val_features_ref_ids, val_features_ref_attention,
                            val_features_2_ids, val_features_2_attention, val_features_3_ids, val_features_3_attention],
                           val_labels['accuracy'].to_numpy()),
          validation_freq=1,
          verbose=1)

model.evaluate(x=[test_features_1_ids, test_features_1_attention, test_features_ref_ids, test_features_ref_attention,
                  test_features_2_ids, test_features_2_attention, test_features_3_ids, test_features_3_attention],
               y=test_labels['accuracy'].to_numpy(),
               batch_size=batchsz,
               verbose=1)

