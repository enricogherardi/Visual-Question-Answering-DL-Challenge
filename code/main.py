import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import cv2
from datetime import datetime
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL.Image
from math import ceil, floor

# Program variables
TRAIN = True
SEED = 1234
img_h = 200
img_w = 350
n_channels = 3
bs = 128;
dataset_split = 0.7
num_questions = 0

tf.random.set_seed(SEED)
np.random.seed(SEED)


# Get current working directory
cwd = os.getcwd()


# Data Preparation
imgs_path = os.path.join('/content/drive/MyDrive/anndl-2020-vqa/VQA_Dataset', 'Images')
train_json_path = os.path.join('/content/drive/MyDrive/anndl-2020-vqa/VQA_Dataset', 'train_questions_annotations.json')
test_json_path = os.path.join('/content/drive/MyDrive/anndl-2020-vqa/VQA_Dataset', 'test_questions.json')


# direct dictionary, word => code
dictionary = {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        'apple': 6,
        'baseball': 7,
        'bench': 8,
        'bike': 9,
        'bird': 10,
        'black': 11,
        'blanket': 12,
        'blue': 13,
        'bone': 14,
        'book': 15,
        'boy': 16,
        'brown': 17,
        'cat': 18,
        'chair': 19,
        'couch': 20,
        'dog': 21,
        'floor': 22,
        'food': 23,
        'football': 24,
        'girl': 25,
        'grass': 26,
        'gray': 27,
        'green': 28,
        'left': 29,
        'log': 30,
        'man': 31,
        'monkey bars': 32,
        'no': 33,
        'nothing': 34,
        'orange': 35,
        'pie': 36,
        'plant': 37,
        'playing': 38,
        'red': 39,
        'right': 40,
        'rug': 41,
        'sandbox': 42,
        'sitting': 43,
        'sleeping': 44,
        'soccer': 45,
        'squirrel': 46,
        'standing': 47,
        'stool': 48,
        'sunny': 49,
        'table': 50,
        'tree': 51,
        'watermelon': 52,
        'white': 53,
        'wine': 54,
        'woman': 55,
        'yellow': 56,
        'yes': 57
}

N_CLASSES = len(dictionary)


# General Words Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
import tqdm.notebook as tq

if 'tokenizer' in globals():        
  del tokenizer

tokenizer = Tokenizer(num_words=500)

with open(train_json_path, 'r') as f:
  data = json.load(f)
  # The total number of questions
  num_questions = len(data)

  for i in tq.tqdm(data):
    quest = data[i]['question'].split(" ")
    quest[-1] = quest[-1].replace("?", "") 
    
    # Update vocabulary
    tokenizer.fit_on_texts(quest)

words_number = len(tokenizer.word_index) + 1
print(words_number)


# Custom Data Generator Class
class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, answers, imageIDs, input_questions, batch_size, training, max_length,
               shuffle=True, img_h=128, img_w=128, channels=3, img_generator=None):
    self.answers = answers
    self.imageIDs = imageIDs
    self.input_questions = input_questions
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.indexes = np.arange(len(self.answers))
    self.max_length = max_length
    self.training = training
    self.img_h = img_h
    self.img_w = img_w
    self.channels = channels
    self.img_generator = img_generator
    self.on_epoch_end()

  
  def __len__(self):
    return int(np.floor(len(self.imageIDs) / self.batch_size))

  
  def __getitem__(self, index):
    bs_index_start = index * self.batch_size;
    bs_index_end = bs_index_start + self.batch_size - 1;
    indexes = self.indexes[bs_index_start:(bs_index_end+1)]
    
    
    X =  self._generate_X(indexes)
    
    if self.training: 
      Y = self._generate_Y(indexes)
      return (X, Y)
    
    else:
      return X

  def on_epoch_end(self):
    if self.shuffle:
      np.random.shuffle(self.indexes)


  def _generate_X(self, indexes):
    RGBimages = np.empty((self.batch_size, self.img_h, self.img_w, self.channels))
    questions = np.empty((self.batch_size, self.max_length))

    for i, ID in enumerate(indexes):
      RGBimages[i, ] = self._load_image(self.imageIDs[ID], self.img_w, self.img_h)
      questions[i, ] = (self.input_questions[ID]).tolist() 

    return [RGBimages, questions]
  


  def _generate_Y(self, indexes):
    y = np.empty((self.batch_size, N_CLASSES), dtype=int)
    

    indexed_answers = [self.answers[i] for i in indexes]
    
    categorical = tf.keras.utils.to_categorical(indexed_answers, num_classes=N_CLASSES)


    for i, elem in enumerate(categorical):
      y[i] = elem;

    return y

  def _load_image(self, img_name, img_w, img_h):
    
    rgba_image = PIL.Image.open(imgs_path + '/' + img_name + ".png")
    rgb_image = rgba_image.convert('RGB')
    image = cv2.resize(np.array(rgb_image), (img_w, img_h))
    if self.img_generator is not None:
      img_t = self.img_generator.get_random_transform(image.shape, seed=SEED)
      image = self.img_generator.apply_transform(image, img_t)   
    image = image/ 255.
    
    return image

# Function to extract info from JSONs
def readTrainJson(data, first, last):
  imageIDs = []
  questions = []
  answers = []

  for i in list(data)[first:last]:
    question = data[i]['question'].split(" ") # splitting questio into words
    question[-1] = question[-1].replace("?", "") # removing question mark
    imageID = data[i]['image_id']
    answer = data[i]['answer']

    questions.append(question)
    imageIDs.append(imageID)
    answers.append(dictionary[answer]) # appending equivalent number of word
  
  return questions, imageIDs, answers

def readTestJson(data):
  questionIDs = []
  imageIDs = []
  questions = []

  for i in data:
    questionIDs.append(i)
    imageID = data[i]['image_id']
    question = data[i]['question'].split(" ") 
    question[-1] = question[-1].replace("?", "") 

    imageIDs.append(imageID)
    questions.append(question)

  return questionIDs, questions, imageIDs



# Setting all the Generators

# train and validation splitting intervals
num_train_questions = floor(num_questions * dataset_split)
num_valid_questions = num_questions - num_train_questions


with open(train_json_path, 'r') as f:
  data = json.load(f)
  (train_questions, train_imageIDs, train_answers) = readTrainJson(data, 0, num_train_questions)#;
  (valid_questions, valid_imageIDs, valid_answers) = readTrainJson(data, num_train_questions, num_questions)#;


with open(test_json_path, 'r') as f:
  test_data = json.load(f)
  (test_questionIDs, test_questions, test_imageIDs) = readTestJson(test_data)

# Transforming questions into series of tokens
train_questions_tokenized = tokenizer.texts_to_sequences(train_questions)
valid_questions_tokenized = tokenizer.texts_to_sequences(valid_questions) 
test_questions_tokenized = tokenizer.texts_to_sequences(test_questions)   


# Max length of questions
max_length_train = max(len(sequence) for sequence in train_questions_tokenized)
max_length_valid = max(len(sequence) for sequence in valid_questions_tokenized)
max_length = max(max_length_train, max_length_valid)

# Padding
train_input_questions = pad_sequences(train_questions_tokenized, maxlen=max_length)
valid_input_questions = pad_sequences(valid_questions_tokenized, maxlen=max_length) 
test_input_questions = pad_sequences(test_questions_tokenized, maxlen=max_length)


train_generator = DataGenerator(answers=train_answers, 
                                imageIDs=train_imageIDs, 
                                input_questions=train_input_questions,
                                batch_size=bs,
                                shuffle=True,
                                training=True,
                                img_h=img_h,
                                img_w=img_w,
                                channels=n_channels,
                                max_length=max_length)

valid_generator = DataGenerator(answers=valid_answers, 
                                imageIDs=valid_imageIDs, 
                                input_questions=valid_input_questions,
                                batch_size=bs,
                                shuffle=False,
                                training=True,
                                img_h=img_h,
                                img_w=img_w,
                                channels=n_channels,
                                max_length=max_length)

test_generator = DataGenerator(answers=test_questionIDs, 
                                imageIDs=test_imageIDs, 
                                input_questions=test_input_questions,
                                batch_size=1,
                                shuffle=False,
                                training=False,
                                img_h=img_h,
                                img_w=img_w,
                                channels=n_channels,
                                max_length=max_length)



#-------------------------------------
# Model Section
#-------------------------------------

# Fine Tuning Base CNN
cnn = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_h, img_w, 3))
finetuning = True
if finetuning:
    freeze_until = 10
    for layer in cnn.layers[:freeze_until]:
        layer.trainable = False
    for layer in cnn.layers[freeze_until:]:
        layer.trainable = True
else:
    cnn.trainable = False
#cnn.summary()

from tensorflow.keras import layers

# Custom CNN
def CNN(out_dim, drop_rate):
  cnn_model = tf.keras.Sequential()
  cnn_model.add(cnn)
  cnn_model.add(tf.keras.layers.GlobalAveragePooling2D())
  cnn_model.add(tf.keras.layers.Dense(units=out_dim, kernel_initializer='he_uniform'))
  cnn_model.add(tf.keras.layers.BatchNormalization())
  cnn_model.add(tf.keras.layers.Activation('relu'))
  cnn_model.add(tf.keras.layers.Dropout(0.3, seed=SEED))

  return cnn_model

# LSTM RNN
def RNN(words_number, embed_dim, max_length, drop_rate, out_dim):  
  rnn_model = tf.keras.Sequential()
  rnn_model.add(tf.keras.layers.Embedding(input_dim=words_number, output_dim=embed_dim, input_length=max_length))
  rnn_model.add(tf.keras.layers.LSTM(int(out_dim/2), return_sequences=True))
  rnn_model.add(tf.keras.layers.Dropout(drop_rate, seed=SEED))
  rnn_model.add(tf.keras.layers.LSTM(int(out_dim/2), return_sequences=False))
  rnn_model.add(tf.keras.layers.Dropout(drop_rate, seed=SEED))
  rnn_model.add(tf.keras.layers.Dense(out_dim, activation='tanh'))

  return rnn_model

# Transformer RNN
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def RNN_transformer(words_number, embed_dim, max_length, drop_rate, out_dim, num_heads):  
  inputs = layers.Input(shape=(words_number,))
  embedding_layer = TokenAndPositionEmbedding(words_number, words_number, embed_dim)
  x = embedding_layer(inputs)
  transformer_block = TransformerBlock(embed_dim, num_heads, out_dim)
  x = transformer_block(x)
  x = layers.GlobalAveragePooling1D()(x)
  x = layers.Dropout(0.1)(x)
  x = layers.Dense(20, activation="relu")(x)
  x = layers.Dropout(0.1)(x)
  outputs = layers.Dense(out_dim, activation="softmax")(x)

  model = keras.Model(inputs=inputs, outputs=outputs)

  return model

# Final VQA (CNN + LSTM-RNN)
def VQA(out_dim = 1024):
  
  CNN_net = CNN(out_dim=out_dim, drop_rate=0.2)
  RNN_net = RNN(words_number=500, embed_dim=512,
                     max_length=max_length, drop_rate=0.2, 
                     out_dim=out_dim)

  merge = tf.keras.layers.Multiply()([CNN_net.output, RNN_net.output])
  dense = tf.keras.layers.Dense(units=out_dim, kernel_initializer='he_uniform')(merge)
  batch = tf.keras.layers.BatchNormalization()(dense)
  act = tf.keras.layers.Activation('relu')(batch)
  drop = tf.keras.layers.Dropout(0.2, seed=SEED)(act)
  out = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(drop)
  VQA_model = tf.keras.models.Model(inputs=[CNN_net.input, RNN_net.input], outputs=out)

  return VQA_model

# Final VQA (CNN + Transformer-RNN)
def VQA_transformer(out_dim = 1024):

  
  CNN_net = CNN(out_dim=out_dim, drop_rate=0.2)
  RNN_net = RNN_transformer(words_number=500, embed_dim=512,
                     max_length=max_length, drop_rate=0.2, 
                     out_dim=out_dim, num_heads=4)

  merge = tf.keras.layers.Multiply()([CNN_net.output, RNN_net.output])
  dense = tf.keras.layers.Dense(units=out_dim, kernel_initializer='he_uniform')(merge)
  batch = tf.keras.layers.BatchNormalization()(dense)
  act = tf.keras.layers.Activation('relu')(batch)
  drop = tf.keras.layers.Dropout(0.2, seed=SEED)(act)
  out = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(drop)
  VQA_model = tf.keras.models.Model(inputs=[CNN_net.input, RNN_net.input], outputs=out)

  return VQA_model

# Final VQA (CNN + BERT)
!pip install keras-bert
def VQA_BERT(out_dim = 1024):

  CNN_net = CNN(out_dim=out_dim, drop_rate=0.2)
  
  BERT_net = keras_bert.get_model(
    token_num=500,
    head_num=5,
    transformer_num=12,
    embed_dim=512,
    feed_forward_dim=1024,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
  )
  RNN_net = RNN_transformer(words_number=500, embed_dim=512,  # forse word_number = word_number
                     max_length=max_length, drop_rate=0.2, 
                     out_dim=out_dim, num_heads=4)

  merge = tf.keras.layers.Multiply()([CNN_net.output, BERT_net.output])
  dense = tf.keras.layers.Dense(units=out_dim, kernel_initializer='he_uniform')(merge)
  batch = tf.keras.layers.BatchNormalization()(dense)
  act = tf.keras.layers.Activation('relu')(batch)
  drop = tf.keras.layers.Dropout(0.2, seed=SEED)(act)
  out = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(drop)
  VQA_model = tf.keras.models.Model(inputs=[CNN_net.input, BERT_net.input], outputs=out)


#VQA_net = VQA(out_dim=1024)
VQA_net = VQA_transformer(out_dim=1024)
VQA_net.summary()


#---------------------------------
# Training Parameters
#---------------------------------

# loss
loss = tf.keras.losses.CategoricalCrossentropy()
# optimizer
lr = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
# metrics
metrics = ['accuracy']

VQA_net.compile(optimizer=optimizer, loss=loss, metrics=metrics)


#---------------------------------
# Training Part
#---------------------------------
from datetime import datetime

if TRAIN:  
  exps_dir = os.path.join(cwd, 'drive/My Drive/Challenge3/')
  if not os.path.exists(exps_dir):
      os.makedirs(exps_dir)

  now = datetime.now().strftime('%b%d_%H-%M-%S')

  model_name = 'Mob-Transf-VQA'

  exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
  if not os.path.exists(exp_dir):
      os.makedirs(exp_dir)
      
  callbacks = []

  # Model checkpoint
  # ----------------
  ckpt_dir = os.path.join(exp_dir, 'ckpts')
  if not os.path.exists(ckpt_dir):
      os.makedirs(ckpt_dir)

  ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_{epoch:02d}.ckpt'), 
                                                    save_weights_only=False)  # False to save the model directly
  callbacks.append(ckpt_callback)

  # Visualize Learning on Tensorboard
  # ---------------------------------
  tb_dir = os.path.join(exp_dir, 'tb_logs')
  if not os.path.exists(tb_dir):
      os.makedirs(tb_dir)
      
  # By default shows losses and metrics for both training and validation
  tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                              profile_batch=0,
                                              histogram_freq=0)  # if 1 shows weights histograms
  callbacks.append(tb_callback)

  reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1, cooldown=0)
  callbacks.append(reduce_lr)

  # Early Stopping
  # --------------
  early_stop = False
  if early_stop:
      es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
      callbacks.append(es_callback)
  

  VQA_net.fit(x=train_generator,
            epochs=30,
            steps_per_epoch=len(train_generator),
            validation_data=valid_generator,
            validation_steps=len(valid_generator),
            callbacks=callbacks)