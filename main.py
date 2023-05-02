import os
import warnings
import numpy as np
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import gc

import random

import tensorflow as tf
from keras.layers import Input, Dense, Activation, Dropout, Concatenate
from keras.models import Model
from keras.utils import plot_model
from keras import callbacks
from keras import metrics
from keras_ding import KerasDing
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

NUM_TRAIN = 50000
VAL_SPLIT = 0.2
PATIENCE = 15
DROPOUT_RATE = 0.2
BATCH_SIZE = 0
EPOCHES = 100
SEED = 2023
EMPTY_SUBSAMPLE = 1/7

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_classes(traindata,feature,before,fs=8, show_percents=True, color_palette='Set3'):
        f, ax = plt.subplots(1, 1, figsize=(2 * fs, 4))
        total = float(len(traindata))
        g = sns.countplot(x=traindata[feature], order=traindata[feature].value_counts().index, palette=color_palette)
        g.set_title("Train Data: Number and percentage of labels for each class of {}".format(feature))
        if show_percents:
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x() + p.get_width() / 2.,
                        height + 3,
                        '{:1.2f}%'.format(100 * height / total),
                        ha="center")
        if before:
            plt.savefig('plots/distribution_before.png', bbox_inches='tight')
            plt.close()
        else:
            plt.savefig('plots/distribution_after.png', bbox_inches='tight')
            plt.close()
        

def plot_heatmaps(traindata):
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    t = pd.DataFrame(traindata.groupby(['classes_wild', 'hour'])['seq_id'].count().reset_index())
    m = t.pivot(index='hour', columns='classes_wild', values='seq_id')
    s = sns.heatmap(m, linewidths=.1, linecolor='black', annot=True, cmap="YlGnBu", fmt='.10g')
    s.set_title('Train: Number of wild animals observed per hour', size=16)
    plt.savefig('plots/heatmap_hour.png', bbox_inches='tight')
    plt.close()

    tmp = traindata[traindata['classes_wild'] != 'empty']
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    t = pd.DataFrame(tmp.groupby(['classes_wild', 'month'])['seq_id'].count().reset_index())
    m = t.pivot(index='month', columns='classes_wild', values='seq_id')
    s = sns.heatmap(m, linewidths=.1, linecolor='black', annot=True, cmap="YlGnBu", fmt='.10g')
    s.set_title('Train: Number of wild animals observed per month', size=16)
    plt.savefig('plots/heatmap_month.png', bbox_inches='tight')
    plt.close()

def ourmodel():
    inp_mlp = Input(shape=(6,), name="inp_mlp")
    mlp_net = Dense(16, input_shape=(6,), kernel_initializer = 'random_normal')(inp_mlp)
    mlp_net = Activation('relu')(mlp_net)

    mlp_net = Dropout(rate = DROPOUT_RATE, seed = SEED)(mlp_net)
    mlp_net = Dense(32, kernel_initializer = 'random_normal')(mlp_net)
    mlp_net = Activation('relu')(mlp_net)

    mlp_net = Dropout(rate = DROPOUT_RATE, seed = SEED)(mlp_net)
    mlp_net = Dense(32, kernel_initializer = 'random_normal')(mlp_net)
    mlp_net = Activation('relu')(mlp_net)

    mlp_net = Dropout(rate = DROPOUT_RATE, seed = SEED)(mlp_net)
    mlp_net = Dense(23, kernel_initializer = 'random_normal')(mlp_net)
    mlp_out = Activation('relu')(mlp_net)


    ### EFFNET
    inp_pretrained = Input(shape=(224, 224, 3),name="inp_pretrained")
    pretrained_import = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights = 'imagenet',
        input_shape=(224, 224, 3),
        pooling = 'avg'
    )
    for layer in pretrained_import.layers:
        layer.trainable=False

    pretrained_model = pretrained_import(inp_pretrained)
    pretrained_model = Dropout(rate = DROPOUT_RATE, seed = SEED)(pretrained_model)
    pretrained_model = Dense(23, kernel_initializer = 'random_normal')(pretrained_model)
    pretrained_out = Activation('relu')(pretrained_model)


    ### COMBINE MLP AND EFFNET
    combined = Concatenate()([pretrained_out,mlp_out])

    combined = Dense(46, kernel_initializer = 'random_normal')(combined)
    combined = Activation('relu')(combined)

    combined = Dropout(rate = DROPOUT_RATE, seed = SEED)(combined)
    combined_out = Dense(23, activation= 'softmax', kernel_initializer = 'random_normal')(combined)
    return Model(inputs = [inp_pretrained, inp_mlp], outputs = combined_out)

def results(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.grid()
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('plots/accuracy.png', bbox_inches='tight')
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.grid()
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('plots/loss.png', bbox_inches='tight')
    plt.close()
    # summarize history for precision
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.grid()
    plt.title('model precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('plots/precision.png', bbox_inches='tight')
    plt.close()
    # summarize history for recall
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.grid()
    plt.title('model recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('plots/recall.png', bbox_inches='tight')
    plt.close()
    # summarize history for auc
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.grid()
    plt.title('model auc')
    plt.ylabel('auc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('plots/auc.png', bbox_inches='tight')
    plt.close()
    # summarize history for prc
    plt.plot(history.history['prc'])
    plt.plot(history.history['val_prc'])
    plt.grid()
    plt.title('model prc')
    plt.ylabel('prc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('plots/prc.png', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    os.environ["TF_GPU_ALLOCATOR"] = 'cuda_malloc_async'
    tf.config.optimizer.set_jit(True)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # print(physical_devices[0])
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    pd.set_option('display.max_columns', None)

    ### READ TRAIN & TEST
    train_df = pd.read_csv('train.csv')

    ### ADD CLASSES NAMES
    classes_wild = {0: 'empty', 1: 'deer', 2: 'moose', 3: 'squirrel', 4: 'rodent', 5: 'small_mammal', \
                    6: 'elk', 7: 'pronghorn_antelope', 8: 'rabbit', 9: 'bighorn_sheep', 10: 'fox', 11: 'coyote', \
                    12: 'black_bear', 13: 'raccoon', 14: 'skunk', 15: 'wolf', 16: 'bobcat', 17: 'cat', \
                    18: 'dog', 19: 'opossum', 20: 'bison', 21: 'mountain_goat', 22: 'mountain_lion'}
    train_df['classes_wild'] = train_df['category_id'].apply(lambda cw: classes_wild[cw])

    ### COUNT HOW MANY ARE UNIQUE
    train_cnt_classes_images = train_df.classes_wild.nunique()
    print("Train: There are {} classes of images".format(train_cnt_classes_images))
    print(pd.DataFrame(train_df.classes_wild.value_counts()).transpose())

    ### CHANGE DATE FORMAT
    try:
        train_df['date_time'] = pd.to_datetime(train_df['date_captured'], errors='coerce')
        train_df["year"] = train_df['date_time'].dt.year
        train_df["month"] = train_df['date_time'].dt.month
        train_df["day"] = train_df['date_time'].dt.day
        train_df["hour"] = train_df['date_time'].dt.hour
        train_df["minute"] = train_df['date_time'].dt.minute
    except Exception as ex:
        print("Exception:".format(ex))
    
    train_df = train_df.astype({"year": "Int64", "month": "Int64", "day": "Int64", "hour": "Int64", "minute": "Int64"})

    ###PLOT
    plot_classes(train_df,'classes_wild',before=True)
    plot_heatmaps(train_df)
    
    ### DROP UNNESCESSARY COLUMNS 
    train_df_input = train_df.drop(
        columns=['date_captured', 'rights_holder', 'width', 'height', 'date_time', 'classes_wild',
                'date_captured', 'frame_num', 'id', 'seq_id', 'seq_num_frames'])


    ### CONSTRUCT THE DATA FOR TRAINING
    imagedata_train = []    # List that hold img,tab,predictionvector
    random.seed(SEED)

    train_df_subsample = pd.DataFrame() # df which will hold the distribution of classes
    train_df_subsample['category_id'] = np.nan

    counter = 0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    wb = cv2.xphoto.createSimpleWB()
    wb.setP(0.4)

    for i in range(NUM_TRAIN):
        

        # check if empty class, if yes, keep with some probability
        vals = train_df_input.iloc[[i]].values.tolist()[0]
        if vals[0] == 0:
            if random.random() > EMPTY_SUBSAMPLE:
                continue
        
        print(counter) # print cycle
        counter += 1

        # get the image
        img =  cv2.imread("train_images/" + train_df_input.at[i, 'file_name'],cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224)) 

        if (len(img.shape)<3):
            img = np.repeat(img[..., np.newaxis], 3, -1)

        img_wb = wb.balanceWhite(img)
        img_lab = cv2.cvtColor(img_wb, cv2.COLOR_BGR2Lab)

        l, a, b = cv2.split(img_lab)
        res_l = clahe.apply(l)
        res = cv2.merge((res_l, a, b))
        res = cv2.cvtColor(res, cv2.COLOR_Lab2BGR)
        img = res

        # get prediction vector
        pred = np.zeros((23,), dtype=int)
        pred[vals[0]] = 1

        # update the data
        train_df_subsample = train_df_subsample.append({'category_id':vals[0]},ignore_index = True)
        imagedata_train.append([img, np.array(vals[2:]), pred])

    # get the new distribution
    train_df_subsample['classes_wild'] = train_df_subsample['category_id'].apply(lambda cw: classes_wild[cw])
    plot_classes(train_df_subsample,'classes_wild',before=False)

    with open('data_count.txt', 'w') as f:
        f.write('%d' % counter)

    ### SPLIT IMAGE, TABULAR DATA, AND PREDICTION OF TRAINING
    train_category = [item[2] for item in imagedata_train]
    train_im =[item[0] for item in imagedata_train ]
    train_tab =[item[1] for item in imagedata_train ]

    del imagedata_train
    del train_df_input
    del train_df
    del train_df_subsample
    gc.collect()

    ### NETWORK
    model = ourmodel()

    ### SHOW MODEL ARCHITECTURE AND SUMMARY
    plot_model(model, "model.png", show_shapes=True,  show_layer_names = True)
    print(model.summary())

    ### CALLBACK AND CHECKPOINT
    callback = callbacks.EarlyStopping(monitor='val_loss', patience = PATIENCE, restore_best_weights = True, verbose = 2)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='checkpoints/EfficientNetB3_23_03_50K.epoch{epoch:02d}-val_loss{val_loss:.3f}.hdf5',
        save_weights_only=True,
        monitor='val_loss',
        verbose = 2,
        save_best_only=True)

    METRICS = [
        metrics.TruePositives(name='tp'),
        metrics.FalsePositives(name='fp'),
        metrics.TrueNegatives(name='tn'),
        metrics.FalseNegatives(name='fn'), 
        metrics.CategoricalAccuracy(name="accuracy"),
        metrics.Precision(name='precision'),
        metrics.Recall(name='recall'),
        metrics.AUC(name='auc'),
        metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]

    
    ### RUN
    with tf.device('/gpu:0'):
        model.compile(
            optimizer = "Adam",
            loss = 'categorical_crossentropy',
             metrics = METRICS,
             jit_compile = True)
        
        history = model.fit(
            x = {"inp_pretrained": np.array(train_im), "inp_mlp": np.array(train_tab)},
            y = np.array(train_category),
            validation_split = VAL_SPLIT,
            shuffle = True, 
            batch_size = BATCH_SIZE,
            epochs = EPOCHES,
            callbacks= [callback,model_checkpoint_callback,KerasDing()],
            use_multiprocessing = True,
            workers = 8,
            verbose = 2)

    results(history)
    
