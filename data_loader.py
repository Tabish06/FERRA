##
## data_loader.py
## Load in brick/ball/cylinder examples for programming challenge.  
##


import numpy as np
from easydict import EasyDict
import glob
import cv2
import pdb

def data_loader(label_indices, 
                channel_means, 
                train_test_split = 0.7, 
                input_image_size = (227, 227), 
                data_path = '../data'):

    '''
    Load, resize, subtract mean, and store data in easydicts.
    '''

    num_classes = len(label_indices)

    #Covert Channel means list to array
    channel_means = np.array(channel_means)

    #Pull in image filenames:
    im_paths = glob.glob('CK+/Emotion_labels/Emotion/*/*/*.txt')
    # rs_paths = glob.glob('extended-cohn-kanade-images/cohn-kanade-images/*/*/*.png')
    # im_paths = glob.glob(data_path + '/*/*.jpg')

    #Train test split
    num_training_examples = int(np.round(train_test_split*len(im_paths)))
    num_testing_examples = len(im_paths) - num_training_examples

    random_indices = np.arange(len(im_paths))
    np.random.shuffle(random_indices)

    training_indices = random_indices[:num_training_examples]
    testing_indices = random_indices[num_training_examples:]

    #Make easydicts for data
    data = EasyDict()
    data.train = EasyDict()
    data.test = EasyDict()

    # Make empty arrays to hold data:
    data.train.X = np.zeros((num_training_examples, input_image_size[0], input_image_size[1], 3), 
                            dtype = 'float32')
    data.train.y = np.zeros((num_training_examples, num_classes), dtype = 'float32')

    data.test.X = np.zeros((num_testing_examples, input_image_size[0], input_image_size[1], 3), 
                            dtype = 'float32')
    data.test.y = np.zeros((num_testing_examples, num_classes), dtype = 'float32')

    for count, index in enumerate(training_indices):
        image_path = "extended-cohn-kanade-images/cohn-kanade-images/"+ "\\".join(im_paths[0].split("\\")[1:]).replace('_emotion.txt','.png')
        im = cv2.imread(image_path)
        # im = cv2.imread(im_paths[index])
        # pdb.set_trace()
        im = cv2.resize(im, (input_image_size[1], input_image_size[0]))
        data.train.X[count, :, :, :] = im
        f = open(im_paths[index],"r")
        cat = f.read()
        class_name = int(cat.replace('e+','').replace(' ','').replace('0','').replace('.',''))
        data.train.y[count, class_name] = 1
        
    for count, index in enumerate(testing_indices):
        image_path = "extended-cohn-kanade-images/cohn-kanade-images/"+ "\\".join(im_paths[0].split("\\")[1:]).replace('_emotion.txt','.png')
        im = cv2.imread(image_path)
        im = cv2.resize(im, (input_image_size[1], input_image_size[0]))
        data.test.X[count, :, :, :] = im
        f = open(im_paths[index],"r")
        cat = f.read()
        class_name = int(cat.replace('e+','').replace(' ','').replace('0','').replace('.',''))
        data.test.y[count, class_name] = 1

    print('Loaded', str(len(training_indices)), 'training examples and ', 
          str(len(testing_indices)), 'testing examples. ')

    return data