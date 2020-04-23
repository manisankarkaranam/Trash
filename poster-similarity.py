import os

from keras import applications
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import numpy as np
from pandas import DataFrame as DF
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
IMG_EXTS = ['jpg', 'jpeg', 'bmp', 'png']

def named_model(name):
    # include_top=False removes the fully connected layer at the end/top of the network
    # This allows us to get the feature vector as opposed to a classification
    if name == 'ResNet50':
        return applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    elif name == 'Xception':
        return applications.xception.Xception(weights='imagenet', include_top=False, pooling='avg')

    elif name == 'VGG16':
        return applications.vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')

    elif name == 'VGG19':
        return applications.vgg19.VGG19(weights='imagenet', include_top=False, pooling='avg')

    elif name == 'InceptionV3':
        return applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')

    elif name == 'MobileNet':
        return applications.mobilenet.MobileNet(weights='imagenet', include_top=False, pooling='avg')
    
    else:
        raise ValueError('Unrecognised model: "{}"'.format(name))
        
def _extract(fp, model):
    # Load the image, setting the size to 224 x 224
    img = image.load_img(fp, target_size=(224, 224))
    
    # Convert the image to a numpy array, resize it (1, 2, 244, 244), and preprocess it
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    # Extract the features
    np_features = model.predict(img_data)[0]
    #print(np_features.shape)
    return np_features
    # Convert from Numpy to a list of values
    #return np.char.mod('%f', np_features)
        
def extract_features(filepath, OUTPUT_PATH, model='ResNet50', write_to=None):
    print('Extracting features')
    
    # Get the model
    print('Acquiring model "{}"'.format(model), end='')
    m = named_model(model)
    print('\rAcquired model {}\t\t\t\t\t'.format(model))
    
    img_fps = []
    if not os.path.exists(filepath):
        print('Filepath does not exist: "{}"'.format(filepath))
        
    if os.path.isfile(filepath):
        ext = filepath.lower().rsplit('.', 1)[-1]
        assert ext in IMG_EXTS,\
            'Specified file "{}" is not in recognised image formats'.format(filepath)
        img_fps.append(filepath)
        
    elif os.path.isdir(filepath):
        for fn in os.listdir(filepath):
            ext = fn.lower().rsplit('.', 1)[-1]
            if ext in IMG_EXTS:
                img_fps.append(os.path.join(filepath, fn))
    else:
        raise ValueError('Filepath should be an image, or a directory containing images')
        
    img_fns = img_fps    
    print('Found {} images'.format(len(img_fns)))
    
    # Run the extraction over each image
    #features = []
    for (i, fp) in tqdm(enumerate(img_fps), position=0):
        filepath = fp.rsplit('\\', 1)[-1].rsplit('.', 1)[0] + '.npy'
        np_features = _extract(fp, m)
        np.save(os.path.join(OUTPUT_PATH, filepath), np_features)
        #features.append()
    print('\nSuccess')
    
    #print(features)
    
img_path = r'H:\Conda Py\Movie Recommendation Engine\Poster\all-shoes\101404.231.jpg'
INPUT_PATH = r'H:\Conda Py\Movie Recommendation Engine\Poster\all-shoes'
OUTPUT_PATH = r'H:\Conda Py\Movie Recommendation Engine\Poster\all-shoes-np-array'
extract_features(INPUT_PATH, OUTPUT_PATH, 'ResNet50')