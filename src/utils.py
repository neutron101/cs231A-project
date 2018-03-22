import os
import numpy as np
import scipy.misc as nd
from attribute_generator import AttributeGenerator
from imagedata import *
import matplotlib.pyplot as plt
import math
import imageio as io
import numbers
import shutil
from inference_mine import infer
from segmentor import Segmentor
import skimage.transform as tr

RAW_IMAGE_DIR_NAME = 'raw-images'
RAW_SILHOUETTES_DIR_NAME = 'raw-silhouettes'

IMAGE_DIR_NAME = 'images'
SILHOUETTES_DIR_NAME = 'silhouettes'

RAW_DATA_FILENAME = 'raw_data'

files_at_dir = lambda base: [d for d in os.listdir(base) if d[0] != '.'] if os.path.exists(base) else []
filepath_at_dir = lambda base: [os.path.join(base, file) for file in files_at_dir(base) if os.path.isfile(os.path.join(base, file))]
dirpath_at_dir = lambda base: [os.path.join(base, file, "") for file in files_at_dir(base) if os.path.isdir(os.path.join(base, file))]
dirs_at_dir = lambda base: [file for file in files_at_dir(base) if os.path.isdir(os.path.join(base, file))]

sorter = lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else x.split('.')[0]


def filepath_at_dir_sorted(base,comparator): 
    result = [os.path.join(base, file) for file in [f for f in sorted(files_at_dir(base), key=comparator) if os.path.isfile(os.path.join(base, f))]]
    return result


def load_data(data_dir, selection_classes=[]):
    
    if len(selection_classes) <= 0:
        class_paths = dirpath_at_dir(data_dir)
    else:
        class_paths = []
        for sel in selection_classes:
           class_paths.append(os.path.join(data_dir, sel, '')) 

    load_data_for_classes(class_paths)



def load_data_for_classes(class_dir_path):

    for path in class_dir_path:        
        attributes = load_raw_attributes(path)

        if not os.path.exists(os.path.join(path, IMAGE_DIR_NAME)):
            img_files = filepath_at_dir_sorted(os.path.join(path, RAW_IMAGE_DIR_NAME), sorter)
            if len(img_files) > 0:

                try:   
                    index = 1
                    os.makedirs(os.path.join(path, IMAGE_DIR_NAME))
                    scale = 1.0
                    for img_file_name in img_files:       
                        img = io.imread(img_file_name)

                        if np.max(img.shape[0:2]) > 500:
                            scale = 500.0 / np.max(img.shape[0:2])
                            img = tr.rescale(img, scale)

                        save_image(os.path.join(path, IMAGE_DIR_NAME), '%d.%s'%(index,extract_file_ext(img_file_name)), img)
                        index+=1

                    attributes['scale'] = attributes['scale']*(1/scale)

                except Exception as e:
                    if os.path.exists(os.path.join(path, IMAGE_DIR_NAME)):
                        shutil.rmtree(os.path.join(path, IMAGE_DIR_NAME))
                    raise e

                format_and_save(path, attributes)


        if not os.path.exists(os.path.join(path, SILHOUETTES_DIR_NAME)):
            mask_and_save_silhouettes(path)


def load_images_of_type(class_dir, dir_name=IMAGE_DIR_NAME):

    images = []
    file_paths = filepath_at_dir_sorted(os.path.join(class_dir, dir_name), sorter)
    for img_file_name in file_paths:       
        img = io.imread(img_file_name)
        images.append(img)

    return images, file_paths



def format_and_save(class_dir_path, attributes):

    FramePropertyWriter.save(class_dir_path, \
        attributes['true_volume'], \
        attributes.get('scale', None), \
        attributes.get('pixel_width', None))
    

def load_raw_attributes(att_dir):
    
    with open(os.path.join(att_dir, RAW_DATA_FILENAME), 'r') as f:
        data = np.array(f.read().split("\n"))

    if data.shape[0] > 0:
        attr = {}
        attr['scale'] = 1.0

        prevIndex = -1
        for line in data:
            if len(line.strip()) > 0:

                values = line.split()
                (type, index) = values[0].split('-')
                
                if type in ['T', 'E', 'I', 'P', 'Pr', 'Pv']:
                    if index != prevIndex:
                        #print "Reading record no. ", index

                        if prevIndex > -1:
                            Ts.append(RT)
                            Ps.append(P)
                            Ks.append(K)

                            RT = np.zeros([3,4])
                            K = np.zeros([3,3])
                            P = np.zeros([3,4])

                        prevIndex = index

                        tIndex = 0
                        pIndex = 0
                        iIndex = 0


                    if type == 'T':
                        RT[tIndex,:] = np.array(values[1:])
                        tIndex+=1
                    elif type == 'E':
                        pass
                    elif type == 'P':
                        P[pIndex,:] = np.array(values[1:])
                        pIndex+=1
                    elif type == 'Pr':
                        pass
                    elif type == 'Pv':
                        pass
                    else:
                        K[iIndex,:] = np.array(values[1:])
                        iIndex+=1
                else:
                    if type.lower() == 'true_volume':
                        if len(values[1].strip()) <= 0:
                            continue;
                        attr_data = {}
                        for data in values[1:]:
                            key, value = data.split('=')
                            attr_data[key] = value

                        attr['true_volume'] = attr_data
                    elif type.lower() == 'pixel_width':
                        if len(values[1].strip()) <= 0:
                            continue;

                        attr['pixel_width'] = float(values[1].split('=')[1])
                    elif type.lower() == 'scale':
                        if len(values[1].strip()) <= 0:
                            continue;

                        attr['scale'] = float(values[1].split('=')[1])
                    else:
                        print "Unknown type", type 

    
    return attr



def diff(x):
    return x[1] - x[0]



def save_image(base_path, filename, im):
    io.imwrite(os.path.join(base_path, filename), im)



def save_np_data(base_path, filename, data):
    np.save(os.path.join(base_path, filename), data)



def load_np_data(base_path, dir_name=SILHOUETTES_DIR_NAME):
    coll = []
    file_paths = filepath_at_dir_sorted(os.path.join(base_path, dir_name), sorter)
    for img_file_name in file_paths:
        data = np.load(img_file_name)
        coll.append(data)

    return coll, file_paths



def mask_and_save_silhouettes(base_path):

    if not os.path.exists(os.path.join(base_path, RAW_SILHOUETTES_DIR_NAME)) and os.path.exists(os.path.join(base_path, IMAGE_DIR_NAME)):
            image_paths = filepath_at_dir_sorted(os.path.join(base_path, IMAGE_DIR_NAME), sorter)
            if len(image_paths)>0:
                os.makedirs(os.path.join(base_path, RAW_SILHOUETTES_DIR_NAME))
                segment = Segmentor()
                try:
                    for path in image_paths:
                        segment.segment_images_at_path(path, os.path.join(base_path, RAW_SILHOUETTES_DIR_NAME, ''))
                except Exception as e:
                    if os.path.exists(os.path.join(base_path, RAW_SILHOUETTES_DIR_NAME)):
                        shutil.rmtree(os.path.join(base_path, RAW_SILHOUETTES_DIR_NAME))
                    raise e
 

    silhouettes, file_paths = load_images_of_type(base_path, RAW_SILHOUETTES_DIR_NAME)
    if len(silhouettes) > 0:
        os.makedirs(os.path.join(base_path, SILHOUETTES_DIR_NAME))

        index = 1
        for sil in silhouettes:
            sil = np.array(sil)
            new_sil = np.zeros([sil.shape[0], sil.shape[1]])
            
            mask = lambda im: im[:,:]>0 
            
            if sil.shape[2] > 1:
                new_sil[mask(sil[:,:,0]) | mask(sil[:,:,1]) | mask(sil[:,:,2])] = 1 
                save_np_data(os.path.join(base_path, SILHOUETTES_DIR_NAME), '%d.npy'%index, new_sil)
                index+=1



def strip_file_name(filepath):
    return filepath.split("/")[-1].split(".")[0]



def extract_file_ext(filepath):
    return filepath.split("/")[-1].split(".")[1]
                
