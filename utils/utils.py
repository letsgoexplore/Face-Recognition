import numpy as np
import torch
from PIL import Image
import face_recognition


def img_processing_save(path_in, path_out):
    image = face_recognition.load_image_file(path_in)
    result, img = img_processing(image)
    img.save(path_out)

#-----------------------------------------------------------------------------#
#   对于输入图像，使用眼睛两点法进行alignment，然后再截取头部，输出处理后的图像
#-----------------------------------------------------------------------------#
def img_processing(image):
    # 查找图像中所有面部的所有面部特征
    face_landmarks_list = face_recognition.face_landmarks(image)
    if(len(face_landmarks_list) > 0):
        face_landmark = face_landmarks_list[0]
        pic_left_eye = face_landmark['left_eye']
        center_x = 0
        center_y = 0
        for item in pic_left_eye:
            coordinate = list(item)
            center_x += coordinate[0]
            center_y += coordinate[1]
        left_center_x = center_x/len(pic_left_eye)
        left_center_y = center_y/len(pic_left_eye)

        pic_right_eye = face_landmark['right_eye']
        center_x = 0
        center_y = 0
        for item in pic_right_eye:
            coordinate = list(item)
            center_x += coordinate[0]
            center_y += coordinate[1]
        right_center_x = center_x/len(pic_right_eye)
        right_center_y = center_y/len(pic_right_eye)
        angle = np.arctan((right_center_y-left_center_y)/(right_center_x-left_center_x))*180/(np.pi)
        img = Image.fromarray(image)
        img = img.rotate(angle)
        image = np.asarray(img)
        
        # 从旋转后的图像中
        if len(face_recognition.face_locations(image))>0:
            face_location = face_recognition.face_locations(image)[0]
            top,right,bottom,left = face_location
            face_image = image[top:bottom,left:right]
            pil_image=Image.fromarray(face_image)
            return True, pil_image
        else:
            return False, image
    else:
        return False, image


#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)
        # print(nw,nh)
        image   = image.resize((nw,nh), Image.BICUBIC)
        # print(image.size)
        new_image = Image.new('RGB', size, (128,128,128))
        # print(new_image.size)
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        # print(new_image.size)
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

def get_num_classes(annotation_path):
    with open(annotation_path) as f:
        dataset_path = f.readlines()

    labels = []
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    return num_classes

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def preprocess_input(image):
    image /= 255.0 
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)