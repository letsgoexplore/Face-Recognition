import os
#------------------------------------------------#
#   图片预处理前需要生成所有文件的路径
#   注意：datasets_path是不同人脸文件夹的上级文件夹，在这里指"training_set"
#   txt_save_path指具体的存储的地址(有txt后缀)，在这里比如"data_path/training_set.txt"
#------------------------------------------------#
def data_address_log(datasets_path, txt_save_path):
    types_name      = os.listdir(datasets_path)
    types_name      = sorted(types_name)

    list_file = open(txt_save_path, 'w')
    for cls_id, type_name in enumerate(types_name):
        photos_path = os.path.join(datasets_path, type_name)
        if not os.path.isdir(photos_path):
            continue
        photos_name = os.listdir(photos_path)

        for photo_name in photos_name:
            list_file.write(str(cls_id) + ";" + '%s'%(os.path.join(os.path.abspath(datasets_path), type_name, photo_name)))
            list_file.write('\n')
    list_file.close()
