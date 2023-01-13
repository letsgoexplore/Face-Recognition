from utils.utils import img_processing_save
from utils.data_address_log import data_address_log

if __name__ == "__main__":
    #------------------------------------------------#
    #   生成训练集中各个图片的地址txt
    #   目前的方案没有解决的一个问题是如果地址是数字型的，它排列的顺序是0，1，10，100，101...
    #   一种潜在的解决方案是再data_address_log中使用多key的sorted
    #------------------------------------------------#
    path = "data_path/training_set.txt"
    data_address_log("training_set", path)

    # with open(path,"r") as f:
    #     lines = f.readlines()
    #
    # for line in lines:
    #     line = line.rstrip('\n')
    #     print(line)
    #     img_processing_save(line, line)