from PIL import Image
import face_recognition

from facenet_for_test import Facenet

if __name__ == "__main__":
    threshold = 1.1
    model = Facenet(model_path = "model_data/facenet_mobilenet.pth", threshold = threshold)
    save_path = "test_result_" + str(threshold) + ".txt"
    with open(save_path, 'a') as f:
        print("begin! path:",save_path)
    for i in range(600): 
        image1 = "test_pair/" + str(i) + "/A.jpg"
        image2 = "test_pair/" + str(i) + "/B.jpg"
        try:
            image_1 = face_recognition.load_image_file(image1)
        except:
            print('Image_1 Open Error! Try again!')
            continue
        try:
            image_2 = face_recognition.load_image_file(image2)
        except:
            print('Image_2 Open Error! Try again!')
            continue
        
        result = model.detect_image(image_1,image_2)
        with open(save_path, 'a') as f:
            f.write(str(int(result)))
            print(i)
            f.write("\n")