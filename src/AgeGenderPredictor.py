#
#       顔検出 + 性別・年齢予測
# 
import argparse
import os.path as os
import numpy as np
from pathlib import Path

import cv2
from PIL import Image
from skimage.transform import resize

import tensorflow as tf
from keras.utils.data_utils import get_file
from tensorflow.compat.v1 import GPUOptions,Session,ConfigProto
from wide_resnet import WideResNet

import align.detect_face

class AgeGenderPredictor():
    # コンストラクタ
    #   特定画像に依存する情報はここで処理しない
    #   モデルの情報のみロードする
    #
    # input
    def __init__(self):
        print("__init__")

    # 性別・年齢を表記する
    #
    # input
    #   image  キャプションを付与する画像 Image データ
    def put_caption(self, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=0.3, thickness=1):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (0,255,255), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, 
                    (0, 0, 0), thickness, lineType=cv2.LINE_AA)


    # 顔検出(MTCNN)
    def detect_face(self, Img, image_size):
        minsize = 20
        threshold = [ 0.6, 0.7, 0.7 ]  
        factor = 0.709 
        margin = 44
        gpu_memory_fraction = 1.0
    
        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = Session(config=ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                dir_model="./align"
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, dir_model)
    
                Img_size = np.asarray(Img.shape)[0:2]
                bounding_boxes, _ = align.detect_face.detect_face(Img, minsize, pnet, rnet, onet, threshold, factor)
                faces = np.zeros((len(bounding_boxes), image_size, image_size, 3), dtype = "uint8")
                bb = np.zeros((len(bounding_boxes), 4), dtype=np.int32)
                for i in range(len(bounding_boxes)):            
                    det = np.squeeze(bounding_boxes[i,0:4])
                    bb[i, 0] = np.maximum(det[0]-margin/2, 0)
                    bb[i, 1] = np.maximum(det[1]-margin/2, 0)
                    bb[i, 2] = np.minimum(det[2]+margin/2, Img_size[1])
                    bb[i, 3] = np.minimum(det[3]+margin/2, Img_size[0])
                    cropped = Img[bb[i, 1]:bb[i, 3],bb[i, 0]:bb[i, 2],:]
                    img_cropped = Image.fromarray(cropped)
                    img_aligned = img_cropped.resize((image_size, image_size),Image.BILINEAR)
                    aligned_arr = np.asarray(img_aligned)
                    faces[i, :, :, :] = cv2.cvtColor(aligned_arr, cv2.COLOR_BGR2RGB)
        return faces, bb


    # 性別・年齢予測
    def predict(self, faces):    
        Ages, Genders = [],[]
        if len(faces) == 0:
            return Ages, Genders
   
        #if os.isdir("model") == False:
        #    pre_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
        #    modhash = 'fbe63257a054c1c5466cfd7bf14646d6'
        #    weight_file = get_file("weights.28-3.73.hdf5", pre_model, cache_subdir="model",
        #                           file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))
        #else:
        #    weight_file = "model/weights.28-3.73.hdf5"            
        weight_file="../conf/age_gender_estimator/weights.28-3.73.hdf5"

        img_size = np.asarray(faces.shape)[1]
        model = WideResNet(img_size, depth=16, k=8)()
        model.load_weights(weight_file)
        
        # 予測
        results = model.predict(faces)
        Genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        Ages = results[1].dot(ages).flatten()
        return Ages, Genders

    # 入力画像ファイルに対して、性年代推定結果を出力します
    #
    # input
    #   input_path  入力画像ファイルのパス
    #   output_paty 出力画像ファイルのパス
    def predict_file(self, input_path, output_path):
        print("input_path:",input_path)
        img = cv2.imread(input_path) #入力画像
        img_size = 64

        faces, bb = self.detect_face(img, img_size)    
        Ages, Genders = self.predict(faces)
        print("ages:",Ages)
        print("Genders:",Genders)
        
        for face in range(len(faces)):        
            cv2.rectangle(img,(bb[face, 0], bb[face, 1]),(bb[face, 2], bb[face, 3]),(0,255,255),2)
            label = "{}, {}".format(int(Ages[face]), "Male" if Genders[face][0] < 0.5 else "Female")
            self.put_caption(img, (bb[face, 0], bb[face, 1]), label)

        # 出力画像の保存        
        cv2.imwrite(output_path, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facial age,gender recognizer')
    parser.add_argument('--input', '-i', default=None, required=True,type=str,
                        help='input file path')
    parser.add_argument('--output', '-o', default=None, required=True,type=str,
                        help='output file path')
    args = parser.parse_args()

    predictor = AgeGenderPredictor()
    predictor.predict_file(args.input, args.output)


