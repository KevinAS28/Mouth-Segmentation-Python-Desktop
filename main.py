import cv2
import tensorflow as tf
import numpy as np
import datetime as dt
import os

def create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def round0(x):
  return round(x, 0)

def find_border_contours(gray_img):
    blur = cv2.GaussianBlur(gray_img, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return cnts

def show_real_only(img, prediction, min_mouth_area, max_mouth_area):
    black = cv2.resize(np.zeros_like(img), (224, 224))
    border_contours = find_border_contours(prediction)
    print(len(border_contours))
    for c in border_contours:
        x,y,w,h = cv2.boundingRect(c)
        image_ori_small = cv2.resize(img, (224, 224))
        ROI = image_ori_small[y:y+h, x:x+w]
        roi_area = ROI.shape[0]*ROI.shape[1]
        if min_mouth_area < roi_area/(image_ori_small.shape[0]*image_ori_small.shape[1]) < max_mouth_area:
          cv2.rectangle(ROI, (0, 0), (w, h), (36,255,12), 10)
          black[y:y+h, x:x+w] = ROI
    return black

print(os.getcwd())
with tf.device('/cpu:0'):
    model = tf.keras.models.load_model('saved_model224/saved_model/my_model')
    cap = cv2.VideoCapture(0)
    success, image_ori = cap.read()
    print('shape:', image_ori.shape)
    while cap.isOpened():
        success, image_ori = cap.read()
        image = np.copy(image_ori)
        image_ori1 = np.copy(image_ori)
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image_height, image_width, _ = image.shape
        image = cv2.resize(image, (224, 224))
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image1 = tf.cast(image, tf.float32) / 255.0
        prediction = np.array(tf.keras.utils.array_to_img(create_mask(model.predict(image1[tf.newaxis, ...], verbose=0))))
        prediction_size = cv2.resize(prediction, (640, 480))

        image_ori[prediction_size==0] = np.array([255,255,255], dtype=np.uint8)

        min_mouth_area, max_mouth_area = 0.1, 0.9 
        black_real = show_real_only(image_ori1, prediction, min_mouth_area, max_mouth_area)

        merged_image = np.concatenate((image_ori, cv2.resize(cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR), (640, 480))), axis=1)
        merged_image1 = np.concatenate((image_ori1, cv2.resize(black_real, (640, 480))), axis=1)
        merged_all = cv2.resize(np.concatenate((merged_image, merged_image1), axis=0), (1280, 640))
        
        cv2.imshow('Mouth segmentation', cv2.flip(merged_all, 1))
        detected_key = cv2.waitKey(5) & 0xFF
        # print(detected_key)
        if detected_key == 27:
          break
    cap.release()

