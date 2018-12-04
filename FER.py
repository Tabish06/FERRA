from keras.models import model_from_json
import numpy as np
import cv2
import argparse
import os
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
import imutils
from imutils.video import VideoStream


class FacialExpressionModel(object):
    # label_indices = {"surprise":1,"fear":2,"disgust":3,"happiness":4,"sadness":5,"anger":6,"neutral":0}

    EMOTIONS_LIST = ["neutral", "surprise","fear" ,"disgust", "happiness", "sadness", "anger"]

    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weights_file)
        print("Model loaded from disk")
        self.loaded_model.summary()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("fps")
args = parser.parse_args()
cap = cv2.VideoCapture(s.path.abspath(0) if not args.source == 'webcam' else 0)
# cap = VideoStream(1).start()
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
cap.set(cv2.CAP_PROP_FPS, int(args.fps))

##
detector    = dlib.get_frontal_face_detector()
predictor   = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa          = FaceAligner(predictor, desiredFaceWidth = 256)


def getdata():
    _, fr = cap.read()
    # fr = cap.read()
    # fr = imutils.rotate(fr, 90)
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    #faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    # detect faces in the grayscale frame
    faces = detector(gray, 0)
    return faces, fr, gray


def start_app(cnn):
    while cap.isOpened():
        faces, fr, gray_fr = getdata()
        #for (x, y, w, h) in faces:
        for rect in faces:

            shape = predictor(gray_fr, rect)
            shape = face_utils.shape_to_np(shape)
            faceAligned = fa.align(fr, gray_fr, rect)
            backtorgb = cv2.resize(faceAligned, (94, 127))

            (x, y, w, h) = rect_to_bb(rect)
            #fc = gray_fr[y:y + h, x:x + w]
            #roi = cv2.resize(fc, (94, 127))
            #backtorgb = cv2.cvtColor(roi,cv2.COLOR_GRAY2RGB)
            pred = cnn.predict_emotion(backtorgb[np.newaxis, :, :,:])

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 1)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 1)

        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('Facial Emotion Recognition', fr)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = FacialExpressionModel("model.json", "weights.h5")
    start_app(model)