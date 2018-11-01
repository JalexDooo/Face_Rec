import face_recognition as fr
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Process

def get_file(file_dir):
    image_rule = ['png', 'jpg']
    labels = []
    images = []
    temp = []
    for root, sub_folders, files in os.walk(file_dir):
        for name in files:
            if name.split('.')[-1] in image_rule:
                images.append(os.path.join(root, name))
        for name in sub_folders:
            temp.append(os.path.join(root, name))
    i = 0
    for one_folder in temp:

        label = one_folder.split('/')[-1]
        labels = np.append(labels, label)
        i += 1
    temp = np.array([images, labels])
    image_list = list(temp[0])
    label_list = list(temp[1])
    return image_list, label_list


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 
        for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


path = '/home/jonty/文档/Face_Rec/training_data'
video_path = '/home/jonty/文档/Face_Rec/training_video/4.mp4'


image_list, label_list = get_file(path)


known_faces = []

for i in range(len(image_list)):
    tmp = fr.load_image_file(image_list[i])
    known_faces.append(list(fr.face_encodings(tmp)[0]))



video_catch = cv2.VideoCapture(0)
video_catch.set(cv2.CAP_PROP_FRAME_HEIGHT, 640);
video_catch.set(cv2.CAP_PROP_FRAME_WIDTH, 480);
video_catch.set(cv2.CAP_PROP_FPS, 30.0);


face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    ret, frame = video_catch.read()

    '''
    lighted_op = np.zeros([frame.shape[0], frame.shape[1], frame.shape[2]], frame.dtype)
    p1 = 1.9
    p2 = 20
    frame = cv2.addWeighted(frame, p1, lighted_op, 1-p1, p2)
    '''
    gamma = 1.0
    if gamma != 1.0:
        frame = adjust_gamma(frame, gamma)

    frame_number += 1

    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = small_frame[:, :, ::-1]


    face_locations = fr.face_locations(rgb_frame, model="cnn")
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    face_names = []

    for face_encoding in face_encodings:
        match = fr.compare_faces(known_faces, face_encoding, tolerance=0.45)

        name = None
        for i in range(len(label_list)):
            if match[i]:
                name = label_list[i]

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4


        pil_im = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_im)
        font = ImageFont.truetype('/home/jonty/文档/Face_Rec/STHeiti Medium.ttc', 24, encoding='utf-8')
        draw.text((left+6, bottom-25), name, (0, 0, 255), font=font)


        frame = np.array(pil_im)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.imshow('Video', frame)

    c = cv2.waitKey(1)
    if c & 0xFF == ord("q"):
        break

video_catch.release()
cv2.destroyAllWindows()