import face_recognition as fr
import cv2
import os
import time
import itertools
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import multiprocessing


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

def num_cam():
    len_cam = 0
    for i in range(100):
        videocapture = cv2.VideoCapture(i)
        if videocapture.isOpened() == False:
            len_cam = i
            break
        videocapture.release()
    return len_cam

def distribute_thread():
    len_cam = num_cam()
    index_th = [[], [], [], []]

    for i in range(len_cam):
        index_th[i % 4].append(i)

    return index_th

def thread(index_th, label_list, known_faces):
    # print(index_th)
    for i in index_th:
        video_catch = cv2.VideoCapture(i)
        # print('????')


        face_locations = []
        face_encodings = []
        face_names = []
        frame_number = 0
        print('%d isOpened!'%i, video_catch.isOpened())
        if not video_catch.isOpened():
            return
        while True:
            ret, frame = video_catch.read()
            # print('%s' % id, ret)

            frame_number += 1
            if not ret:
                break
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_frame = small_frame[:, :, ::-1]

            face_locations = fr.face_locations(rgb_frame)  # ,model="cnn"
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
                font = ImageFont.truetype('/home/jonty/文档/Face_Rec/STHeiti Medium.ttc', 24,
                                          encoding='utf-8')
                draw.text((left + 6, bottom - 25), name, (0, 0, 255), font=font)

                frame = np.array(pil_im)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.imshow('Video_%s' % id, frame)

            c = cv2.waitKey(1)
            if c & 0xFF == ord("q"):
                break

        video_catch.release()
        cv2.destroyWindow('Video_%s' % id)
        

def process_images_in_process_pool(index_th, number_of_cpus, label_list, known_faces):
    if number_of_cpus == -1:
        processes = None
    else:
        processes = number_of_cpus

    # macOS will crash due to a bug in libdispatch if you don't use 'forkserver'
    context = multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")

    pool = context.Pool(processes=processes)

    function_parameters = zip(
        index_th,
        itertools.repeat(label_list),
        itertools.repeat(known_faces),
    )

    pool.starmap(thread, function_parameters)

def multiprocessing_face_recognition():
    path = '/home/jonty/文档/Face_Rec/training_data'
    video_path = '/home/jonty/文档/Face_Rec/training_video/4.mp4'
    image_list, label_list = get_file(path)
    known_faces = []
    for i in range(len(image_list)):
        tmp = fr.load_image_file(image_list[i])
        known_faces.append(list(fr.face_encodings(tmp)[0]))
    # time.sleep(2)
    index_th = distribute_thread()
    print(len(index_th))
    process_images_in_process_pool(index_th, -1, label_list, known_faces)
    pass


if __name__ == '__main__':
    multiprocessing_face_recognition()
    # thread(1, index_th, label_list, known_faces)
    # thread(0, index_th, label_list, known_faces)
    '''
    p1 = Process(target=thread, args=(0, index_th, label_list, known_faces, ))
    p2 = Process(target=thread, args=(1, index_th, label_list, known_faces, ))
    p3 = Process(target=thread, args=(2, index_th, label_list, known_faces, ))
    p4 = Process(target=thread, args=(3, index_th, label_list, known_faces, ))

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    '''

