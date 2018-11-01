import face_recognition as fr
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Process


class multiprocessing_thread(object):
	"""docstring for multiprocessing_thread"""
	def __init__(self, real_time=True):
		super(multiprocessing_thread, self).__init__()
		self.index_th = self.distribute_thread()
		self.real_time = real_time
		self.path = '/home/jonty/文档/Face_Rec/training_data'
		self.video_path = '/home/jonty/文档/Face_Rec/training_video/4.mp4'
		self.image_list, self.label_list = self.get_file(self.path)
		self.known_faces = []
		for i in range(len(self.image_list)):
			tmp = fr.load_image_file(self.image_list[i])
			self.known_faces.append(list(fr.face_encodings(tmp)[0]))

		self.do()

	def do(self):
		p1 = Process(target=self.th0, args=('thread_1', ))
		p2 = Process(target=self.th1, args=('thread_2', ))
		p3 = Process(target=self.th2, args=('thread_3', ))
		p4 = Process(target=self.th3, args=('thread_4', ))

		p1.start()
		p2.start()
		p3.start()
		p4.start()

		p1.join()
		p2.join()
		p3.join()
		p4.join()

	def get_file(self, file_dir):
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


	def adjust_gamma(self, image, gamma=1.0):
		invGamma = 1.0 / gamma
		table = np.array([((i / 255.0) ** invGamma) * 255 
		for i in np.arange(0, 256)]).astype("uint8")

		return cv2.LUT(image, table)

	def num_cam(self):
		len_cam = 0
		for i in range(100):
			videocapture = cv2.VideoCapture(i)
			if videocapture.isOpened() == False:
				len_cam = i
				break
		return len_cam
	
	def distribute_thread(self):
		len_cam = self.num_cam()
		max_num_th = int(len_cam / 4) + 1
		index_th = [[], [], [], []]

		for i in range(len_cam):
			index_th[i%4].append(i)
		
		return index_th

	def th0(self, process_name):
		list_ = self.index_th[0]
		print("thread %s has not one camera to run!"%process_name) if len(list_) == 0 else print('%s OK!'%process_name)
		for i in list_:
			multiprocessing_face_rec(i, self.label_list, self.known_faces)
		pass

	def th1(self, process_name):
		list_ = self.index_th[1]
		print("thread %s has not one camera to run!"%process_name) if len(list_) == 0 else print('%s OK!'%process_name)
		for i in list_:
			multiprocessing_face_rec(i, self.label_list, self.known_faces)
		pass

	def th2(self, process_name):
		list_ = self.index_th[2]
		print("%s has not one camera to run!"%process_name) if len(list_) == 0 else print('%s OK!'%process_name)
		for i in list_:
			multiprocessing_face_rec(i, self.label_list, self.known_faces)
		pass

	def th3(self, process_name):
		list_ = self.index_th[3]
		print("%s has not one camera to run!"%process_name) if len(list_) == 0 else print('%s OK!'%process_name)
		for i in list_:
			multiprocessing_face_rec(i, self.label_list, self.known_faces)
		pass


class multiprocessing_face_rec(object):
	"""docstring for multiprocessing_face_rec"""
	def __init__(self, id, label_list, known_faces, real_time=True):
		super(multiprocessing_face_rec, self).__init__()
		self.real_time = real_time
		self.id = id
		self.face_rec(label_list, known_faces)

	def face_rec(self, label_list, known_faces):
		video_catch = cv2.VideoCapture(self.id)
		'''
		video_catch.set(cv2.CAP_PROP_FRAME_HEIGHT, 640);
		video_catch.set(cv2.CAP_PROP_FRAME_WIDTH, 480);
		video_catch.set(cv2.CAP_PROP_FPS, 30.0);
		'''
		
		face_locations = []
		face_encodings = []
		face_names = []
		frame_number = 0

		while True:
			ret, frame = video_catch.read()


			frame_number += 1
			if not ret:
				break
			small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
			rgb_frame = small_frame[:, :, ::-1]

			"""
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
			"""
			cv2.imshow('Video_%s'%self.id, frame)

			c = cv2.waitKey(1)
			if c & 0xFF == ord("q"):
				break
		video_catch.release()
		cv2.destroyAllWindows()



mul_thread = multiprocessing_thread()
# print(mul_thread.index_th)
