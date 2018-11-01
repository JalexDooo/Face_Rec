import cv2
import numpy as np
from multiprocessing import Process


def th1(process_name):
	video_catch= cv2.VideoCapture(0)
	while True:
		ret, frame = video_catch.read()
		
		# cv2.namedWindow("thread1")
		cv2.imshow('thread1', frame)

		c = cv2.waitKey(1)
		if c & 0xFF == ord("q"):
			break
	video_catch.release()
	cv2.destroyWindow('thread1')

def th2(process_name):
	video_catch = cv2.VideoCapture(1)
	while True:
		ret, frame = video_catch.read()
		
		# cv2.namedWindow("thread2")
		cv2.imshow('thread2', frame)

		c = cv2.waitKey(1)
		if c & 0xFF == ord("q"):
			break
	video_catch.release()
	cv2.destroyWindow('thread2')


p1 = Process(target=th1, args=('thread1', ))
p2 = Process(target=th2, args=('thread2', ))

p1.start()
p2.start()
p1.join()
p2.join()