import math
import random

import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)




class SnakeGameClass:
    def __init__(self, pathFood):
        self.points = []  #All the points of the sname
        self.lengths = [] # Distance between points
        self.currentLength = 0 #Total Lenght of the snake
        self.allowedLength = 150 #Total allowed length
        self.previousHead = 0, 0 #Previous head point

        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 0, 0
        self.randomFoodLocation()


    def randomFoodLocation(self):
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)


    def update(self, imgMain, currentHead):
        px, py = self.previousHead
        cx, cy = currentHead

        self.points.append((cx, cy))
        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)

        self.currentLength += distance
        self.previousHead = cx, cy

        #Length Reduction
        if self.currentLength> self.allowedLength:
            for i, length in enumerate(self.lengths):
                self.currentLength -= length
                self.lengths.pop(i)
                self.points.pop(i)
                if self.currentLength < self.allowedLength:
                    break

        # Drawing the snake
        if self.points:
            for i,point in enumerate(self.points):
                if i !=0:
                    cv2.line(imgMain, self.points[i-1], self.points[i], (0,0,255), 20)
            cv2.circle(imgMain, self.points[-1], 20, (200, 0, 200), cv2.FILLED)

        return imgMain

game = SnakeGameClass()




while True:
    success, img = cap.read()
    img= cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmlist = hands[0]['lmList']
        pointIndex = lmlist[8][0:2]
        img = game.update(img, pointIndex)


    cv2.imshow("Image", img)
    cv2.waitKey(1)

