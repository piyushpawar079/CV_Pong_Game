import cv2, cvzone, numpy as np
from cvzone.HandTrackingModule import HandDetector
import mediapipe

cam = cv2.VideoCapture(0)

cam.set(3, 1280)
cam.set(4, 720)

imgBackground = cv2.imread('Resources/Background.png')
imgGameOver = cv2.imread('Resources/gameOver.png')
imgBall = cv2.imread('Resources/Ball.png', cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread('Resources/bat1.png', cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread('Resources/bat2.png', cv2.IMREAD_UNCHANGED)

ballSpd = [100, 100]
speedX = speedY = 15
score = [0, 0]

gameOver = False

detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:

    success, img = cam.read()
    img = cv2.flip(img, 1)
    imgRaw = img.copy()

    hands, img = detector.findHands(img, flipType=False)

    img = cv2.addWeighted(img, 0.2, imgBackground, 0.8, 0)

    if hands:

        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = imgBat1.shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 415)

            if hand['type'] == 'Left':
                img = cvzone.overlayPNG(img, imgBat1, (59, y1))
                if 59 < ballSpd[0] < 59 + w1 and y1 < ballSpd[1] < y1 + h1:
                    speedX *= -1
                    ballSpd[0] += 20
                    score[0] += 1

            if hand['type'] == 'Right':
                img = cvzone.overlayPNG(img, imgBat2, (1195, y1))
                if 1120 < ballSpd[0] < 1170 + w1 and y1 < ballSpd[1] < y1 + h1:
                    speedX *= -1
                    ballSpd[0] -= 20
                    score[1] += 1

    if ballSpd[1] >= 500 or ballSpd[1] <= 10:
        speedY *= -1

    if ballSpd[0] < 10 or ballSpd[0] > 1200:
        gameOver = True

    if gameOver:
        img = imgGameOver
        cv2.putText(img, str(max(score[0], score[1])).zfill(2), (585, 360), cv2.FONT_HERSHEY_COMPLEX, 3, (200, 0, 200), 5)

    else:
        ballSpd[0] += speedX
        ballSpd[1] += speedY

        cv2.putText(img, str(score[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(img, str(score[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)

        img = cvzone.overlayPNG(img, imgBall, ballSpd)

    img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))

    cv2.imshow('img', img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        ballSpd = [100, 100]
        speedX = speedY = 15
        score = [0, 0]
        gameOver = False
        imgGameOver = cv2.imread('Resources/gameOver.png')

