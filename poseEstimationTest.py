import cv2
import mediapipe as mp
import time
import poseEstimation as pe

pTime = 0
# videoPath = './testData/one-by-one-person-detection.mp4'
# videoPath = './testData/people-detection.mp4'
# videoPath = './testData/worker-zone-detection.mp4'
videoPath = './testData/head-pose-face-detection-female-and-male.mp4'

cap = cv2.VideoCapture(videoPath)
detector = pe.PoseEstimation()

while True:
    success, video = cap.read()
    if not success:
        print("Video not found")
        break
    else:
        # print("Video found")
        frame = detector.findPose(video, draw=False)
        lmlist = detector.getPostion(frame, draw=False)
        # print(lmlist[10])

        cTime = time.time()
        fps = 1 / (cTime -pTime)
        pTime = cTime

        cv2.putText(frame, f'FPS: {int(fps)}', (600,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if lmlist:
            # print(lmlist[10])
            cv2.circle(frame, (lmlist[0][1], lmlist[0][2]), 10, (0, 255, 255), cv2.FILLED)
        cv2.imshow('Video', frame)
        # 2 ms of delay
        # print(frame.shape)
        cv2.waitKey(2)