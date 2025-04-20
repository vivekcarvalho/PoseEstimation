import cv2
import mediapipe as mp
import time

class PoseEstimation:
    def __init__(self, mode = False
                 , smooth_landmarks = True
                 , enable_segmentation = False
                 , min_detection_confidence = 0.7
                 , min_tracking_confidence = 0.5): 
        self.mode = mode
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Initialize the Pose class
        self.mpPose = mp.solutions.pose

        self.pose = self.mpPose.Pose(static_image_mode = self.mode,
                            smooth_landmarks = self.smooth_landmarks,
                            enable_segmentation = self.enable_segmentation,
                            min_detection_confidence = self.min_detection_confidence,
                            min_tracking_confidence = self.min_tracking_confidence)

        # For Drawing landmarks
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, video, draw = True):
        # Since CV2 has BRG color space we need to convert to RGB for mediapipe
        frameRGB = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(frameRGB)
        # print(results.pose_landmarks)

        if self.results.pose_landmarks:
            if draw:
                # Draw landmarks using existing function
                self.mpDraw.draw_landmarks(video, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        return video
    
    def getPostion(self, video, draw = True):
        lmlist = []

        if self.results.pose_landmarks:
            # Get the landmarks
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = video.shape
                # print(id, lm)
                #Scale the landmark coordinates to the video size
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(video, (cx, cy), 3, (255, 255, 0), cv2.FILLED, 2)
            
        return lmlist
        
        


def main():
    pTime = 0
    # videoPath = './testData/one-by-one-person-detection.mp4'
    # videoPath = './testData/people-detection.mp4'
    # videoPath = './testData/worker-zone-detection.mp4'
    videoPath = './testData/head-pose-face-detection-female-and-male.mp4'

    cap = cv2.VideoCapture(videoPath)
    detector = PoseEstimation()

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


if __name__ == "__main__":
    main()