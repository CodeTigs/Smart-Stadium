import cv2
import mediapipe as mp


# initialize Pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

video = cv2.VideoCapture(0)

while True:
    success, img = video.read()
    img = cv2.flip(img, 1)
    frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    try:
        results = pose.process(frameRGB)
        posePoints = results.pose_landmarks  # Changed from results.multi_pose_landmarks
        # print(posePoints)  # Uncomment this line to see the pose landmarks in the console

        if posePoints:
            mp_drawing.draw_landmarks(img, posePoints, mp_pose.POSE_CONNECTIONS)
            for id, cord in enumerate(posePoints.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)
                cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    except Exception as e:  # Avoid using a bare except clause, catch specific exceptions if possible
        print("Exception:", e)
        continue

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()