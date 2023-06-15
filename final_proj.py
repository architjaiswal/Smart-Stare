# This program will detect a person in using a Pi Camera mounted on Adafruit Pan-Tilt servo kit.
# It uses Jetson.inference libraries to detect a person and then uses the pan-tilt servo kit to track the person.

from adafruit_servokit import ServoKit
import jetson_inference
import jetson_utils

class KalmanFilter:
    def __init__(self, q, r, p=1, x=0):
        self.q = q
        self.r = r
        self.p = p
        self.x = x
    
    def update(self, measurement):
        self.p += self.q
        self.k = self.p / (self.p + self.r)
        self.x += self.k * (measurement - self.x)
        self.p *= (1 - self.k)
        return self.x
    
if __name__ == "__main__":

    kit = ServoKit(channels=16)
    yaw = 90 # pan
    pitch = 135 # tilt
    kit.servo[0].angle = yaw
    kit.servo[1].angle = pitch

    net = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
    camera = jetson_utils.videoSource("csi://0")      # '/dev/video0' for V4L2
    display = jetson_utils.videoOutput("display://0") # 'my_video.mp4' for file
    
    while display.IsStreaming():
        img = camera.Capture()
        detections = net.Detect(img)

        kalman_yaw = KalmanFilter(0.1, 0.1)
        kalman_pitch = KalmanFilter(0.1, 0.1)

        for detection in detections:
            if detection.ClassID == 1:
                center_x = int((detection.Left + detection.Right)/2)
                center_y = int((detection.Top + detection.Bottom)/2)
                # print('Person: ({}, {})'.format(center_x, center_y))
                # print screen resulution
                # print('Image: ({}, {})'.format(img.width, img.height))
                error_yaw = center_x - img.width/2
                error_pitch = center_y - img.height/2
                
                if abs(error_yaw) > 15:
                    yaw += error_yaw/60

                if abs(error_pitch) > 15:
                    pitch -= error_pitch/60

                if yaw > 180:
                    yaw = 180
                elif yaw < 0:
                    yaw = 0

                if pitch > 180:
                    pitch = 180
                elif pitch < 0:
                    pitch = 0

                # apply kalman filtering
                yaw = kalman_yaw.update(yaw)
                pitch = kalman_pitch.update(pitch)

                kit.servo[0].angle = yaw
                kit.servo[1].angle = pitch

            # print(detection)
            # print(detection.ClassID)


            

        display.Render(img)
        display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

