#!/usr/bin/python

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import mpio

class FaceDetector:
    def __init__(self, model_path, input_size=(320, 240), conf_threshold=0.6,
                 center_variance=0.1, size_variance=0.2,
                 nms_max_output_size=200, nms_iou_threshold=0.3):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.nms_max_output_size = nms_max_output_size
        self.nms_iou_threshold = nms_iou_threshold

        # Load the TFLite model
        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Initialize GPIO pin for controlling the LED
        self.led_pin = 108  # Replace with your GPIO pin number
        self.led = mpio.GPIO(self.led_pin, mpio.GPIO.OUT)

    def _pre_processing(self, img):
        resized = cv2.resize(img, self.input_size)
        image_rgb = resized[..., ::-1].astype(np.float32)
        image_norm = (image_rgb - 127.5) / 127.5  # Normalize between -1 and 1
        return image_norm[None, ...]

    def detect_faces(self, img):
        # Preprocess the image
        input_data = self._pre_processing(img)

        # Set input tensor and invoke the interpreter
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Get output tensors
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[1]['index'])[0]

        # Post-processing
        num_faces = self._post_processing(boxes, scores)

        return num_faces

    def _post_processing(self, boxes, scores):
        # Count the number of detected faces
        num_faces = np.sum(scores[:, 1] > self.conf_threshold)

        # Control the LED and print status based on the number of detected faces
        if num_faces < 14:
            self.led.set(False)  # Turn on the LED
            print("Secured")
        else:
            self.led.set(True)  # Turn off the LED
            print("Not Secured")

        return num_faces

    def cleanup(self):
        # Release GPIO resources
        self.led.close()

if __name__ == "__main__":
    # Define your model path
    model_path = '/root/version-slim-320_without_postprocessing.tflite'

    # Create an instance of the face detector
    face_detector = FaceDetector(model_path)

    # Start capturing from the camera
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame and get the number of faces
        num_faces = face_detector.detect_faces(frame)

        # Check for 'q' key press to exit
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break

    # Release the capture, destroy all windows, and clean up GPIO
    cap.release()
    cv2.destroyAllWindows()
    face_detector.cleanup()

