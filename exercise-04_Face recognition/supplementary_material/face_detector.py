import cv2
from mtcnn import MTCNN
import numpy as np


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(self, tm_window_size=20, tm_threshold=0.7, aligned_image_size=224):
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference = None

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size

        # ToDo: Specify all parameters for template matching.
        self.tm_threshold = tm_threshold
        self.tm_window_size = tm_window_size

    # ToDo: Track a face in a new image using template matching.
    def track_face(self, image):

        # re-initialize
        if self.reference is None:
            self.reference = self.detect_face(image)
            return self.reference

        # last face para
        last_rect = self.reference["rect"]
        height = last_rect[2] - last_rect[0]
        width = last_rect[3] - last_rect[1]
        template = self.crop_face(self.reference["image"], last_rect)

        # 20 pixel extended range
        win_size = self.tm_window_size
        extend_top = last_rect[0] - win_size
        extend_left = last_rect[1] - win_size
        extend_bottom = last_rect[2] + 2 * win_size
        extend_right = last_rect[3] + 2 * win_size
        extend_rect = [extend_top, extend_left, extend_bottom, extend_right]
        extend_crop_img = self.crop_face(image, extend_rect)

        # locate the position of a templatein a new frame by searching for the position with
        # maximum similarity to the reference
        res = cv2.matchTemplate(extend_crop_img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val < self.tm_threshold:
            self.reference = self.detect_face(image)
            return self.reference
        else:
            # update reference
            # max_loc is relative coordinate
            face_rect = [max_loc[0] + extend_top, max_loc[1] + extend_left,
                         max_loc[0] + extend_top + height, max_loc[1] + extend_left + width]
            face_align = self.align_face(image, face_rect)
            # self.reference["aligned"] = face_align
            # self.reference["rect"] = face_rect
            # self.reference["image"] = image
            return {"rect": face_rect, "image": image, "aligned": face_align, "response": 0}


    # Face detection in a new image.
    def detect_face(self, image):
        # Retrieve all detectable faces in the given image.
        detections = self.detector.detect_faces(image)
        if not detections:
            self.reference = None
            return None

        # Select face with largest bounding box.
        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = detections[largest_detection]["box"]

        # Align the detected face.
        aligned = self.align_face(image, face_rect)
        return {"rect": face_rect, "image": image, "aligned": aligned, "response": 0}

    # Face alignment to predefined size.
    def align_face(self, image, face_rect):
        return cv2.resize(self.crop_face(image, face_rect), dsize=(self.aligned_image_size, self.aligned_image_size))

    # Crop face according to detected bounding box.
    def crop_face(self, image, face_rect):
        # top, left, bottem, right = face_rect
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]
