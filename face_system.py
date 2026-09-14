"""
Core face detection and recognition for Raspberry Pi.
Lightweight and optimized for edge devices.
"""

import cv2
import os
import numpy as np
import pickle


class FaceSystem:
    def __init__(self, known_dir="known_faces"):
        self.known_dir = known_dir
        os.makedirs(known_dir, exist_ok=True)

        # Haar cascade (fast on Raspberry Pi)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)

        # LBPH recognizer (lightweight)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_to_name = {}
        self.name_to_label = {}
        self.next_label = 0
        self.model_path = "lbph_model.yml"
        self.label_path = "label_map.pkl"

        if os.path.exists(self.model_path) and os.path.exists(self.label_path):
            self.load_model()

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces

    def enroll(self, face_img, name):
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, (100, 100))

        if name not in self.name_to_label:
            self.name_to_label[name] = self.next_label
            self.label_to_name[self.next_label] = name
            self.next_label += 1

        label = self.name_to_label[name]
        self.recognizer.update([face_img], np.array([label]))
        self.save_model()
        print(f"Enrolled: {name}")

    def predict(self, face_img):
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, (100, 100))

        try:
            label, confidence = self.recognizer.predict(face_img)
        except Exception:
            return "Unknown", 999.0

        if confidence > 70 or label not in self.label_to_name:
            return "Unknown", confidence
        return self.label_to_name[label], confidence

    def save_model(self):
        self.recognizer.write(self.model_path)
        with open(self.label_path, "wb") as f:
            pickle.dump({
                "label_to_name": self.label_to_name,
                "name_to_label": self.name_to_label,
                "next_label": self.next_label
            }, f)

    def load_model(self):
        self.recognizer.read(self.model_path)
        with open(self.label_path, "rb") as f:
            data = pickle.load(f)
            self.label_to_name = data["label_to_name"]
            self.name_to_label = data["name_to_label"]
            self.next_label = data["next_label"]
        print(f"Loaded model with {len(self.label_to_name)} people")
