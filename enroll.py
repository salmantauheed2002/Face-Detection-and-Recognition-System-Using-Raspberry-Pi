"""
Enroll a new face.
Usage: python enroll.py --name "PersonName"
Press 's' to save the face, 'q' to quit.
"""

import argparse
import cv2
from face_system import FaceSystem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Person name")
    args = parser.parse_args()

    system = FaceSystem()
    cap = cv2.VideoCapture(0)

    print(f"Look at the camera. Press 's' to enroll '{args.name}', 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = system.detect_faces(frame)
        display = frame.copy()

        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Enroll Face", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") and len(faces) > 0:
            # Use largest face
            faces = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]
            system.enroll(face, args.name)
            print(f"Successfully enrolled: {args.name}")
            break
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
