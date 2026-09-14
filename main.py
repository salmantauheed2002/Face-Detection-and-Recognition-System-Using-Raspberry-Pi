"""
Run real-time face detection and recognition on Raspberry Pi.
Press 'q' to quit.
"""

import cv2
from face_system import FaceSystem


def main():
    system = FaceSystem()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting system... Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = system.detect_faces(frame)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            name, conf = system.predict(face)

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            text = f"{name} ({conf:.0f})"
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
