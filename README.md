# Edge-Optimized Face Detection and Recognition System (Raspberry Pi)

Real-time face detection and recognition system deployed on Raspberry Pi using OpenCV and the Pi Camera module.

**Year:** 2022

## Features
- Real-time face detection and recognition on edge device (Raspberry Pi)
- Optimized for low latency and limited memory
- Complete pipeline: capture → detection → feature extraction → matching
- Simple enrollment of known faces

## Requirements
- Raspberry Pi 3/4
- Raspberry Pi Camera Module
- Python 3.7+

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Enroll a known face
```bash
python enroll.py --name "PersonName"
```
(Look at the camera and press `s` to save)

### 2. Start the system
```bash
python main.py
```
Press `q` to quit.

## Project Structure
```
├── README.md
├── requirements.txt
├── main.py          # Run the real-time system
├── enroll.py        # Add new faces
└── face_system.py   # Core detection + recognition code
```

## Notes
- Uses OpenCV Haar Cascade for detection (fast on Pi)
- Uses LBPH recognizer (lightweight, suitable for edge devices)
- Designed for continuous monitoring under resource constraints

## Author
Salman Tauheed  
GitHub: [salmantauheed2002](https://github.com/salmantauheed2002)
