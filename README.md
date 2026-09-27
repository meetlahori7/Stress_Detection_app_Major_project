# AI-Based Multimodal Human Stress Identification — Streamlit App

## Modes

1. Upload Video
2. Live Camera + Microphone

## Final inference pipeline

Face:
YuNet → 15% face margin → ResNet18 → 512-D embedding → StandardScaler → MLP

Voice:
mono → 16 kHz → MFCC + delta + delta2 → mean/std pooling → StandardScaler → Logistic Regression

Fusion:
0.40 × Face probability + 0.60 × Voice probability

Threshold:
0.50

## Folder

```text
Stress_Detection_App/
├── app.py
├── requirements.txt
├── README.md
└── models/
    ├── stress_detection_model.pkl
    ├── resnet18_face_encoder.pth
    └── face_detection_yunet_2026may.onnx
```

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Live mode uses WebRTC. The browser asks for camera and microphone permission.

## Important

Live mode captures a short camera/microphone window and runs the trained model after the stream is stopped. It is not continuous frame-by-frame clinical monitoring.

The project's binary targets are emotion-derived proxy labels, not clinically or physiologically validated stress labels.
