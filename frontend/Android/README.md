# 길눈 Android

Native Android version of the iOS Gilnun app. Like iOS, everything runs on the phone, with no server and no network permission, so camera frames never leave the device.

- **Live guidance**: the sidewalk YOLO11m model (`app/src/main/assets/sidewalk_detector.onnx`, the same checkpoint and 640x384 input as the iOS Core ML model) runs in ONNX Runtime, at most five times a second. `SceneAnalyzer` then tracks approaching objects, scores risk, estimates distance and picks Korean sentences (`ApproachTracker`, `GuidanceEngine`). Android phones have no LiDAR, so distance comes from each box's size and position, like the backend.
- **Text reading**: ML Kit's on-device Korean recognizer finds text; the app tells the user how to aim the phone ("휴대폰을 왼쪽으로 조금 옮겨 주세요."), waits for steady text, then reads a crop around it again so small print gets more resolution. Lines with digits are spoken only when two frames read them the same, the torch turns on in the dark, and a tap repeats the last text.
- **Speech and vibration**: more urgent guidance cuts off calmer sentences, other apps' audio is lowered while speaking, and vibration patterns show the text search state.
- **Mode switch**: the buttons at the bottom, or a double tap anywhere on the camera view.

`GuidanceEngine.kt`, `SceneAnalyzer.kt`, `ApproachTracker.kt` and `TextReading.kt` are ports of the iOS Swift files of the same names (and of `backend/services/guidance_message_service.py`). Keep them in sync.

## Run

1. Open `frontend/Android` in Android Studio and let it sync Gradle.
2. Run the `app` configuration on a physical Android 8.0+ device. Camera and vibration need real hardware.

## Tests

The guidance, detector and text-reading code is plain Kotlin, so it runs on the JVM without the Android SDK (JDK 17+, Gradle and `python3`):

```bash
frontend/Android/tests/run_tests.sh
```

It checks the Kotlin guidance logic against the Python backend (the same fixtures as the iOS tests), the ONNX model against the PyTorch reference detections, and the text-reading rules.

## Updating the model

After retraining, export the new checkpoint and refresh the reference detections:

```bash
python frontend/Android/tools/export_onnx.py      # requires ultralytics, onnx, onnxslim
python frontend/IOS_Swift/Tests/make_detector_golden.py
```
