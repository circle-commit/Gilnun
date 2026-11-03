# 길눈 Android

iOS 길눈 앱의 네이티브 Android 버전입니다. iOS처럼 모든 분석을 폰 안에서 하고 서버도 인터넷 권한도 쓰지 않으므로, 카메라 영상이 기기 밖으로 나가지 않습니다.

- **실시간 보행 안내**: iOS Core ML 모델과 같은 체크포인트·같은 입력(640×384)의 인도 YOLO11m 모델(`app/src/main/assets/sidewalk_detector.onnx`)을 ONNX Runtime으로 최대 초당 5회 실행합니다. 이어서 `SceneAnalyzer`가 다가오는 물체 추적, 위험도 점수, 거리 추정, 한국어 안내 문장 선택을 합니다(`ApproachTracker`, `GuidanceEngine`). Android 폰에는 LiDAR가 없어서, 거리는 백엔드처럼 상자의 크기와 위치로 추정합니다.
- **문자 읽기**: ML Kit의 기기 내 한국어 인식으로 글자를 찾고, 휴대폰을 어디로 옮길지 알려 줍니다("휴대폰을 왼쪽으로 조금 옮겨 주세요."). 글자가 안정되면 그 부분을 잘라 다시 읽어서 작은 글씨도 더 잘 읽습니다. 숫자가 있는 줄은 두 프레임에서 똑같이 읽혔을 때만 읽고, 어두우면 플래시를 켜며, 화면을 한 번 누르면 마지막 글자를 다시 읽습니다.
- **보행 신호 인식 (실험 기능)**: 한국 보행 신호등(세로로 긴 상자, 위에 빨간 사람, 아래에 초록 사람)이 보이면 켜진 등을 읽고, 바뀔 때 신호등 쪽에서 들리게 알려 줍니다(`PedestrianSignal.kt`). 여러 프레임이 일치해야 말하고, 초록불이어도 "건너세요"만 말하지 않으며("차 소리를 확인하고 건너세요"), 중간부터 본 초록불이나 깜빡이는 초록불은 "다음 신호를 기다리세요"로 안내합니다. 실제 횡단보도에서 충분히 검증하기 전에는 이 안내에만 의존하면 안 됩니다.
- **물체 쪽에서 들리는 안내**: 이어폰을 끼면 실시간 안내 문장마다 그 물체가 있는 쪽에서 소리가 납니다(`SpatialAudio`). 목표 방향에서 들리는 소리가 말로 하는 방향 안내보다 길 안내에 효과적이라는 시각장애인 길 안내 연구를 따랐습니다(Loomis, Golledge & Klatzky 1998; Microsoft Soundscape).
- **음성과 진동**: 더 위급한 안내만 말하던 문장을 끊고 끼어들고, 말하는 동안 다른 앱 소리를 줄이며, 문자를 찾는 상태를 진동 패턴으로 알려 줍니다.
- **탐지 모델 설정**: 제목 옆 "설정" 버튼으로 "정확도 우선"(기본값, float 모델)과 "속도 우선"(`sidewalk_detector_int8.onnx`, 같은 모델을 int8로 압축: 약 2배 빠르고 크기는 절반이지만 멀리 있는 흐릿한 물체를 놓칠 수 있음, `tools/quantize_onnx.py` 참고) 중 고릅니다.
- **걸을 때 편의**: 안내하는 동안 화면이 꺼지지 않습니다. 폰을 주머니나 가방에 넣으면(근접 센서가 1.5초 이상 가려짐, `PocketDetector`) 분석을 멈추고 화면이 꺼질 수 있게 하며, 꺼내면 다시 시작합니다. 시끄러운 길에서도 알 수 있게 음성과 함께 진동합니다: 보행 신호가 초록불로 바뀌면 길게 한 번, 기다려야 하면 짧게 세 번, 바로 앞이 위험하면 강하게 두 번.
- **카메라 방향·가림 알림**: 폰이 바닥이나 하늘을 비추거나 옆으로 누워 있으면(중력 센서) "조금 들어서 앞을 비춰 주세요"처럼, 렌즈가 가려지거나 너무 어두우면 "카메라가 가려졌어요"처럼 알려 줍니다(`CameraViewCheck.kt`). 1.5초 이상 계속될 때만 말하고, 고쳐지면 "이제 앞이 잘 보여요"라고 한 번 알려 줍니다.
- **TalkBack**: 버튼을 "버튼"으로 알리고 두 번 탭하면 무엇을 하는지 말합니다("다시 읽기", "문자 읽기 모드로 바꾸기"). 장식용 아이콘은 숨기고, 앱 제목은 제목(heading)으로 표시하며, 모든 버튼은 높이 48dp 이상입니다.
- **모드 전환**: 아래쪽 버튼을 누르거나, 카메라 화면 아무 곳이나 두 번 누릅니다.

`GuidanceEngine.kt`, `SceneAnalyzer.kt`, `ApproachTracker.kt`는 같은 이름의 iOS Swift 파일(과 `backend/services/guidance_message_service.py`)을, `TextReading.kt`는 iOS의 `OCRFrameAnalyzer.swift`와 `TextStabilityTracker.swift`를 옮긴 것입니다. 한쪽을 고치면 다른 쪽도 함께 맞춰 주세요.

## 실행

1. Android Studio에서 `frontend/Android`를 열고 Gradle 동기화를 기다립니다.
2. Android 8.0 이상 실제 기기에서 `app` 구성을 실행합니다. 카메라와 진동은 실제 기기에서만 동작합니다.

## 테스트

안내·탐지·문자 읽기 코드는 순수 Kotlin이라 Android SDK 없이 JVM에서 실행됩니다(JDK 17 이상, Gradle, `python3` 필요).

```bash
frontend/Android/tests/run_tests.sh
```

Kotlin 안내 로직을 Python 백엔드와 비교하고(iOS 테스트와 같은 기준 데이터), ONNX 모델을 PyTorch 기준 탐지 결과와 비교하며, 문자 읽기·보행 신호·주머니 감지 규칙을 확인합니다.

## 모델 업데이트

모델을 다시 학습했다면 새 체크포인트를 변환하고 기준 탐지 결과를 갱신합니다.

```bash
python frontend/Android/tools/export_onnx.py      # ultralytics, onnx, onnxslim 필요
python frontend/Android/tools/quantize_onnx.py    # int8 "속도 우선" 모델, onnxruntime도 필요
python frontend/IOS_Swift/Tests/make_detector_golden.py
```
