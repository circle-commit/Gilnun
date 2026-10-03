# 길눈 (Gilnun)

**시각장애인을 위한 카메라 기반 보행 보조 어시스턴트**

'길눈이 밝다'의 길눈처럼, 길을 잘 찾는 눈이 되어 주는 앱입니다. (이전 이름: KOJINGAPLA)

스마트폰 카메라 영상을 실시간으로 분석하여 두 가지 방식으로 음성 안내를 제공하는 프로토타입입니다.

- **문자 읽기(OCR) 모드** — 표지판, 라벨, 문서, 메뉴 등 화면 속 글자를 읽어 음성으로 안내
- **실시간 보행 안내(Live) 모드** — 전방의 장애물·차량·사람 등을 탐지하고 위치(왼쪽/정면/오른쪽)와 위험도를 분석해 음성으로 안내

**iOS 앱은 서버 없이 기기 안에서 모든 분석을 수행합니다.** YOLO11s(Core ML)로 물체 28종을 탐지하고 Apple Vision으로 한국어 문자를 인식하므로, 네트워크가 없어도 동작하고 카메라 영상이 기기 밖으로 나가지 않습니다. Android 앱은 아직 카메라 프레임을 FastAPI 백엔드로 전송하며, 백엔드는 PaddleOCR(문자 인식)과 YOLOv8n(객체 탐지)으로 음성 안내 문장을 생성합니다.

---

## 시스템 구성

```
iOS (온디바이스)
┌──────────────────────────────────────────────────────────┐
│ 카메라 ─► Live: YOLO11s (Core ML, 초당 약 5회) ─► 위치·위험도 분석 ─► 음성/햅틱 │
│        └► Text: Apple Vision 한국어 OCR ───────────────────► 음성/햅틱 │
└──────────────────────────────────────────────────────────┘

Android (서버 연동)
┌──────────────┐   카메라 프레임   ┌─────────────────────┐
│  Android 앱   │  (multipart 업로드) │   FastAPI 백엔드       │
│ - 카메라 캡처   │ ───────────────► │   POST /analyze      │
│ - 모드 전환    │                  │  mode=text → PaddleOCR │
│ - 음성/햅틱 출력 │ ◄─────────────── │  mode=live → YOLOv8n   │
└──────────────┘   음성 안내 JSON   │  위치·위험도 분석·문장 생성 │
                                   └─────────────────────┘
```

백엔드는 같은 `/analyze` 엔드포인트가 `mode` 파라미터(`text` / `live`)로 두 모드를 모두 처리합니다. iOS 앱의 위험도·거리·안내 문장 로직은 백엔드 Python 코드를 Swift로 옮긴 것이며, 테스트로 두 구현의 결과가 같은지 확인합니다.

---

## 프로젝트 구조

```text
Gilnun/
├── backend/                    # FastAPI 백엔드
│   ├── main.py                 # API 진입점 (/analyze, /health, /health/ocr)
│   ├── ocr_runtime.py          # OCR 실행 환경 설정
│   ├── core/                   # 탐지/OCR 엔진 코어
│   ├── services/               # 비즈니스 로직
│   │   ├── guide_service.py            # live 모드 진입점
│   │   ├── safety_service.py           # 객체 탐지 + 안내 조합
│   │   ├── vision_service.py           # YOLO 추론, 한글 라벨 매핑
│   │   ├── guidance_message_service.py # 위험도 점수·음성 문장 생성
│   │   └── text_service.py             # PaddleOCR 문자 인식
│   └── tests/                  # pytest 테스트
├── frontend/
│   ├── IOS_Swift/Gilnun/       # SwiftUI iOS 앱 (온디바이스 Core ML 모델 포함)
│   ├── IOS_Swift/Tests/        # iOS 로직·모델 검증 테스트 (run_tests.sh)
│   └── Android/                # Kotlin(CameraX) Android 앱
├── vision/                     # YOLO 학습/검증/추론 스크립트
│   ├── train.py                # 모델 학습
│   ├── validate.py             # 모델 검증
│   ├── predict.py              # 추론 CLI + 백엔드용 Detector
│   ├── export_coreml.py        # iOS용 Core ML 변환
│   └── tracker_logic.py        # 접근(approach) 추적 로직
├── scripts/
│   └── convert_cvat_to_yolo.py # CVAT 어노테이션 → YOLO 포맷 변환
├── datasets/
│   ├── 15.인도보행영상/바운딩박스/  # 원본 입력 이미지 + CVAT 어노테이션 (Bbox_0001/ ...)
│   └── yolo_sidewalk/          # 변환된 YOLO 데이터셋 (data.yaml, images/train·val·test, 28개 클래스)
├── runs/                       # 학습/검증/예측 결과물
├── docs/                       # 비전 파이프라인·데모 문서
├── test_images/                # 추론 테스트용 입력 샘플 이미지 (Bbox_*.jpg)
├── requirements.txt            # Python 의존성
└── yolov8n.pt                  # YOLOv8n 사전학습 가중치
```

---

## 백엔드 실행

```bash
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

`backend/venv` 대신 새 Python 환경(Ubuntu 22.04 / Python 3.10 권장)을 만든다면 의존성을 먼저 설치하세요.

```bash
# CPU 전용 PyTorch 먼저 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 나머지 의존성 설치
python -m pip install -r ../requirements.txt
```

주요 의존성: `ultralytics`(YOLOv8), `paddleocr` / `paddlepaddle`(OCR), `fastapi`, `uvicorn`, `opencv-python-headless`.

### API 엔드포인트

| 메서드 | 경로 | 설명 |
| --- | --- | --- |
| `GET` | `/health` | 서버 상태 확인 |
| `GET` | `/health/ocr` | OCR 모델 사용 가능 여부 확인 |
| `POST` | `/analyze` | 이미지 분석. form-data: `image`(파일), `mode`(`live` 또는 `text`) |

`/analyze` 응답 예시(live 모드)에는 `voice_guide`(음성 안내 문장), `warnings`, `risk_level`, `detections`(탐지된 객체 목록) 등이 포함됩니다.

---

## 프론트엔드

### iOS (`frontend/IOS_Swift`)
SwiftUI로 만든 `길눈` 앱. 서버 없이 기기에서 동작합니다. Xcode에서 `Gilnun.xcodeproj`를 열고 실제 기기에서 실행합니다(시뮬레이터에는 카메라가 없습니다).

- **실시간 보행 안내**: `ObjectDetector`가 번들된 `SidewalkDetector.mlpackage`(YOLO11s, 28종, 입력 640×384)를 초당 약 5회 실행하고, `SceneAnalyzer`가 접근 추적·위험도·거리 추정·안내 문장 생성을 수행합니다(`ApproachTracker`, `GuidanceEngine`).
- **문자 읽기**: `OCRFrameAnalyzer`가 Apple Vision으로 한국어·영어 문자를 인식하고, 화면이 안정되면 인식한 문장을 읽어 줍니다.
- 그 밖에 중복 음성 억제, 음성 출력(`SpeechManager` — 더 위급한 안내만 말을 끊고 끼어듦), 햅틱 피드백(`HapticFeedbackManager`)을 포함합니다.
- 무음 모드에서도 안내 음성이 나오고(다른 앱의 음악은 안내하는 동안만 작아지고, 팟캐스트는 잠시 멈춤), 앱을 쓰는 동안에는 화면이 자동으로 잠기지 않습니다.
- **화면**: 안내 카드 하나에 집중하고, 위험할수록 카드와 화면 테두리가 강조됩니다(색·아이콘·문구로 함께 구분). 시스템 글자 크기(Dynamic Type)를 따르고, iOS 26 이상에서는 리퀴드 글래스, 그 아래 버전에서는 반투명 재질을 씁니다. 상태별 화면은 `ContentView.swift`의 Xcode 미리보기에서 바로 볼 수 있습니다.
- **모드 전환**: 아래 버튼 외에도 화면 아무 곳이나 두 번 누르면(VoiceOver 사용 중에는 두 손가락으로 두 번 탭) 실시간 안내와 문자 읽기가 바뀝니다. 바뀔 때마다 진동과 음성으로 알려 주고, 처음 세 번은 전환 방법도 함께 안내합니다.
- **음성**: 기기에 내려받은 고품질 한국어 음성(향상된 품질·프리미엄)을 자동으로 쓰고, VoiceOver를 켠 사용자는 VoiceOver에서 정한 음성과 속도로 안내를 듣습니다. 고품질 음성은 설정 > 손쉬운 사용 > 읽기 및 말하기 > 음성 > 한국어에서 내려받습니다.

학습한 모델을 다시 변환하려면 macOS에서 다음을 실행합니다(`ultralytics`, `coremltools` 필요).

```bash
python -m vision.export_coreml
```

iOS 로직이 백엔드와 같은 결과를 내는지, Core ML 모델이 PyTorch 모델과 같은 객체를 찾는지 macOS에서 확인할 수 있습니다(Xcode 명령줄 도구와 `python3`만 필요).

```bash
frontend/IOS_Swift/Tests/run_tests.sh
```

이 테스트와 iOS 시뮬레이터 빌드는 PR마다 GitHub Actions(`.github/workflows/ios.yml`)에서도 자동으로 실행됩니다. `backend/services/guidance_message_service.py` 등 안내 로직을 바꿀 때는 Swift 코드도 함께 고치고 이 테스트를 실행하세요. 모델을 다시 학습했다면 `python frontend/IOS_Swift/Tests/make_detector_golden.py`로 기준 탐지 결과도 갱신합니다.

### Android (`frontend/Android`)
iOS 앱의 네이티브 Android 버전입니다. iOS 버전이 완성되면 이를 기준으로 다시 이식할 예정이라, 그때까지는 이전 앱 이름(Glass)을 그대로 씁니다. CameraX로 카메라 프레임을 스트리밍하고, ML Kit 한국어 텍스트 인식으로 OCR 대상의 안정성을 로컬에서 판단한 뒤 백엔드 `/analyze`를 호출합니다. Android Studio에서 `frontend/Android`를 엽니다.

> Android 앱의 `SERVER_URL`을 백엔드 주소에 맞게 수정해야 합니다. 카메라·진동 기능은 실제 기기에서만 동작합니다.

---

## 비전 파이프라인

인도 보행 환경에 맞춘 28개 클래스를 탐지합니다: `person, car, truck, bus, bicycle, motorcycle, scooter, wheelchair, stroller, traffic_light, traffic_sign, pole, bollard, bench, tree_trunk, movable_signage, potted_plant, parking_meter, stop(버스 정류장), table, barricade, chair, fire_hydrant, kiosk, carrier, dog, traffic_light_controller, power_controller`.

iOS 앱의 모델은 YOLO11s로, AI Hub 인도보행영상 전체(약 27만 장)로 학습했습니다. 처음 모델(YOLOv8n, 20종, 약 6만 7천 장)과 학습에 쓰지 않은 녹화(test 세트)에서 앱과 같은 조건(9:16 세로 화면, 화면의 1% 이상인 물체, 신뢰도 0.25)으로 비교한 결과입니다.

| 클래스 | YOLOv8n 재현율 | YOLO11s 재현율 | YOLO11s 정밀도 |
|---|---|---|---|
| 기존 20종 전체 | 0.852 | **0.908** | 0.742 (이전 0.702) |
| 전동 킥보드 | 0.000 | **0.800** | 0.727 |
| 휠체어 | 0.000 | **0.978** | 0.822 |
| 유모차 | 0.068 | **0.951** | 0.770 |
| 사람 | 0.890 | 0.941 | 0.898 |
| 새 8종 (바리케이드, 의자, 소화전 등) | — | 0.64~0.96 | 0.57~0.89 |

다시 학습하는 절차(데이터 받기, GPU 서버 학습, 앱 반영)는 [`docs/retraining.md`](docs/retraining.md)에 정리되어 있습니다. Android 앱이 쓰는 백엔드는 아직 처음 모델(YOLOv8n)을 씁니다.

```bash
# 학습 (기본값: YOLO11s, 40 에폭, 배치 128, 사용 가능한 GPU 자동 선택)
python scripts/build_balanced_train_list.py   # 드문 클래스 보강 → data_balanced.yaml
python -m vision.train --data datasets/yolo_sidewalk/data_balanced.yaml

# 검증 (Ultralytics mAP)
python -m vision.validate --weights runs/sidewalk/yolo11s_sidewalk/weights/best.pt

# 앱 기준 평가: 9:16 세로 화면, 화면의 1% 이상인 물체, 신뢰도 0.25에서 클래스별 정밀도·재현율
python -m vision.evaluate_app --weights runs/sidewalk/yolo11s_sidewalk/weights/best.pt

# 추론 (이미지 / 웹캠 + 접근 추적)
# 입력 이미지는 test_images/ 폴더의 샘플(Bbox_*.jpg)을 사용하거나 직접 지정합니다.
python -m vision.predict --weights .../best.pt --source test_images/Bbox_0099_MP_SEL_011528.jpg
python -m vision.predict --weights .../best.pt --source test_images/   # 폴더 전체
python -m vision.predict --weights .../best.pt --source 0 --track-approach
```

CPU 추론은 `yolov8n`, `device="cpu"`, `imgsz=416`, `conf=0.35`, `iou=0.5`, `max_det=20` 설정을 사용합니다. 자세한 내용은 [`docs/vision_pipeline.md`](docs/vision_pipeline.md)를 참고하세요.

### 안내 문장 생성 방식
백엔드는 단순 탐지에 그치지 않고 다음을 수행합니다.

- **위치 분석**: 바운딩 박스 중심을 기준으로 왼쪽/정면/오른쪽 구분
- **위험도 점수화**: 객체 종류, 위치, 화면 점유 면적, 수직 위치, 신뢰도, 접근 여부로 0~100 점수 산출 → `low/medium/high/critical` 등급화
- **거리 추정**: 객체 그룹별 임계값으로 `far/near/close/very_close` 단계 추정
- **접근 추적**: 프레임 간 박스 크기 증가율로 다가오는 물체 감지
- **음성 중복 억제**: 동일 상황의 반복 안내를 쿨다운으로 억제하고, 가중치 기반 템플릿으로 자연스러운 한국어 안내 문장 생성

---

## 데이터셋 준비

CVAT로 어노테이션한 데이터를 YOLO 포맷으로 변환할 수 있습니다.

```bash
python scripts/convert_cvat_to_yolo.py
```

- **입력 이미지 위치**: `datasets/15.인도보행영상/바운딩박스/` — `Bbox_0001/`, `Bbox_0002/` … 폴더 아래에 원본 이미지와 CVAT 어노테이션이 함께 들어 있습니다. `scripts/download_aihub.py`로 AI Hub에서 받습니다.
- **출력 위치**: `datasets/yolo_sidewalk/images/{train,val,test}` 및 `datasets/yolo_sidewalk/labels/{train,val,test}` — 줄인 JPEG 이미지와 YOLO 라벨(28종)이 저장됩니다. test는 학습에서 통째로 뺀 녹화 묶음입니다.

데이터 받기부터 GPU 서버 학습, 앱 반영까지의 전체 순서는 [docs/retraining.md](docs/retraining.md)에 있습니다.

데이터셋 설정은 [`datasets/yolo_sidewalk/data.yaml`](datasets/yolo_sidewalk/data.yaml)에 정의되어 있습니다.

> `datasets/`는 용량 문제로 `.gitignore`에 등록되어 있어, 저장소에는 구조 확인용 **샘플 파일만** 포함되어 있습니다 — CVAT 입력 예시(`Bbox_0001/MP_SEL_000001.jpg` + `0617_01.xml`), 변환된 YOLO 이미지·라벨 쌍 일부, `data.yaml`. 전체 데이터셋은 AI Hub **인도보행영상**에서 받아 위 변환 스크립트로 생성합니다.

### 학습 데이터 예시

데이터셋은 AI Hub **인도보행영상** 공개 데이터셋을 YOLO 포맷으로 변환한 것입니다. 지금 모델은 녹화 2,480개 전체로 만든 **train 269,324장 / val 67,390장 / test 13,027장**, 28개 클래스로 학습했습니다. 처음 모델은 그중 `Bbox_0001`~`Bbox_0410`(train 66,538장 / val 16,715장), 20개 클래스로 학습했습니다. 분할은 고정 시드로 수행되며 `datasets/yolo_sidewalk/split_manifest.json`에 기록됩니다. 아래 예시 이미지는 처음 학습 때의 것입니다.

#### 1. 학습 배치 시각화 (Ground Truth 박스 포함)

모델이 실제로 학습하는 이미지에 정답 바운딩 박스를 그린 모자이크입니다. 모자이크 증강(mosaic augmentation)이 적용된 상태 그대로입니다.

| | | |
| --- | --- | --- |
| ![train_batch0](docs/images/dataset/train_batch0.jpg) | ![train_batch1](docs/images/dataset/train_batch1.jpg) | ![train_batch2](docs/images/dataset/train_batch2.jpg) |

학습 후반(마지막 에폭 구간, 모자이크 증강 비활성화 이후)의 배치입니다.

| | | |
| --- | --- | --- |
| ![train_batch291130](docs/images/dataset/train_batch291130.jpg) | ![train_batch291131](docs/images/dataset/train_batch291131.jpg) | ![train_batch291132](docs/images/dataset/train_batch291132.jpg) |

#### 2. 데이터셋 통계

전체 학습 데이터의 클래스 분포와 바운딩 박스 크기·위치 분포입니다.

![labels](docs/images/dataset/labels.jpg)

#### 3. 원본 이미지 + YOLO 라벨 쌍

학습 데이터는 이미지 1장당 라벨 텍스트 파일 1개로 구성됩니다. 라벨 형식은 `class_id x_center y_center width height`(0~1 정규화 좌표)입니다.

![raw_sample](docs/images/dataset/Bbox_0001_MP_SEL_000001.jpg)

`datasets/yolo_sidewalk/labels/train/Bbox_0001_MP_SEL_000001.txt`:

```text
14 0.678070 0.519764 0.030734 0.202676   # tree_trunk
15 0.580904 0.392657 0.033911 0.124796   # movable_signage
14 0.616492 0.390560 0.027745 0.266491   # tree_trunk
14 0.579716 0.382088 0.019630 0.150176   # tree_trunk
1 0.885651 0.369583 0.028698 0.031204    # car
10 0.973464 0.336532 0.017865 0.031213   # traffic_sign
1 0.959010 0.426852 0.075729 0.070370    # car
1 0.925755 0.628333 0.148490 0.331296    # car
1 0.913073 0.411991 0.063437 0.061204    # car
15 0.649911 0.391602 0.043427 0.222093   # movable_signage
```

다양한 촬영 구간(`Bbox_0100`, `Bbox_0200`, `Bbox_0300`)의 학습 이미지 예시입니다. 각 이미지에는 같은 이름의 YOLO 라벨 파일이 `datasets/yolo_sidewalk/labels/train/`에 함께 존재합니다.

| Bbox_0100 | Bbox_0200 | Bbox_0300 |
| --- | --- | --- |
| ![sample_0100](docs/images/dataset/Bbox_0100_MP_SEL_011651.jpg) | ![sample_0200](docs/images/dataset/Bbox_0200_MP_SEL_034653.jpg) | ![sample_0300](docs/images/dataset/Bbox_0300_MP_SEL_045651.jpg) |

#### 4. 정답(Ground Truth) vs 모델 예측 비교

검증 세트에서 정답 라벨과 학습된 모델의 예측 결과를 나란히 비교한 것입니다.

| Ground Truth | Prediction |
| --- | --- |
| ![val_labels](docs/images/dataset/val_batch0_labels.jpg) | ![val_pred](docs/images/dataset/val_batch0_pred.jpg) |

---

## 문서

- [`docs/vision_pipeline.md`](docs/vision_pipeline.md) — YOLOv8n 비전 파이프라인 상세
- [`docs/retraining.md`](docs/retraining.md) — GPU 서버에서 탐지 모델을 다시 학습하고 앱에 넣는 방법
- [`docs/demo_presentation_summary.md`](docs/demo_presentation_summary.md) — 데모 발표 자료 요약(시스템 아키텍처, 모드별 UX, 학습 결과)

---

## 향후 개선 방향

- 실제 아이폰 촬영 영상으로 탐지 성능 검증
- 세션별 객체 추적으로 접근 경고 정교화
- LiDAR(아이폰 Pro) 기반 실제 거리 안내
- 문자 감지·위험 경고·방향 안내용 햅틱 피드백 확장
- Android 앱 온디바이스 전환(TFLite) 및 다국어 OCR 개선
