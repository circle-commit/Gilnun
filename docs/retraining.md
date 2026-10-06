# 탐지 모델 재학습

데이터 준비와 평가는 Mac에서 하고, 시간 단위로 비용이 드는 GPU 서버에서는 학습만 합니다. 새 모델은 YOLO11s이고 28종을 탐지합니다. YOLO11은 지금 모델(YOLOv8n)과 출력 형식이 같아서, 앱 코드는 고치지 않고 모델 파일만 바꾸면 됩니다.

| 단계 | 어디서 | 걸리는 시간 |
|---|---|---|
| 1. 데이터 받기 | Mac | 약 6시간 (40MB/s 기준) |
| 2. 학습용으로 변환 | Mac | 약 20분 |
| 3. 기준선 측정 | Mac | 약 10분 |
| 4. 학습 | GPU 서버 | 약 8시간 (A100 1장) |
| 5. 비교와 앱 반영 | Mac | 약 30분 |
| 6. 다른 카메라 영상으로 확인 | Mac | 다운로드(약 40GB) 후 약 30분 |

Mac에서 쓰는 Python 환경은 한 번만 만들면 됩니다.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install ultralytics coremltools
```

## 1. 데이터 받기 (Mac)

AI Hub에서 인도보행영상 데이터 사용 신청이 승인되어 있어야 하고, AI Hub API 키가 필요합니다. 승인에는 유효기간이 있어서, 기간이 지나면 다운로드할 때 "데이터 승인 유효기간이 도래하였습니다"라는 메시지가 나옵니다. 그때는 AI Hub에서 다시 신청합니다.

바운딩박스 파일 30개(약 300GB)를 내려받아 `datasets/15.인도보행영상/바운딩박스/`에 풉니다. 실행하면 API 키를 묻습니다. 입력한 키는 화면에 보이지 않고 어디에도 저장되지 않습니다. 중간에 끊겨도 같은 명령을 다시 실행하면 이어서 받습니다.

```bash
python3 scripts/download_aihub.py --list   # 받을 파일과 크기만 확인
python3 scripts/download_aihub.py
python3 scripts/download_aihub.py --status # 진행 상황 확인 (받는 중에 다른 터미널에서)
```

> AI Hub의 aihubshell은 macOS에서 큰 파일을 잘못 합칠 수 있어서(조각 10번이 2번보다 앞에 정렬됨) 이 스크립트를 씁니다. 압축 파일은 푼 뒤 바로 지우므로, 풀린 데이터 약 300GB에 작업 공간 35GB 정도만 더 있으면 됩니다.

## 2. 학습용으로 변환 (Mac)

```bash
# datasets/yolo_sidewalk 생성: 28종 라벨, 줄인 JPEG 이미지
python scripts/convert_cvat_to_yolo.py

# 드문 클래스가 담긴 이미지를 반복해서 넣은 학습 목록 + 에폭마다 확인할 검증 이미지 8,000장
# → datasets/yolo_sidewalk/data_balanced.yaml
python scripts/build_balanced_train_list.py

# 서버에 올릴 파일 하나로 묶기 → datasets/sidewalk_train.tar
python scripts/pack_training_data.py
```

- **분할**
  - **test:** `Bbox_0410` 이후 폴더 중 번호가 20의 배수인 녹화 묶음을 통째로 학습에서 뺍니다. 같은 녹화의 프레임은 서로 비슷해서, 모델이 한 번도 보지 못한 녹화로 재야 실제 성능에 가깝습니다. 예전 모델은 `Bbox_0001`~`Bbox_0410`(83,253장)으로 학습했기 때문에, 두 모델 모두에게 처음 보는 장면입니다.
  - **train/val:** 나머지 이미지를 예전과 같은 방식으로 나눕니다. 그래서 예전 모델의 학습 이미지는 train에, 검증 이미지는 val에 그대로 들어갑니다.
- **이미지 크기:** train과 val은 긴 변 640(학습 크기), test는 긴 변 1280으로 줄입니다. 평가할 때 test 이미지에서 앱 화면 비율(9:16)만큼 잘라 쓰기 때문입니다.
- **클래스:** 0~19번은 예전 모델과 같고, 20~27번(바리케이드, 의자, 소화전, 가판대, 카트, 강아지, 신호 제어함, 전기 분전함)을 새로 넣었습니다.

## 3. 지금 모델 성능 측정 (Mac)

지금 앱에 들어 있는 모델을 test 세트에서 앱과 같은 조건으로 측정합니다. 조건은 9:16 세로 화면, 화면의 1% 이상인 물체, 신뢰도 기준값 0.25입니다. 모델이 모르는 클래스와, 세로 화면에 절반 넘게 잘린 물체는 점수에서 빠집니다. 오류의 종류(놓침, 배경 오탐, 다른 클래스로 착각)와 사진은 `python -m vision.analyze_errors`로 볼 수 있습니다.

```bash
python -m vision.evaluate_app --out baseline.json
```

## 4. GPU 서버에서 학습 (엘리스 클라우드)

서버는 켜져 있는 동안 계속 과금됩니다(A100 80GB 1장인 G-NAHP-80 기준 시간당 2,500원). 그래서 서버를 만들기 전에 1~3단계를 모두 끝내 둡니다.

학습 속도는 GPU만큼 데이터를 준비하는 CPU 수에도 좌우됩니다. 엘리스의 A100 상품은 GPU 1장마다 CPU 16개가 붙어 있고, 같은 돈으로 쓸 수 있는 CPU가 H100 상품이나 MIG(쪼갠 A100) 상품보다 많습니다. 그래서 A100 1장이 가장 경제적입니다. A100 2장(G-NAHP-160)을 쓰면 총비용은 비슷하고 시간은 절반 가까이로 줄어듭니다. 이때는 학습 명령에 `--device 0,1`을 붙입니다.

### 서버 만들기
1. **비밀키:** 런박스 → 비밀키 관리에서 비밀키를 발급합니다. PEM 파일은 **한 번만** 받을 수 있습니다. 받은 파일은 아래처럼 옮기고 권한을 바꿉니다.
   ```bash
   mv ~/Downloads/<받은 파일>.pem ~/.ssh/elice.pem && chmod 600 ~/.ssh/elice.pem
   ```
2. **인스턴스:** 런박스 → 인스턴스에서 G-NAHP-80(A100 80GB 1장), 실행 환경 VSCode (CUDA 12.8), 스토리지 128GiB로 만듭니다.
3. **접속 정보:** 인스턴스의 **연결 → 다른 SSH 클라이언트 사용**에 나오는 명령에서 호스트(예: `central-02.tcp.tunnel.elice.io`)와 포트를 확인합니다. 아래 명령의 `<호스트>`와 `<포트>`에 넣습니다.

### 데이터 올리기 (Mac)
```bash
rsync -P -e "ssh -i ~/.ssh/elice.pem -p <포트>" datasets/sidewalk_train.tar elicer@<호스트>:~/
```

엘리스 접속 경로(터널)를 거치면 연결 하나에 초당 6~13MB 정도라 21.5GB를 올리는 데 30분~1시간이 걸립니다. 끊기면 같은 명령을 다시 실행해 이어서 올립니다. 그동안에도 서버 요금이 나가므로, 서버 준비(아래 설치 명령)를 업로드와 동시에 해 두면 좋습니다.

### 학습 (서버)
```bash
ssh -i ~/.ssh/elice.pem -p <포트> elicer@<호스트>
git clone https://github.com/circle-commit/Gilnun.git && cd Gilnun
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics==8.4.171
pip uninstall -y opencv-python && pip install opencv-python-headless
tar -xf ~/sidewalk_train.tar -C datasets/ && rm ~/sidewalk_train.tar
setsid nohup python -m vision.train --data datasets/yolo_sidewalk/data_balanced.yaml > train.log 2>&1 < /dev/null &
tail -f train.log
```

- **PyTorch 버전:** 엘리스 서버의 GPU 드라이버(535)는 CUDA 12.2까지 지원합니다. 그래서 그냥 `pip install`로 받는 최신 PyTorch 대신 CUDA 12.6용 빌드를 먼저 설치합니다. 설치 뒤 `python -c "import torch; print(torch.cuda.is_available())"`가 `True`인지 확인합니다.
- **OpenCV:** 서버에 화면용 라이브러리(libGL)가 없어서 일반 OpenCV는 불러올 때 오류가 납니다. 화면 기능을 뺀 headless 판으로 바꿉니다.

- **기본값:** YOLO11s, 40 에폭, 배치 128, 이미지 크기 640입니다. A100 1장에서 한 에폭에 약 12분, 전체 약 8시간(약 2만 원)이 걸립니다. 배치를 키울수록 단계마다 생기는 고정 처리 시간이 줄어서, 초당 학습 이미지가 배치 32·64·128에서 312·397·499장이었습니다. 배치 128은 GPU 메모리를 약 31GB 쓰므로, 메모리가 작은 서버에서 부족하다는 오류가 나면 `--batch 64`나 `--batch 32`로 줄입니다.
- **조기 종료:** 15 에폭 동안 나아지지 않으면 학습이 자동으로 멈춥니다(`--patience`).
- **비용 확인:** 학습이 시작되고 몇 분 뒤 진행 표시줄에 나오는 속도(it/s)로 전체 시간을 계산할 수 있습니다. 예상보다 비싸면 Ctrl+C로 멈추고 `--epochs`를 줄여 다시 시작합니다.
- **접속이 끊겼다면:** 다시 접속해서 `tail -f ~/Gilnun/train.log`로 이어서 봅니다. `setsid nohup`으로 실행했기 때문에 학습은 계속됩니다. 브라우저의 VS Code 터미널에서도 같은 명령으로 볼 수 있습니다.
- **학습이 멈췄다면:** 마지막으로 끝난 에폭부터 처음과 같은 설정으로 이어서 학습합니다. 크레딧이 떨어지면 인스턴스가 중지되는데, 인스턴스를 삭제하지 않았다면 저장공간이 남아 있어서 다시 켠 뒤 이어 갈 수 있습니다.
  ```bash
  setsid nohup python -m vision.train --resume >> train.log 2>&1 < /dev/null &
  ```
- **크레딧:** 학습 전체(약 8~9시간)에 2만 원 남짓 듭니다. 시작하기 전에 그만큼 충전하거나 결제 수단을 등록해 둡니다.

### 결과 가져오기와 서버 삭제
학습이 끝나면 `train.log` 마지막 줄에 결과 파일 위치가 나옵니다. 보통 `runs/sidewalk/yolo11s_sidewalk/weights/best.pt`입니다. Mac에서 결과 폴더를 통째로 가져옵니다.

```bash
mkdir -p runs/sidewalk
rsync -a -e "ssh -i ~/.ssh/elice.pem -p <포트>" elicer@<호스트>:Gilnun/runs/sidewalk/yolo11s_sidewalk runs/sidewalk/
```

가져온 뒤에는 엘리스에서 인스턴스를 **중지한 다음 삭제**합니다. 중지만 하면 저장공간 요금이 계속 나갑니다.

## 5. 새 모델과 비교하고 앱에 넣기 (Mac)

```bash
python -m vision.evaluate_app --weights runs/sidewalk/yolo11s_sidewalk/weights/best.pt --out yolo11s.json
```

`baseline.json`과 `yolo11s.json`의 클래스별 재현율·정밀도를 비교합니다. 특히 전동 킥보드, 휠체어, 유모차, 차량을 봅니다. test 세트는 학습에 쓰지 않은 녹화 묶음이지만, 같은 데이터셋의 같은 카메라로 찍은 영상입니다. 실제 성능은 아이폰으로 새로 찍은 영상으로도 확인하세요.

새 모델이 더 나으면 앱에 넣습니다. CoreML로 변환하고, 앱 탐지 코드와 PyTorch 결과를 비교하는 정답 데이터를 새 모델로 다시 만든 뒤 테스트합니다.

```bash
python -m vision.export_coreml --weights runs/sidewalk/yolo11s_sidewalk/weights/best.pt
python frontend/IOS_Swift/Tests/make_detector_golden.py --weights runs/sidewalk/yolo11s_sidewalk/weights/best.pt
frontend/IOS_Swift/Tests/run_tests.sh
```

Xcode에서 앱을 빌드하면 `SidewalkDetector.mlpackage`가 새 모델로 바뀌어 있습니다.

> `best.pt`는 수십 MB입니다. 앱에 필요한 것은 CoreML 모델뿐이므로, `.pt` 파일은 저장소에 넣지 말고 GitHub Releases 등에 따로 보관하는 것을 권장합니다.

## 6. 다른 카메라 영상으로 확인 (Mac)

test 세트는 학습 데이터와 같은 카메라로 찍은 영상입니다. 처음 보는 카메라와 동네에서도 잘 되는지는 AI Hub의 **1인칭 시점 보행영상** 데이터로 확인합니다. 이 데이터도 사용 신청이 승인되어 있어야 합니다. 라벨과, 165cm 높이에서 찍은 실외 영상의 검증 이미지(약 40GB)만 받습니다.

```bash
python3 scripts/download_aihub.py --dataset-key 159 --folder '' --path BBOX --path 라벨링 --keep-tree --out datasets/aihub
python3 scripts/download_aihub.py --dataset-key 159 --folder '' --path 2.Validation --path BBOX --path 원천 --path Average_stature/out --keep-tree --out datasets/aihub
python scripts/convert_first_person.py --split Validation --out-split test
python -m vision.evaluate_app --data datasets/first_person/data.yaml --out first_person.json
```

라벨 체계가 달라서 다음처럼 맞춰서 잽니다.

- **이름이 다른 종류:** 우리 클래스로 바꿉니다(kickboard → 전동 킥보드, powerpole → 기둥 등). 승용차, 트럭, 버스를 모두 car로 라벨링한 데이터라서 세 클래스는 하나로 합쳐 셉니다.
- **우리 클래스가 없는 종류:** 라바콘, 길에 놓인 쓰레기 같은 물체를 탐지한 것은 오탐으로 세지 않습니다. 신호등, 표지판처럼 이 데이터가 라벨링하지 않는 클래스는 점수에서 뺍니다.
- **정밀도:** 한 장면의 물체를 전부 라벨링하지는 않아서, 정밀도가 실제보다 낮게 나옵니다. 이 데이터로는 재현율을 봅니다.

## 지난 학습 기록 (2026년 10월)

- **서버:** 엘리스 G-NAHP-80(A100 1장), 40 에폭, 배치 128. 학습에 약 8시간 반이 걸렸고, 업로드와 설정을 합쳐 약 2만 5천 원이 들었습니다. 중간에 크레딧이 떨어져 한 번 멈췄다가 `--resume`으로 이어서 학습했습니다.
- **결과 (test 세트, 앱 조건):** 기존 20종의 재현율이 0.852에서 0.908로, 정밀도가 0.764에서 0.817로 올랐습니다. 처음에는 정밀도를 0.702→0.742로 기록했는데, 세로 화면에 잘린 물체를 오탐으로 센 평가 오류였습니다. 전동 킥보드는 0.00에서 0.80, 휠체어는 0.00에서 0.98, 유모차는 0.07에서 0.95가 됐습니다. 새로 넣은 8종의 재현율은 0.64(바리케이드)~0.96(강아지)입니다.
- **약한 클래스:** 주차 정산기(학습 이미지 206장)는 거의 찾지 못하고, 카트와 두 제어함은 정밀도가 0.6 안팎입니다. 데이터를 더 모으거나 비슷한 클래스를 합치는 것을 다음 개선 후보로 둡니다.
- **다른 카메라 (1인칭 시점 보행영상 검증 이미지 14,237장):** 재현율이 0.920으로, 같은 카메라로 찍은 test 세트(0.903)만큼 나왔습니다. 정밀도는 0.557이었지만, 오탐을 클래스별로 무작위로 뽑아 본 144개 중 약 90%가 라벨이 빠진 실제 기둥, 차, 사람, 나무, 입간판, 카트였습니다. 그래서 이 데이터로 기존 클래스를 더 학습할 필요는 적습니다. 대신 지금 모델에 없는 장애물이 많이 담겨 있습니다. 실외 학습 영상 기준으로 라바콘이 약 7,300장, 길에 놓인 쓰레기가 약 5,700장, 바퀴 달린 쓰레기 수거함이 약 1,700장에 있어서, 새 클래스를 넣을 때 쓸 수 있습니다.

