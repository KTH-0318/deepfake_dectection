# 딥페이크 AI 탐지 모델

KoDF·Celeb-DF·FaceForensics++ 세 공개 데이터셋을 **영상(Video ID) 단위로 분할**해 학습·평가 데이터 누수를 막고,
EfficientNet-B0 전이학습과 3종 모델 앙상블(B4·B3·ResNet-50)로 Real/Fake를 분류한 프로젝트.
Grad-CAM으로 판별 근거를 시각화하고, 프레임 단위 학습의 한계와 개선 방향(3D CNN·CNN-LSTM)까지 분석했습니다.

**단일 EfficientNet-B0 Test AUC 0.7027 → 앙상블 0.7045** · 인공지능 수업 팀 프로젝트 (5인, 조장)

**프로젝트 상세 → [Notion 포트폴리오](https://zany-meeting-aba.notion.site/30583f9aaedb83509e0e81ed3e56fa49)**

**발표 자료 → [Google Drive](https://drive.google.com/drive/folders/1Yw1ZkKn_484lvV0iVR_mV13tE-LBxdNH)**

## 핵심 포인트
- **누수 없는 분할**: 같은 영상의 프레임이 Train/Val/Test에 섞이지 않도록 Video ID 기준 분할, EDA에서 누수·Split 공정성 검증
- **샘플링**: 영상당 최대 50프레임, Real/Fake·데이터셋 비율 유지 → Train 21,325 / Val 6,128 / Test 3,074
- **학습 전략**: Label Smoothing 0.1 · Dropout 0.6 · AdamW(5e-4) · Early Stopping(Val AUC) · MixUp · Backbone Freeze · AMP · Cosine + ReduceLROnPlateau

## 내 역할 (조장)
- 처음 ResNet18·일반 CNN으로 500장을 학습했을 때 정확도가 지나치게 높게 나와 데이터를 다시 확인 → KoDF는 진짜 19명·가짜 30명의 얼굴뿐이라 같은 사람이 학습과 평가에 함께 들어간 것을 발견, 팀에 공유하고 얼굴 800명인 FF++로 먼저 진행하도록 제안
- 이후 영상(Video ID) 단위 분할로 데이터셋 재구성
- 팀원들이 나눠 맡은 EfficientNet-B4·B3·ResNet-50을 GPU 환경에서 모두 학습시키고 앙상블까지 실행 · Early Stopping·MixUp·Backbone Freeze 적용 · Grad-CAM 시각화

## 결과
| 모델 | Accuracy | F1 | AUC |
|---|---|---|---|
| EfficientNet-B0 | 0.6233 | 0.6118 | 0.7027 |
| Ensemble (B4+B3+ResNet-50) | 0.6236 | 0.6164 | 0.7045 |

## 파일 구성
| 파일 | 내용 |
|---|---|
| 01_dataset_sampling.ipynb | 영상 프레임 추출 · 데이터셋별 샘플링 · 메타데이터 저장 · 프레임 검증 |
| 02_EDA.ipynb | 분포 분석 · 데이터 누수 검증 · Split 공정성 · 이미지 품질 분석 |
| 03_Final_model1.ipynb | Video-Level Split · 증강 · EfficientNet-B0(Backbone Freeze) · 3종 앙상블 |
| 04_Final_model2.ipynb | Full Fine-Tuning(차등 학습률) · Mixup/CutMix 앙상블 실험 |

※ 초상권 보호를 위해 영상 및 추출 프레임은 포함하지 않습니다.

## 데이터
AI Hub 딥페이크 변조 영상(KoDF) · Kaggle Celeb-DF(v2) · Kaggle FaceForensics++

## 기술 스택
Python · PyTorch · torchvision · OpenCV · scikit-learn · Grad-CAM
