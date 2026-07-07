# deepfake_dectection

KoDF·CelebDF·FaceForensics++ 3개 데이터셋을 **video 단위로 분할(데이터 누수 방지)**하고,
EfficientNet-B0 전이학습 + 3종 모델 앙상블로 real/fake를 분류.
**Grad-CAM으로 판별 근거를 시각화**하고, 프레임 단위 학습의 한계와 개선 방향(3D CNN·CNN-LSTM)까지 분석
(Test AUC 0.70).

** 초상권을 침해하는 영상이나 추출된 프레임 등은 포함하지 않음.

`Python` `PyTorch` `EfficientNet` `OpenCV` `Grad-CAM`
