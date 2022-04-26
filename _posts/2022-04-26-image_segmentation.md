---
title : "[딥러닝 연구주제 탐색] Image segmentation"
excerpt : "관심있는 딥러닝-컴퓨터 비전 분야 탐색"

categories : 
- Research Topic
- Machine Learning
- Deep Learning

tags : 
- [Research Topic, Machine Learning, Deep Learning]

toc : true 
toc_sticky : true 
use_math : true

date : 2022-04-26
last_modified_at : 2022-04-26

---

*아래는 관심있는 딥러닝-컴퓨터비전 분야인 Image segmentation 에 대해 알아보고, 내용을 간략히 기록한 글이다. 완성된 문서가 아니며, 계속 add-up 해 나가기 위해 작성했다. 

---

# Image segmentation

## 정의 

이미지 구성하는 단일 픽셀 하나하나를 특정 클래스로 분류하는 작업.

## 종류 

### 1. Semantic segmentation

단일 픽셀 하나하나를 특정 클래스로 분류한다. 대신, 동일한 객체 끼리는 구분 못한다. (이미지에 서로 다른 고양이 2마리 있으면 둘 다 똑같은 고양이로 분류한다.)

### 2. Instance segmentation

Semantic segmentation 처럼 단일 픽셀 하나하나를 특정 클래스로 분류하되, 이미지에 나타난 모든 객체를 구분해서 분류한다. (고양이 1, 고양이 2, 고양이 3)

# Image segmentation 적용 가능 분야 

- Handwriting Recognition: 손글씨 인식 할 수 있다. 
- Portrait mode photo: 인물모드 촬영 
- YouTube stories: 유튜브 스토리 촬영 중 배경 마음대로 바꿀 수 있음
- Virtual make-up
- Virtual try-on: 이미지에 가상으로 옷 착용 해볼 수 있다. 
- Visual Image Search: 사용자가 업로드한 이미지에서 특정 객체만 뽑아내서, 비슷한 객체 가진 이미지들 웹에서 찾아준다. 
- Self-driving cars: 자율주행 기술은 주변 사물에 대해 단일 픽셀 레벨에서 완벽한 이해 필요하다. Image segmentation 이 이미지(프레임)에서 각 객체 완벽하게 구분.분할 해 내기 위해 사용된다. 

https://nanonets.com/blog/semantic-image-segmentation-2020/

---

# Semantic Image Segmentation with DeepLab in TensorFlow

### Cutting-edge semantic image segmentation model: DeepLab-v3+

(2018 기준)

- 텐서플로 기반 모델이다. 
- CNN backbone 모델 위에 DeepLab-v3+ 얹은 형태다. 
- 구글은 PascalVOC 2012 & Cityscapes 데이터에 대해 사전훈련된 모델 제공한다.  
- 방법론, 하드웨어, 데이터셋 발전으로 최근 semantic image segmentation system 들 성능 비약적 향상 되었다. 

‘딥 라벨링’은 신경망 사용해 각 픽셀마다 분류 예측값을 할당하는 방식으로 동작한다. 

https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html



