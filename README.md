# Yelp Recommendation

##Overview
Yelp에서 고객들의 이용했던 business의 별점(stars rating) 정보를 기반으로 이용자들에게 추천 business를 제공한다  

### Target Customer
Yelp application을 사용하는 customer

## Dataset
[Yelp Dataset][ydlink]에 있는 데이터 중에 사용자들의 business review 점수를 활용
* yelp_academic_dataset_review.json: 사용자-비지니스간의 별점 정보
* yelp_academic_dataset_user.json: 사용자 정보
* yelp_academic_dataset_business.json: 비지니스 정보

## Feature Transformation
* index string to numeric
    * review dataset의 user_id와 business_id가 string 타입이기 때문에 학습을 위해 numeric 타입으로 변경

## Training
ALS model을 이용하여 training

## Evaluation
* Regression evaluator를 이용하여 Root mean squre error를 측정
* Ranking Metric을 이용하여 Mean average precison 측정

## Implementation
### Test environment
* single machine
    * 4 core
    * 16G memory
* run spark job as Local mode on Intellij IDE
* vm option: -Xms8g -Xmx8g
* 6.33GB 정도의 원본데이터를 로컬에서 테스트 하기 위해 0.5 샘플링 하여 테스트 진행

### Prerequisite
* Yelp 데이터셋을 project의 yelp-dataset 디렉토리에 복사

### Run
* Driver class
    * ALSRunner: ALS 모델을 이용하여 추천 데이터 생성
    * ALSPipelineRunner: 위 모델에서 최적으로 결과를 찾기 위해 Pipeline을 이용하여 반복적으로 테스트         

### Results Example
```
# 3 recommendForUserSubset
+-------+---------------------------------------------------------------+
|user_no|recommendations                                                |
+-------+---------------------------------------------------------------+
|444410 |[[207801, 143.54684], [195392, 143.28445], [179736, 141.68715]]|
|347323 |[[207659, 336.00647], [209305, 333.65787], [131683, 330.3789]] |
|203854 |[[201256, 315.79272], [206973, 313.43314], [195286, 312.62524]]|
+-------+---------------------------------------------------------------+
```

```
# explode and add user name and business name
+-------+------+-------------------+----------------------------------+
|user_no|name  |reco               |name                              |
+-------+------+-------------------+----------------------------------+
|203854 |Lauren|[195286, 312.62524]|On Q Wax & Hair Studio            |
|203854 |Lauren|[206973, 313.43314]|Carter Electrical                 |
|203854 |Lauren|[201256, 315.79272]|Anew Medspa                       |
|347323 |Diana |[131683, 330.3789] |The Art Department                |
|347323 |Diana |[209305, 333.65787]|Hometown Heroes                   |
|347323 |Diana |[207659, 336.00647]|A-1 Auto Service                  |
|444410 |Kevin |[179736, 141.68715]|Breakthrough Performance and Rehab|
|444410 |Kevin |[195392, 143.28445]|Life Storage                      |
|444410 |Kevin |[207801, 143.54684]|Toy Florist                       |
+-------+------+-------------------+----------------------------------+
```
* Root-mean-square error: 5.278573251959344
* Mean average precision: 0.9853041114509005
* precision at 5: 0.2246687054026504



[ydlink]: https://www.kaggle.com/yelp-dataset/yelp-dataset