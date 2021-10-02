# 회귀분석 복습 

# 선형예측모형=선형회귀모형


x = c(50, 60, 55, 40, 80, 35, 45, 65, 30, 70)
y = c(100, 160, 120, 90, 150, 130, 110, 120, 80, 140)
length(x)== length(y)

plot(x=x, y=y)

# 단순회귀분석 예 
# 종속변수에 영향 미치는 독립변수 1개인 경우 
# age 독립변수 와 heartrate 종속변수 사이 관계를 
# 설명하는 선형모형을 만들자. 

x = c(18, 23, 25, 35, 65, 54, 34, 56, 72, 19, 23, 42, 18, 39, 37)
y = c(202, 186, 187, 180,156, 169, 174, 172, 153, 199, 193, 174, 198, 183, 178)

length(x) == length(y)

# (x,y) 벡터로 스캐터 플롯 그리기
plot(x,y) 

# 선형모형 정의 
lm(y~x)
# intercept 는 최적 b0 값, x값은 최적 b1 값이다. 
summary(lm(y~x)) # 회귀분석 결과 출력
# coefficients 에 나오는 값들이 가중치 값들이다. 

# p-value 는 각 값들이 의미가 있는지 없는지 알려준다.
# 별 세개 는 1% 유의수준에서 귀무가설이 의미가 있다(대립가설 채택) 는 의미다. 
# 귀무가설: 선형모형 의미 없다.
# 대립가설: 선형모형 의미 있다. 

# 회귀선 그리기 명령
# 회귀선만 그린다. 회귀분석 명령 아니다. 
abline(lm(y~x))

# 선형회귀모형으로 특정 독립변수일 때 예측값 출력하기 
lm_result = lm(y~x)
# 데이터프레임만 들어갈 수 있다.
predict(lm_result, data.frame(x=c(50,79)))# 독립변수 x 값 50, 79 일 때 예측값

# 90% 신뢰수준 에서 예측값을 구할 수 있다. 
predict(lm_result, data.frame(x=sort(x)), level=.9, interval='confidence')
# 99% 신뢰수준에서 예측값 구할 수 있다. 
predict(lm_result, data.frame(x=sort(x)), level=.99, interval='confidence')

# 다중회귀분석
x = 1:10
y = sample(1:100,10)
z = x+y
lm(z~x+y)

z = x+y+rnorm(10,0,2) # 10개 표본 추출, 정규분포 기댓값 = 0 , 표준편차 = 2
summary(lm(z~x+y))

