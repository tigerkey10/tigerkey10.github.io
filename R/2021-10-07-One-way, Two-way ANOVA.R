# ANOVA one-way 실습 
f1 <- c(50.5, 52.1, 51.9, 52.4, 50.6, 51.4, 51.2, 52.2, 51.5, 50.8)
f2 <- c(47.5, 47.7, 46.6, 47.1, 47.2, 47.8, 45.2, 47.4, 45.0, 47.9)
f3 <- c(46.0, 47.1, 45.6, 47.1, 47.2, 46.4, 45.9, 47.1, 44.9, 46.2)

y <- c(f1, f2, f3)

n<- rep(10,3)
group<- rep(1:3, n)
group

# combining into data.frame
group_df = data.frame(y, group)
group_df

sapply(group_df, class)

# transfrom from integer to factor

transform(group_df, group=factor(group)) -> group_df
sapply(group_df, class)
# 요인수준에 해당하는 변수는 반드시 요인형 factor 여야 한다. 

tapply(y, group, summary) 
# group 값에 따라 y에 summary를 적용한다. 

# one way anova
summary(aov(y~group, data=group_df))
# p-value 매우 작다. 귀무가설 기각, 대립가설 채택한다. 

# 세 분포의 등분산성 검정
bartlett.test(y~group, data=group_df)
# 만족한다. 

#-------

# two-way anova , 관측값이 하나인 경우 
# 2개 요인, 3개 이상의 그룹 

일원분산분석은 1개 요인 r개 요인수준 분포들의 평균 차이를 비교하는 것이다. 

이원분산분석은 2개 요인 r개 요인수준 간 조합 각각이 개별 집단이 되어, 이 집단 분포들 평균이 같은지 다른지 비교한다. 

rep(c('1000', '1500', '1800'), 2) -> car_type
as.factor(car_type) -> car_type

insurance <- as.factor(c(rep('K',3), rep('M',3)))
insurance

y<- c(140, 210, 220, 100, 180, 200)

data.frame(car_type, insurance, y) -> data

summary(aov(y~car_type+insurance, data=data))

#-----

# 이원분산분석 예제 

data1 <- c(71,77, 78, 76, 77, 78, 71, 70, 69)
data2 <- c(80, 76, 80, 79, 78, 77, 73, 71, 70)
c(data1, data2) -> score

rep(c(1,1,1,2,2,2,3,3,3),2) -> class
as.factor(class) ->class

rep('M',9) ->M
rep('F',9)-> F
c(M, F) -> gender
as.factor(gender) -> gender

data.frame(score,class, gender) -> score
score
class(score$score)

summary(aov(score~class+gender+class:gender, data=score))

shapiro.test(data1)
shapiro.test(data2)

bartlett.test(list(data1, data2))

t.test(data1, data2, alternative = 'two.sided', var.equal = TRUE)


summary(aov(score~class, data=score))

summary(aov(score~gender, data=score))

summary(aov(score~class+gender, data=score))

score

#-----------------------------------

# MANOVA 다변량분산분석 
iris12 = iris
head(iris)

# 귀무가설: 모든 평균들은 다 같다. 
# 대립가설: 귀무가설이 아니다. 

# 종속변수1: 꽃받침 길이 / 종속변수2: 꽃잎 길이

sepl <- iris$Sepal.Length
petl <- iris$Petal.Length

# do MANOVA
# 종속변수는 sepl, petl 두 개 
# 독립변수는 species
summary(manova(cbind(sepl, petl)~Species, data=iris))
# p-value 가 모든 유의수준보다 작다. 귀무가설 기각하고 , 대립가설 채택한다. 
# 곧, 꽃 종에 따라 꽃받침, 꽃잎 길이에는 통계적으로 유의미한 차이가 존재한다. 

# 독립변수에 대한 종속변수 각각의 결과를 보자. 
summary(aov(cbind(Sepal.Length, Petal.Length)~Species, data=iris))
#== 위 명령은 아래와 같다. 
summary(aov(Sepal.Length~Species, data=iris))
summary(aov(Petal.Length~Species, data=iris))

# 이 경우 sepal length, petal length 둘 다 species에 의해 영향을 받는다. 

#----------

# 공분산분석 ANCOVA
anova(lm(종속변수~공변량+독립변수, data=데이터셋))
#---------
library(moonBook)
install.packages('moonBook')
library(moonBook)
acs -> acs
head(data, 2)

out1 = lm(age~smoking, data=data)
summary(out1)

anova(out1) # smoking을 독립변수로, age를 종속변수로 삼아 anova 하겠다. 
# smoking은 age에 영향 미친다. 
summary(out1)
# 회귀분석 결과를 보니까 smokingsmoker가 age에 영향을 미친다. 

#---------

head(acs,1)

out1 <- lm(age~smoking, data=acs)
summary(out1)

lm(age~BMI+smoking, data=acs) -> out2
summary(out2)

# 아래 둘은 결과가 완전히 같다. 
anova(out2) #1. 선형모형을 이용해서 anova 한 경우

summary(aov(age~BMI+smoking, data=acs)) # 2. two-way anova 를 실시한 경우

# 공분산분석은 결국 이원분산분석 과도 같다. 
# 이원분산분석인데 독립변수 2개 중 하나가 공변량인 경우다. 



