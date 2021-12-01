# 독립이표본 검정 - 정규 모집단 

A <- c(79.98, 80.04, 80.02, 80.04, 80.03, 80.03, 80.04,79.97, 80.05, 80.03, 80.02, 80.00,80.02)
length(A)
B <- c(80.02, 79.94, 79.98, 79.97, 79.97, 80.03, 79.95, 79.97)
length(B)

boxplot(A,B)

t.test(A,B)

# 소표본. 정규성. 독립. 
t.test(A,B, var.equal=F)

# 소표본. 정규성. 독립. 등분산성
t.test(A,B, var.equal=T)

# 등분산성 검정 
sd(A)/sd(B)

# 쌍체비교 
d = c(2,8,10,6,18,10,4,26,18,-8,0,32,0,-4,10)
length(d)
sd(d)
mean(d)

x <- c(70, 80, 72, 76, 76, 76, 72, 78, 82, 64, 74, 92, 74, 68, 84)
y <- c(68, 72, 62, 70, 58, 66, 68, 52, 64, 72, 74, 60, 74, 72, 74)

t.test(x,y, paired=T, conf.level= 0.95)

t.test(d)

# 대표본, 두 모비율 p1 p2 적당한 값일 때 
# 모비율 비교 
# 이항분포 정규근사 하고 연속수정 함. 
prop.test(x=c(88, 126), n = c(100, 150))

p1 = 88/100
p2 = 126/150

p1-p2

xx = sqrt((p1*(1-p1)/100) + (p2*(1-p2)/150))*1.96

0.04 + xx
0.04 - xx

## 1 Dec 2021
# 단순선형회귀모형 

x <- c(3,3,4,5,6,6,7,8,8,9)
y <- c(9,5,12,9,14,16,22,18,24,22)

length(x) == length(y)

# always plot my data 
plot(x,y)
# 선형상관관계가 보인다. 
cor(x,y) # 피어슨 상관계숫값도 0.9로 매우 높다.

# 단순선형회귀모형을 설정모형으로 삼고, 적합모형을 찾자. 
fit = lm(y~x)
summary(fit)

# 잔차값만 보려면 
resid(fit) # 10개 잔차 
length(x)

fitted(fit) # 예측값(표본회귀선 위 값들) 

y - fitted(fit) -> err 
err # 잔차를 구하는 또 다른 방법 

print(err); print(resid(fit))

# 회귀모형에 추정된 beta0, beta1 즉 회귀계수만 보고 싶을 때 
coef(fit)
fit$coefficients

confint(fit, levels=0.95)

# 분산분석표 anova table
anova(fit)

# 스캐터 플롯에 회귀선 표시 
abline(fit)

