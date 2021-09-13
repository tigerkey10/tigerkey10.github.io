a = c(1,2,3)

b = c(3,4,5)

a%in%b

x = 1:5
y = c(3:7)

# 1. 
x>7

# 2. 

x<= 3

# 3. 
x[(x>2)&(x<=5)]

# 4. 요소별 연산 할 것이다. 
x
y
x-y# 요소별 연산


# 리스트 
c = c(1,2)
d = 3
list(x,y,c,d)

z = 1:8
is.vector(z)

z = c(1:8)
is.vector(z)

mode(z)
typeof(z)

# 객체의 속성
names(x)

dim(x)
attributes(x)

class(x)

x.names('a','b','c','d','e')
names(x)<- c('a')
names(x)

# array()
array(1:30, dim=c(2,3,5))

# list()

A <- list(c(1,2,3), c('a','b','c','d'), c(TRUE, FALSE, TRUE))
A

B <- list(matrix(c(1,2,3), ncol=1, nrow=2), c('a','b','c'), c(TRUE, FALSE, TRUE))
B

x= c(1,2,3)
y = c(1,2,9)
z = as.matrix(data.frame(x,y))
z

# ---

x <- c('male', 'female', 'female', 'male', 'female')

x # 문자열 자료형

factor(x) -> x2

x2

unclass(x2)

attributes(x2)

# 03. 연습문제 
# 문자형 벡터를 숫자형 벡터로 전환하고, 
# 속성값을 확인하는 명령문을 작성하라. 

c('1','5','10') -> z
as.integer(z)-> z

typeof(z)

# 04. 연습문제 
# 하나의 벡터에 문자형, 숫자형, 요인형, 논리형 값들이 모두 
# 존재할 때, 어떠한 자료형으로 강제변환되어 나타나는가? 

#--> 문자형 > 숫자형 > 논리형 
# 문자형으로 변환된다. 

f1 = function(x) {
    x+3
}
f1(8)-> d
d

round(0.1968, digits=2)

1:50 -> a
# 표준편차 
sd(a)

# 중위값
median(a)

# 합계 
sum(a)

# 연습문제 08
dices = 1:6

r_dice = function() {
    dice_value = sample(1:6, size=2)
    return (sum(dice_value))
}
r_dice()


a = 3
b = 2

if (a==1) {
    print('a')
} else {
    print('b')
    }

# ifelse 조건문
a = 7
ifelse(a==7, '참', '거짓')

for (i in c(1,2,3,4,5)) {
    print(i)
}

a = 0
repeat {
    print('k')
    a = a+1
    if (a==4) break
}

# 연습문제 10
# z가 홀수이면 x와 y 를 덧셈, 짝수이면
# 뺄셈을 실행하는 나만의 함수를 정의하고, 아래와 같을 경우의 결과 값을 적으시오. 

z_xy = function(x,y,z) {
    if (z%%2 == 0) {
        return (x-y)
    } else {
        return (x+y)
    } 

}
z_xy(1,2,5)
z_xy(4,2,3)
z_xy(1,3,2)

library(ggplot2)

a = 3
ifelse (a == 3, 'true', 'false')

if (a==3) {
    print(a)
} else if()

a = 91
if (a == 3) {
    print('a')
} else if (a == 4) {
    print('b')
} else if (a == 7) {
    print('c')
} else {
    print('d')
}

