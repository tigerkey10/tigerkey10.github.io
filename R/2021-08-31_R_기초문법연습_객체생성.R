# 데이터프레임 생성해보자. 

x<- c(1,2,3) # 실수벡터 
n <- c(6,7,8) # 실수벡터 

data.frame(x, n) # 데이터프레임 생성,x와 n 실수벡터를 열로 쌓아서 만든 '행렬'의 한 종류

a = c(1,2,3)
b = c(1,2,3)

test_dataframe <- data.frame(a,b)

a_numeric = as.vector(a, mode="character") # 실수 numeric 벡터를 문자열 character 벡터로 변경

mode(a_numeric) # a_numeric 벡터의 자료형은? 
a_numeric # 문자열 벡터로 변경됨. 

###

# 데이터프레임 객체에 속성 부여해보자. 

names(test_dataframe) <- c(1,2)
names(test_dataframe)

dim(test_dataframe) # 데이터프레임의 사이즈 

attributes(test_dataframe)

names(test_dataframe) <- 'A'

dim(test_dataframe) <-NULL
dim(test_dataframe)

### 

# 데이터프레임에 속성을 부여하는 법칙이 뭐지? 

x = c(1,3,5)

names(x) <- c('a','b','c')

names(x) <- NULL
attributes(x)


# 속성은 객체에 부여할 수 있다. 


x = 3
names(x) <- 'A'
names(x)

# 함수 생성 

test_function <- function(a,b){
    result <- a + b
    return (result)} # return 에 () 괄호 쳐 줘야 한다. 

test_function(3,8)

# 제곱
y <- 3
y^3
round(3.56, digits=1)
abs(-3.56)

k = 1

repeat {
    k = k+ 1
    if (k == 5) {break}

}
print(k)

# next (파이썬의 continue)

for (i in c(1:10)) {
    if (i == 4) {next}
    print(i)}


###

5%/%2

# 함수 정의 연습 -2

z_xy = function(z, x, y) {
    if(z%%2==0){
        result = x-y
        return(result)
    }
    else {
        result = x+y
        return(result)
    }
}
z_xy(5,2,3)
z_xy(4,2,3)
z_xy(7,2,3)

