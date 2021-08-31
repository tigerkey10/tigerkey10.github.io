# 데이터프레임 복습 

x <- c(1,3,5,9)
y = c(1,2,3,4)

y <- as.character(y) # 실수 numerical(double) 벡터 문자열 벡터로 변환

test_dataframe <- data.frame(x, y)

test_dataframe

y <- as.vector(y, mode='numeric') # 실수벡터로 다시 전환
y

test_dataframe

# 행렬객체 생성

X <- matrix(x, nrow=2, ncol=2, byrow=TRUE)
X

X[2, 2]
X[1,2]

### 반복문 연습
for (i in c(1,2,3)) {
    print(i)
}

for (i in x){
    print(i)
}

---
print('a')
