gig -> gig
# 데이터를 살펴보자. 
head(gig)
as.factor(gig$Job) -> job
unclass(job) -> job_unclass
min(job_unclass)
max(job_unclass)

hist(job_unclass)
attr(job_unclass, 'levels')[5]


read.csv('gig.csv', na.strings = c('', ' ', NA)) -> gig
sum(is.na(gig))
# 26개 결측값 있다. 
dim(gig)

# 결측값 있는 행을 통째로 제거한다
gig[complete.cases(gig),] -> gig2
dim(gig2)

sum(is.na(gig2)) # 이제 결측값 없다. 

# ---

# obtain information on the numbers of employees who 
# 1. worked in the auto industry
head(gig2)

dim(gig2[gig2$Industry=='Automotive',])
# 186명이다. 

# 또는 
dim(gig2[which(gig2$Industry == 'Automotive'),])
which(gig2$Industry=='Automotive')
# 186명이다. 

# 2. Earned more than 30$/hour

# 첫번째 방법 
dim(gig2[which(gig2$HourlyWage > 30),])
# 515명이다. 
gig2[which(gig2$HourlyWage > 30),] -> more_than_30
sum(more_than_30$HourlyWage > 30)

# 두번째 방법
sum(gig2$HourlyWage > 30)
# 515명이다. 

# 3. worked in the auto industry & earned more than 30$/hour
gig2[which(gig2$Industry == 'Automotive' & gig2$HourlyWage > 30),] -> auto_30
head(auto_30)
sum(auto_30$Industry == 'Automotive')
dim(auto_30)[1]

min(auto_30$HourlyWage)
# 178명이다. 


# 4. Hourly wage of the lowest- and highest- paid employees overall & 
# accountants who worked in the auto and tech industries

# 앞에꺼는 스칼라 값 두개로 빼고, 뒤에 꺼는 데이터셋으로 빼자. 

gig2[which(gig2$Job=='Accountant' & gig2$Industry == c('Automotive', 'Tech')),] -> technauto 
technauto #accountants who worked in the auto and tech industries

which(gig2$HourlyWage == min(gig2$HourlyWage))
gig2$HourlyWage[448] # lowest hourly wage

which(gig2$HourlyWage == max(gig2$HourlyWage))
gig2$HourlyWage[104] # highest hourly wage

#---

# import gig data file into a dataframe and label it mydata

as.data.frame(gig) -> mydata
dim(mydata)


# verify that fist employee is an analyst who worked in the construction industry 
# and made $32.81/hour

head(mydata)
mydata[1,]

which(is.na(mydata$Industry)) # which 명령은 조건에 맞는 행번호를 출력한다
mydata[24,]
mydata[139,]


# identify employees in the automotive industry with which function and count the number of 
# employees using length function

mydata[which(mydata$Industry == 'Automotive'),]->auto
length(auto$EmployeeID)
#또는

length(which(mydata$Industry == 'Automotive'))

# determine number of employees earning more than $30/hour
length(unique(mydata$EmployeeID)) == dim(mydata)[1]

length(which(mydata$HourlyWage > 30))
# 536개다. 

# count employees in automotive industry & earning more than 45/hour, use & operator
length(which(mydata$Industry=='Automotive' & mydata$HourlyWage > 45))

# order 함수 : 값들을 오름차순 정렬했을 때 인덱스만 반환한다. 
order(c(5,3,7,1))

# sort the hourlywage variable and store ordered data in a new dataframe
order(mydata$HourlyWage) -> or_index
as.data.frame(mydata$HourlyWage[or_index]) -> hourly
head(hourly)

# 또는
mydata[or_index, ] -> hourly
head(hourly)

# sort it with descending order 내림차순 정렬하기 
mydata[order(mydata$HourlyWage, decreasing=TRUE),] -> hourly2

# 여러 변수에 대해 오름차순 정렬하기 
order(mydata$EmployeeID, mydata$HourlyWage, mydata$Job) -> new_order# 문자열도 정렬된다
mydata[new_order,] # employee id, hourly wage, job 순으로 정렬된다. 

# to sort data by industry and job classification in ascending order and then by 
# hourly wage in decreasing order, insert minus sign in front of the hourly wage variable
# 실수로 된 변숫값 앞에만 - 써서 내림차순 정렬시킬 수 있다. 
mydata[order(mydata$Industry, mydata$Job, -mydata$HourlyWage),]

# categorical variable (범주형 확률변수) 를 - 써서 내림차순 정렬하는 법
# xtfrm() 을 쓰면 된다. : categorical 값들을 각각 특정 실수값에 대응시켜 준다.

View(mydata[order(xtfrm(mydata$Industry)),])
mydata[order(-xtfrm(mydata$Industry), mydata$Job, mydata$HourlyWage),] -> sorted_data
View(sorted_data)

# to export sorted data from step m as a comma-separated value file, use the write.csv function
# csv 파일로 저장하는 법
write.csv(sorted_data, 'sorted_data.csv')


#---
# find the number of missing values for each variable
sum(is.na(sorted_data$HourlyWage))

sum(is.na(sorted_data$Industry))

sum(is.na(sorted_data$Job))

#---

# obtain information on the number of employees who
# 1. worked in the auto industry

mydata[complete.cases(mydata),] -> mydata
dim(mydata)

dim(mydata[which(mydata$Industry == 'Automotive'),])
# null값 제거하면 186

dim(gig[which(gig$Industry=='Automotive'),])
# null값 있을 때는 190

# 2. Earned more than $30/hour
sum(gig$HourlyWage > 30)
# 536명

# 3. worked in the auto industry & earned more than 30$/hour
length(gig[which(gig$Industry=='Automotive' & gig$HourlyWage > 30),]$EmployeeID)

# Obtain information on the number of employees who
# hourly wage of the lowest - and highest-paid employees overall & 
# accountatns who worked in the auto and tech industries

gig[which(gig$Industry==c('Automotive', 'Tech')),] -> account
min(account[which(account$Industry == 'Automotive'),]$HourlyWage)
max(account[which(account$Industry == 'Automotive'),]$HourlyWage)


