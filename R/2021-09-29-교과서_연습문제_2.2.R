# Excercises 2.2, p.44
#8. 

read.csv('ex2_8.csv') -> ex
View(ex)

# a. for x2, how many of the observations are equal to 2?
length(ex[which(ex$x2 == 2),]$x2)
# 50 개다. 

# b. Sort x1 and then x2, both in ascending order. after the variables have been sorted, what
# is the first observation for x1 and x2?

head(ex[order(ex$x1, ex$x2),])[1,]

# c. sort x1 and then x2, both in descending order. after the variables have been sorted, 
# what is the first observation for x1 and x2? 

ex[order(-ex$x1, -ex$x2),][1,]

# d. sort x1 in ascending order and x2 in decreasing order. after the variables have been sorted,
# what is the first observation for x1 and x2?

ex[order(ex$x1, -ex$x2),][1,]

# e. how many missing values are there in x1 and x2?
sum(is.na(ex$x1))
sum(is.na(ex$x2))

#---

#9. 
read.csv('ex_2_9.csv') -> ex2
head(ex2)
dim(ex2)

# a. for x1, how many of the observations are greater than 30? 
# NA값 제거 
ex2[complete.cases(ex2),] -> ex2

sum(ex2$x1 > 30) 
# 17개 

# b. sort x1, x2 and then x3 all in ascending order. after the variables have been sorted, 
# what is the first observation for x1, x2 and x3? 

ex2[order(ex2$x1, ex2$x2, ex2$x3), ][1,]

# c. sort x1 and x2 in descending order and x3 in ascending order. 
# after the variables have been sorted, what is the first observation for x1, x2, and x3? 

ex2[order(-ex2$x1, -ex2$x2, ex2$x3), ][1,]

# d. how many missing values are there in x1, x2, and x3? 
sum(is.na(ex2$x1))
sum(is.na(ex2$x2))
sum(is.na(ex2$x3))

#----

# 10. 

read.csv('ex2_10.csv') -> ex3
head(ex3)
summary(ex3)

# a. for x4, how many of the observations are less than three? 
sum(ex3$x4 < 3)
# 85개다. 

# b. sort x1, x2, x3 and then x4 all in ascending order. after the variables have been sorted,
# what is the first observation for x1, x2, x3 and x4? 

ex3[order(ex3$x1, ex3$x2, ex3$x3, ex3$x4), ][1,]

# c. sort x1,x2,x3 and then x4 all in descending order. after the variables have been sorted, 
# what is the first observation for x1,x2,x3, and x4? 

for (i in list(ex3$x1, ex3$x2, ex3$x3, ex3$x4)) {
  print(typeof(i))
}

ex3[order(-ex3$x1, -ex3$x2, -ex3$x3, -ex3$x4), ][1,]

# d. how many missing values are there in x1, x2, x3, and x4? 

for (i in list(ex3$x1, ex3$x2, ex3$x3, ex3$x4)) {
  print(sum(is.na(i)))
}
sum(is.na(ex3))
# 0개. 하나도 없다 .

# e. how many observations are there in each category in x4? 

hist(ex3$x4, labels=TRUE, nclass=3)
max(ex3$x4)
min(ex3$x4)

# 1: 40, 2:45, 3:33 개 있다. 

# ---

# Applications
# 11. SAT dataset
read.csv('sat.csv') -> sat
head(sat)

# a. sort the data by writing scores in descending order. which state has the highest average writing
# score? what is the average math score of that state? 
sat[order(-sat$Writing), ] -> sorted_sat
head(sorted_sat)
# 미네소타, with 643. 미네소타의 math score : 655

# b. sort the data by math scores in ascending order. which state has the 
# lowest average math score? what is the average writing score of that state? 

sat[order(sat$Math), ] -> sorted_math
head(sorted_math)[1,]
# virgin islands : 445, , writing score : 490

# c. how many states reported an average math score higher than 600? 

dim(sat)[1] == length(unique(sat$State))

sum(sat$Math > 600)
# 13 개 있다. 

# d. how many states reported an average writing score lower than 550? 

sum(sat$Writing < 550)
# 25개 있다. 

#---

# 12. Fitness dataset

read.csv('fitness.csv', na.strings=c(''," ", NA)) -> fitness

head(fitness)
View(fitness)

# a. sort the data by annaul income. of the 10 highest income earners, how many of them are
# married and always excercise? 

fitness[order(-fitness$Income), ]-> sorted_fitness
head(sorted_fitness, 10) -> highest
dim(highest[which(highest$Exercise == 'Always' & highest$Married=='Yes'),])[1]
# 1명 있다. 

# b. sort the data by martial status and excercise both in descending order. how many of the 
# individuals who are married and excercise someitmes earn more than $110,000 per year? 

fitness[order(-xtfrm(fitness$Married), -xtfrm(fitness$Exercise)), ] -> sorted_fit

# na 값 갯수는? 
sum(is.na(sorted_fit))
# 7개다. 제거하자. 
sorted_fit[complete.cases(sorted_fit),] -> sorted_fit # NA가 제거되었다. 

# 각 행이 모두 유니크한가? 
length(unique(sorted_fit$ID)) == dim(sorted_fit)[1] # 유니크하다. 

# 이제 조건에 맞는 데이터만 추출하자. 
fitness[which(fitness$Married=='Yes' & fitness$Exercise == 'Sometimes'), ] -> conditional_fit

length(unique(conditional_fit$ID)) == dim(conditional_fit)[1]

sum(conditional_fit$Income > 110000)
# 조건 모두 충족하는 사람 9 명 있다.

# c. How many missing values are there in each variable? 

sum(is.na(fitness$ID))
sum(is.na(fitness$Exercise))
sum(is.na(fitness$Married))
sum(is.na(fitness$Income))

# d. how many individuals are married and unmarried? 

fitness[complete.cases(fitness),]-> fitness
as.factor(fitness$Married)-> married
length(married)
sum(married == 'Yes') -> number_of_yes
length(married) - number_of_yes -> number_of_no
print(number_of_yes) # 결혼한 사람 276명
print(number_of_no) # 비혼자 131 명

number_of_yes+number_of_no == length(married)

# e. how many married individuals always excercise? how many unmarried individuals never excercise?











