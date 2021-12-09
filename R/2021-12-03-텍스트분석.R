read.csv('/Users/kibeomkim/Desktop/유튜브 광고 데이터셋/50퍼센트 이상.csv') -> more_than

R.version

more_than[2:1826, ] -> nalchi
more_than[1828:2041,] -> views
more_than[2043:2111,] -> spen
more_than[2113:2231,] -> togo
more_than[2233:2263,] -> ultra 
more_than[2265:2294,] -> memory
more_than[2296:3037,] -> unboxing
more_than[3039:3347,] -> jp_ep3
more_than[3349:3547,] -> nalchi_unboxing
more_than[3549:3571,] -> panthom
more_than[3573:3579,] -> epicineveryway
more_than[3581:3597,] -> bang
more_than[3599:3695,] -> nobcut
more_than[3697:3724,] -> meperfect


#-----------------------------------

# 텍스트 분석 
install.packages("tidyverse")
install.packages('reshape')
install.packages('stringr')

library(tidyverse)
library(reshape)
library(stringr)
library(multilinguer)

library(rJava)
library(memoise)
library(stringr)
library(hash)
library(tau)
library(Sejong)
library(RSQLite)
library(devtools)
library(remotes)
library(KoNLP)
useNIADic()

#-----------------------------


nalchi[, 'X.4']

extract_nouns = function(name, i) {
  a = SimplePos22(name$X.4[i])
  r = melt(a)
  result = str_match(a, '([가-힣]+)/N')
  rr = result[,2]
  nouns = rr[!is.na(rr)]
  if (str_detect(name$X.5[i], '[0-9+K]') == TRUE) {
      el = gsub('\\.', '', name$X.5[i])
      heart = as.integer(gsub('K', '', el))*100
  } else {
      heart = as.integer(name$X.5[i])
  }
  nouns = rep(nouns, heart)
  return(nouns)
}


noun_list = c()
for (i in 1:length(nalchi$X.4)) {
  noun_list = c(noun_list ,extract_nouns(i))
}
noun_list 

length(noun_list)

# 한 단어 짜리 조사 제거 
new_list = c()
a = 1
for (i in noun_list) {
  if (nchar(i) == 1) {
    next
  }
  new_list[a] = i
  a = a+1
}
head(sort(table(new_list), decreasing=T),100)

#--------------------------위 과정 하나의 함수로 구성 
extract_freq_words = function(data) {
  noun_list = c()
  for (i in 1:length(data)) {
    noun_list = c(noun_list, extract_nouns(i))
  }
  new_list = c()
  a = 1
  for (i in noun_list) {
    if (nchar(i) == 1) {
      next
    }
    new_list[a] = i
    a = a+1
  }
  return(new_list)
}

extract_freq_words(nalchi) -> extract
head(sort(table(extract), decreasing=T),80)

#------------------------------


new_list = gsub('한국', '한국적멋', extract)
new_list = gsub('개귀', '개귀여움', extract)
new_list = gsub('퀄이', '높은퀄리티', extract)
new_list = gsub('이거', '대박', extract)
new_list = gsub('릴보이가', '릴보이', extract)
new_list = gsub('부분', '좋음', extract)
new_list = gsub('마인드', '좋은기획', extract)
new_list = gsub('조합', '최고조합', extract)
new_list = gsub('아이폰쓰는사람', '좋음', extract)
new_list = gsub('삼성광고말고릴보이같이', '힙함', extract)
new_list = gsub('삼성쓰는사람을', '삼성뽕', extract)
new_list = gsub('잘뽑은', '잘뽑은광고', extract)
new_list = gsub('넘겼', '조회수', extract)
new_list = gsub('가사도', '가사', extract)
new_list = gsub('왤케', '왤케좋음', extract)
new_list = gsub('눈앞', '생생함', extract)
new_list = gsub('아쉽긴', '짧아서아쉬움', extract)
new_list = gsub('힙하고', '힙함', extract)
new_list = gsub('완전', '완전좋음', extract)
new_list = gsub('릴보이랑', '릴보이', extract)
new_list = gsub('다살다', '살다살다', extract)
new_list = gsub('한국적적', '한국적', extract)
new_list = gsub('릴보이님', '릴보이', extract)
new_list = gsub('개띵곡을', '개띵곡', extract)
new_list = gsub('검색해서', '검색해서찾아본광고', extract)
new_list = gsub('홍보', '홍보효과', extract)
new_list = gsub('무엇', '대박', extract)
new_list = gsub('광고티', '광고같지않음', extract)
new_list = gsub('글거리', '오글거리지않아서좋음', extract)
new_list = gsub('센스', '센스있는광고', extract)
new_list = gsub('클릭해서', '일부러찾아본광고', extract)
new_list = gsub('유일', '유일함', extract))
new_list = gsub('릴보이만', '릴보이', extract)
new_list = gsub('아놔', '대박', extract)
new_list = gsub('트웬티원', '발음', extract)
new_list = gsub('성미친', '미친발성', extract)
new_list = gsub('광고인거', '광고인거모를만큼좋음', extract)
new_list = gsub('랩쫄깃해서', '쫄깃한랩', extract)
new_list = gsub('알고리즘에떠서', '내취향광고', extract)
new_list = gsub('뭔데', '대박', extract)
new_list = gsub('릴보이라니', '릴보이', extract)
new_list = gsub('삼성님', '삼성', extract)
new_list = gsub('가슴이', '가슴이웅장', extract)
new_list = gsub('지랄났', '지랄났다', extract)
new_list = gsub('멋있', '멌있다', extract)
new_list = gsub('릴뽀', '릴보이', extract)
new_list = gsub('직접쓰', '직접쓴가사', extract)
new_list = gsub('앨범', '앨범내라', extract)
new_list = gsub('마케팅센', '마케팅센스', extract)
new_list = gsub('스킵못해서', '스킵없이보는광고', extract)
new_list = gsub('잘어울릴줄', '잘어울림', extract)
new_list = gsub('개지리', '개지림', extract)

new_list = gsub('중독', '중독성', extract)
new_list = gsub('중독성성쩐다', '중독성', extract)


new_list = gsub('쓴거', '직접쓴가사', extract)
new_list = gsub('우리', '', extract)
new_list = gsub('경우', '', extract)
new_list = gsub('이번', '', extract)
new_list = gsub('보름달', '', extract)
new_list = gsub('광고', '', extract)
new_list = gsub('보름달', '', extract)
new_list = gsub('삼성', '', extract)


head(sort(table(new_list), decreasing=T), 100)
data_unlist <- Filter(function(x){nchar(x)>=2}, new_list)
head(sort(table(data_unlist), decreasing=T),80)

library(RColorBrewer)
library(wordcloud)
names(head(sort(table(data_unlist), decreasing=T),50))
head(sort(table(data_unlist), decreasing=T),50)
color <- brewer.pal(12, "Set3")
wordcloud(
  names(head(sort(table(data_unlist), decreasing=T),50)),
  head(sort(table(data_unlist), decreasing=T),50), 
  scale=c(5, 1.5), 
  random.order=FALSE, 
  random.color=TRUE,
  colors=color, 
  family='AppleGothic')

#------------------------

head(sort(table(data_unlist), decreasing=T), 8) -> x
plot(x, family='AppleGothic')
x

data_unlist = gsub('삼성', '', data_unlist)
data_unlist = gsub('광고', '', data_unlist)
data_unlist = gsub('진짜', '', data_unlist)
data_unlist = gsub('이번', '', data_unlist)
data_unlist = gsub('이거', '', data_unlist)
data_unlist <- Filter(function(x){nchar(x)>=2}, data_unlist)


head(sort(table(data_unlist), decreasing=T), 8) -> x
plot(x, family='AppleGothic', main='이날치 X 릴보이', xlab='Keyword')
x


extract_freq_words = function(data) {
  noun_list = c()
  for (i in 1:length(data)) {
    noun_list = c(noun_list, extract_nouns(i))
  }
  new_list = c()
  a = 1
  for (i in noun_list) {
    if (nchar(i) == 1) {
      next
    }
    new_list[a] = i
    a = a+1
  }
  return(new_list)
}
#--------------------------------------------------------------
extract_freq_words(views$X.4) -> extract
head(sort(table(extract), decreasing=T),80)

more_than[2:1826, ] -> nalchi
more_than[1828:2041,] -> views
more_than[2043:2111,] -> spen
more_than[2113:2231,] -> togo
more_than[2233:2263,] -> ultra 
more_than[2265:2294,] -> memory
more_than[2296:3037,] -> unboxing
more_than[3039:3347,] -> jp_ep3
more_than[3349:3547,] -> nalchi_unboxing
more_than[3549:3571,] -> panthom
more_than[3573:3579,] -> epicineveryway
more_than[3581:3597,] -> bang
more_than[3599:3695,] -> nobcut
more_than[3697:3724,] -> meperfect

emp = c()


extract_freq_words(views) -> extract1
edited1 = head(sort(table(extract1), decreasing=T),8) ; edited1
extract1 <- Filter(function(x){nchar(x)>=2}, extract1)


extract1 = gsub('웅장함함', '웅장함', extract1)
extract1 = gsub('시작', '', extract1)
extract1 = gsub('세상', '', extract1)
extract1 = gsub('성공', '', extract1)
extract1 = gsub('광고', '', extract1)
extract1 = gsub('이번', '', extract1)
extract1 = gsub('웅장하', '웅장함', extract1)
extract1 = gsub('삼성', '', extract1)
extract1 = gsub('누구', '', extract1)
extract1 = gsub('누구', '', extract1)
extract1 = gsub('성공해서', '성공', extract1)
extract1 = gsub('시작할', '시작', extract1)
extract1 = gsub('성공해서', '성공', extract1)
extract1 = gsub('연출한', '연출', extract1)
extract1 = gsub('성공해서', '성공', extract1)
extract1 = gsub('지점', '', extract1)
extract1 = gsub('페이스데', '', extract1)
extract1 = gsub('몰랐', '', extract1)
extract1 = gsub('이롭', '', extract1)
extract1 = gsub('지점', '', extract1)
extract1 = gsub('거구', '', extract1)
extract1 = gsub('예쁘', '예쁨', extract1)
extract1 = gsub('오지', '오진다', extract1)
extract1 = gsub('웅장', '웅장함', extract1)
extract1 = gsub('이거', '', extract1)
extract1 = gsub('던져지죵', '', extract1)
extract1 = gsub('일만', '', extract1)
extract1 = gsub('발전하면', '발전', extract1)
extract1 = gsub('맞는거', '', extract1)
extract1 = gsub('안준다고', '', extract1)
extract1 = gsub('인가요', '', extract1)
extract1 = gsub('인조인간설이', '인조인간', extract1)
extract1 = gsub('나올줄알', '', extract1)
extract1 = gsub('으로', '', extract1)
extract1 = gsub('잘어울', '잘어울림', extract1)
extract1 = gsub('잘만들었', '잘만들었음', extract1)
extract1 = gsub('잘생겼', '잘생겼음', extract1)
extract1 = gsub('고싶', '', extract1)
extract1 = gsub('같았구', '', extract1)
extract1 = gsub('고급진', '고급스러움', extract1)
extract1 = gsub('부분', '', extract1)
extract1 = gsub('세련됬네', '세련됨', extract1)
extract1 = gsub('지점', '', extract1)

plot(edited1, family='AppleGothic', main='views', xlab='keyword')

library(RColorBrewer)
library(wordcloud)
names(head(sort(table(data_unlist), decreasing=T),50))
head(sort(table(data_unlist), decreasing=T),50)
color <- brewer.pal(12, "Set3")
wordcloud(
  names(head(sort(table(extract1), decreasing=T),50)),
  head(sort(table(extract1), decreasing=T),50), 
  scale=c(5, 1.5), 
  random.order=FALSE, 
  random.color=TRUE,
  colors=color, 
  family='AppleGothic')








extract_freq_words(spen) -> extract2
edited2 = head(sort(table(extract2), decreasing=T),7) ; edited2
extract2 <- Filter(function(x){nchar(x)>=2}, extract2)

extract2 = gsub('삼성', '', extract2)
extract2 = gsub('노래실화', '노래', extract2)
extract2 = gsub('노랜', '노래', extract2)
extract2 = gsub('광고', '', extract2)
extract2 = gsub('공식', '', extract2)
extract2 = gsub('그거', '', extract2)
extract2 = gsub('이노래', '노래', extract2)
extract2 = gsub('광고나와', '', extract2)
extract2 = gsub('다시보', '다시봄', extract2)
extract2 = gsub('모두', '', extract2)
extract2 = gsub('에서', '', extract2)
extract2 = gsub('폴드에도', '폴드', extract2)

extract2 = gsub('나와', '', extract2)
extract2 = gsub('다시봄', '', extract2)
extract2 = gsub('폴드에도', '폴드', extract2)
extract2 = gsub('하나', '', extract2)
extract2 = gsub('폴드에도', '폴드', extract2)

extract2 = gsub('언제', '', extract2)
extract2 = gsub('작년', '', extract2)
extract2 = gsub('올해', '', extract2)
extract2 = gsub('마음', '', extract2)
extract2 = gsub('정돈', '', extract2)
extract2 = gsub('언제', '', extract2)

extract2 = gsub('플리에', '', extract2)
extract2 = gsub('최고인데노트는', '노트', extract2)
extract2 = gsub('언제', '', extract2)

extract2 = gsub('이거', '', extract2)

extract2 = gsub('단종되는겁니까', '단종', extract2)
extract2 = gsub('단종하지', '단종', extract2)
extract2 = gsub('이거', '', extract2)

extract2 = gsub('존버는', '존버', extract2)
extract2 = gsub('비교해서', '', extract2)
extract2 = gsub('조은데', '', extract2)
extract2 = gsub('존버는', '존버', extract2)

plot(edited2, family='AppleGothic', main='s-pen', xlab='keyword', ylab='frequency')

library(RColorBrewer)
library(wordcloud)

color <- brewer.pal(12, "Set3")
wordcloud(
  names(head(sort(table(extract2), decreasing=T),50)),
  head(sort(table(extract2), decreasing=T),50), 
  scale=c(5, 1.5), 
  random.order=FALSE, 
  random.color=TRUE,
  colors=color, 
  family='AppleGothic')











#-------여기서부터 해야 한다. 

extract_freq_words(togo) -> extract3
edited3 = head(sort(table(extract3), decreasing=T),8)
edited3

extract3 = gsub('진짜', '', extract3)
extract3 = gsub('갤럭시', '', extract3)
extract3 = gsub('사람을', '', extract3)
extract3 = gsub('삼성', '', extract3)
extract3 = gsub('누구', '', extract3)
extract3 = gsub('사람', '', extract3)
extract3 = gsub('진짜', '', extract3)
extract3 = gsub('지우', '지우개', extract3)
extract3 = gsub('겠습니ㄷ', '', extract3)
extract3 = gsub('ㄷ 음', '', extract3)
extract3 = gsub('좋아보인', '좋아보임', extract3)
extract3 = gsub('닷컴에', '', extract3)
extract3 = gsub('못따라가', '', extract3)
extract3 = gsub('언제', '', extract3)
extract3 = gsub('없던데레드', '레드', extract3)
extract3 = gsub('지우개개', '지우개', extract3)

extract3 <- Filter(function(x){nchar(x)>=2}, extract3)
plot(edited3, family='AppleGothic', main='to go', xlab='keyword', ylab='frequency')

library(RColorBrewer)
library(wordcloud)

color <- brewer.pal(12, "Set3")
wordcloud(
  names(head(sort(table(extract3), decreasing=T),50)),
  head(sort(table(extract3), decreasing=T),50), 
  scale=c(5, 1.5), 
  random.order=FALSE, 
  random.color=TRUE,
  colors=color, 
  family='AppleGothic')


#----------------------------

extract_freq_words(ultra) -> extract4
edited4 = head(sort(table(extract4), decreasing=T),8)
edited4

extract4 = gsub('진짜', '', extract4)
extract4 = gsub('이상한', '', extract4)
extract4 = gsub('언팩보다 ', '', extract4)
extract4 = gsub('이쪽보는게', '', extract4)
extract4 = gsub('좋긴', '', extract4)
extract4 = gsub('광고', '', extract4)
extract4 = gsub('삼성', '', extract4)
extract4 = gsub('실화', '', extract4)
extract4 = gsub('안드로이드', '', extract4)
extract4 = gsub('의미', '', extract4)
extract4 = gsub('이번', '', extract4)
extract4 = gsub('진짜', '', extract4)
extract4 = gsub('진짜', '', extract4)

extract4 = gsub('카피', '', extract4)
extract4 = gsub('장점', '', extract4)

extract4 <- Filter(function(x){nchar(x)>=2}, extract4)
plot(edited4, family='AppleGothic', main='Ultra model', xlab='keyword', ylab='frequency')

library(RColorBrewer)
library(wordcloud)

color <- brewer.pal(12, "Set3")
wordcloud(
  names(head(sort(table(extract4), decreasing=T),50)),
  head(sort(table(extract4), decreasing=T),50), 
  scale=c(5, 1.5), 
  random.order=FALSE, 
  random.color=TRUE,
  colors=color, 
  family='AppleGothic')





#----------------------------------------

extract_freq_words(memory) -> extract5
edited5 = head(sort(table(extract5), decreasing=T),8)
edited5

extract5 = gsub('광고', '', extract5)
extract5 = gsub('달고리즘', '달사진', extract5)
extract5 = gsub('달사진찍으면', '달사진', extract5)
extract5 = gsub('이거', '', extract5)
extract5 = gsub('자기', '', extract5)
extract5 = gsub('엄마', '', extract5)
extract5 = gsub('진짜', '', extract5)
extract5 = gsub('아버지', '', extract5)
extract5 = gsub('유출', '', extract5)
extract5 = gsub('으로', '', extract5)
extract5 = gsub('갤럭시', '', extract5)
extract5 = gsub('서브', '', extract5)
extract5 = gsub('코트', '', extract5)
extract5 = gsub('테니스', '', extract5)
extract5 = gsub('수준', '', extract5)
extract5 = gsub('과장광', '과장광고', extract5)
extract5 = gsub('손등', '', extract5)

extract5 <- Filter(function(x){nchar(x)>=2}, extract5)
plot(edited5, family='AppleGothic', main='memory', xlab='keyword', ylab='frequency')

library(RColorBrewer)
library(wordcloud)

color <- brewer.pal(12, "Set3")
wordcloud(
  names(head(sort(table(extract5), decreasing=T),50)),
  head(sort(table(extract5), decreasing=T),50), 
  scale=c(5, 1.5), 
  random.order=FALSE, 
  random.color=TRUE,
  colors=color, 
  family='AppleGothic')




#---------------------------------------

extract_freq_words(unboxing) -> extract6
edited6 = head(sort(table(extract6), decreasing=T),8)
edited6

extract6 = gsub('광고', '', extract6)
extract6 = gsub('삼성', '', extract6)
extract6 = gsub('갤럭시', '', extract6)
extract6 = gsub('펭수와', '펭수', extract6)
extract6 = gsub('우리', '', extract6)
extract6 = gsub('유느님과', '유재석', extract6)
extract6 = gsub('유재석님', '유재석', extract6)
extract6 = gsub('펭수르', '펭수', extract6)
extract6 = gsub('펭수가', '펭수', extract6)
extract6 = gsub('유느님', '유재석', extract6)
extract6 = gsub('대박나기', '대박', extract6)
extract6 = gsub('기원', '', extract6)
extract6 = gsub('에서도', '', extract6)
extract6 = gsub('펭귄', '펭수', extract6)
extract6 = gsub('펭러', '펭수', extract6)
extract6 = gsub('유튜브', '', extract6)

extract6 <- Filter(function(x){nchar(x)>=2}, extract6)
plot(edited6, family='AppleGothic', main='pengsu yu jae seok unboxing', xlab='keyword', ylab='frequency')

library(RColorBrewer)
library(wordcloud)

color <- brewer.pal(12, "Set3")
wordcloud(
  names(head(sort(table(extract6), decreasing=T),50)),
  head(sort(table(extract6), decreasing=T),50), 
  scale=c(5, 1.5), 
  random.order=FALSE, 
  random.color=TRUE,
  colors=color, 
  family='AppleGothic')


#-------------------------------------------------------

extract_freq_words(jp_ep3) -> extract7
edited7 = head(sort(table(extract7), decreasing=T),8)
edited7

extract7 = gsub('광고', '', extract7)
extract7 = gsub('이거', '', extract7)
extract7 = gsub('갤럭시', '', extract7)
extract7 = gsub('넘재밌어서', '넘재밌음', extract7)
extract7 = gsub('연관된거', '', extract7)
extract7 = gsub('진짜', '', extract7)
extract7 = gsub('유펭듀잇', '', extract7)
extract7 = gsub('유펭두잇', '', extract7)
extract7 = gsub('유느님', '유재석', extract7)
extract7 = gsub('유느님과', '유재석', extract7)
extract7 = gsub('펭수랑', '펭수', extract7)
extract7 = gsub('고네요초대박', '대박', extract7)
extract7 = gsub('봐지네요펭수', '펭수', extract7)
extract7 = gsub('이네요유펭두잇최고', '최고', extract7)
extract7 = gsub('이렇게호감가고', '호감', extract7)
extract7 = gsub('광고를하네', '', extract7)
extract7 = gsub('된거같은데펭수가', '펭수', extract7)
extract7 = gsub('삼성까지우리나라', '', extract7)
extract7 = gsub('유재석과', '유재석', extract7)
extract7 = gsub('넘재밌음', '꿀잼', extract7)
extract7 = gsub('펭수유재석', '등장인물', extract7)
extract7 = gsub('봐도또보고', '반복시청', extract7)

extract7 <- Filter(function(x){nchar(x)>=2}, extract7)

plot(edited7, family='AppleGothic', main='pengsu yu jae seok ep-3', xlab='keyword', ylab='frequency')

library(RColorBrewer)
library(wordcloud)

color <- brewer.pal(12, "Set3")
wordcloud(
  names(head(sort(table(extract7), decreasing=T),50)),
  head(sort(table(extract7), decreasing=T),50), 
  scale=c(5, 1.5), 
  random.order=FALSE, 
  random.color=TRUE,
  colors=color, 
  family='AppleGothic')



#----------------------------------------------------------------여기서부터 하면 된다. 



extract_freq_words(nalchi_unboxing) -> extract8
edited8 = head(sort(table(extract8), decreasing=T),80)



extract_freq_words(panthom) -> extract9
edited9 = head(sort(table(extract9), decreasing=T), 80)


extract_freq_words(epicineveryway) -> extract10
edited10 = head(sort(table(extract10), decreasing=T),80)



extract_freq_words(bang) -> extract11
edited11 = head(sort(table(extract11), decreasing=T),80)



extract_freq_words(nobcut) -> extract12
edited12 = head(sort(table(extract12), decreasing=T),80)



extract_freq_words(meperfect) -> extract13
edited13 = head(sort(table(extract13), decreasing=T),80)
edited13



#--------------------------------------------------------
extract_nouns = function(name, i) {
  a = SimplePos22(name$X.4[i])
  r = melt(a)
  result = str_match(a, '([가-힣]+)/N')
  rr = result[,2]
  nouns = rr[!is.na(rr)]
  if (str_detect(name$X.5[i], '[0-9+K]') == TRUE) {
      el = gsub('\\.', '', name$X.5[i])
      heart = as.integer(gsub('K', '', el))*100
  } else {
      heart = as.integer(name$X.5[i])
  }
  nouns = rep(nouns, heart)
  return(nouns)
}


head(extract_nouns(bang, 1))


#----------------------------------------------------------

extract_freq_words = function(data) {
  noun_list = c()
  for (i in 1:length(data$X.4)) {
    noun_list = c(noun_list, extract_nouns(data, i))
  }
  new_list = c()
  a = 1
  for (i in noun_list) {
    if (nchar(i) == 1) {
      next
    }
    new_list[a] = i
    a = a+1
  }
  return(new_list)
}

#---------------------------------------------

table(extract_freq_words(nalchi))
