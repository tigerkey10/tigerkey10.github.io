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

extract_nouns = function(i) {
  a = SimplePos22(nalchi$X.4[i])
  r = melt(a)
  result = str_match(a, '([가-힣]+)/N')
  rr = result[,2]
  nouns = rr[!is.na(rr)]
  if (str_detect(nalchi$X.5[i], '[0-9+K]') == TRUE) {
      el = gsub('\\.', '', nalchi$X.5[i])
      heart = as.integer(gsub('K', '', el))*100
  } else {
      heart = as.integer(nalchi$X.5[i])
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

extract_freq_words(nalchi$X.4) -> extract
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


extract_freq_words(views$X.4) -> extract1
edited1 = head(sort(table(extract1), decreasing=T),80)

extract_freq_words(spen$X.4) -> extract2
edited2 = head(sort(table(extract2), decreasing=T),80)

extract_freq_words(togo$X.4) -> extract3
edited3 = head(sort(table(extract), decreasing=T),80)

extract_freq_words(ultra$X.4) -> extract4
edited4 = head(sort(table(extract4), decreasing=T),80)

extract_freq_words(memory$X.4) -> extract5
edited5 = head(sort(table(extract5), decreasing=T),80)

extract_freq_words(unboxing$X.4) -> extract6
edited6 = head(sort(table(extract6), decreasing=T),80)

extract_freq_words(jp_ep3$X.4) -> extract7
edited7 = head(sort(table(extract7), decreasing=T),80)

extract_freq_words(nalchi_unboxing$X.4) -> extract8
edited8 = head(sort(table(extract8), decreasing=T),80)

extract_freq_words(panthom$X.4) -> extract9
edited9 = head(sort(table(extract9), decreasing=T),80)

extract_freq_words(epicineveryway$X.4) -> extract10
edited10 = head(sort(table(extract10), decreasing=T),80)

extract_freq_words(bang$X.4) -> extract11
edited11 = head(sort(table(extract11), decreasing=T),80)

extract_freq_words(nobcut$X.4) -> extract12
edited12 = head(sort(table(extract12), decreasing=T),80)

extract_freq_words(meperfect$X.4) -> extract13
edited13 = head(sort(table(extract13), decreasing=T),80)

extract_freq_words(meperfect$X.4)
