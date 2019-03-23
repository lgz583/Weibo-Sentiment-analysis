import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

df_news = pd.read_csv('./data/data.csv',names=['data','label'],encoding='utf-8')
df_news = df_news.dropna()
content = df_news.data.values.tolist()
content_S = []
for line in content:
    current_segment = jieba.lcut(line)
    if len(current_segment) >= 1 and current_segment != '\r\n': #换行符
        content_S.append(current_segment)

df_content=pd.DataFrame({'data_S':content_S})
stopwords=pd.read_csv("stopwords.txt",index_col=False,sep="\n",quoting=3,names=['stopword'], encoding='utf-8')

def drop_stopwords(contents,stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
            
        if line_clean is None:
            line_clean.append("空")
        contents_clean.append(line_clean)
    return contents_clean,all_words
    #print (contents_clean)
        

contents = df_content.data_S.values.tolist()    
stopwords = stopwords.stopword.values.tolist()
data_clean,all_words = drop_stopwords(contents,stopwords)
df_content=pd.DataFrame({'data_clean':data_clean})
df_train=pd.DataFrame({'data_clean':data_clean,'label':df_news['label']})
label_mapping = {"none": 1, "like": 2, "disgust": 3, "anger": 4, "happiness":5, "fear": 6,"sadness": 7,"surprise": 8}
df_train['label'] = df_train['label'].map(label_mapping)

x_train = df_train['data_clean']
y_train = df_train['label']

words = []
for line_index in range(len(x_train)):
    try:
        #x_train[line_index][word_index] = str(x_train[line_index][word_index])
        words.append(' '.join(x_train[line_index]))
    except:
        print (line_index)
print (len(words))
print(words[0])
vec = CountVectorizer(analyzer='word', max_features=4000, lowercase = False)
countresult = vec.fit(words)
x = vec.transform(words)
print(x.shape)
classifier = MultinomialNB()
result = classifier.fit(x, y_train)
joblib.dump(countresult,'countresult.model')
joblib.dump(result,'result.model')
    
   

