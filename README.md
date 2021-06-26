# pythonproject
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk#Importing the natural language toolkit 
import warnings #Subclass of exception
%matplotlib inline 

warnings.filterwarnings('ignore') #This allows you to use known-deprecated code without having to see the warning while not suppressing the warning for other code that might not be aware of its use of deprecated code.



df = pd.read_csv('/content/train_E6oV3lV (2).csv') #Import the CSV file into python using read_csv from pandas
df.head()


# datatype info
df.info()


# removes pattern in the input text
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)# findall() module is used to search for “all” occurrences that match a given pattern
    for word in r:
        input_txt = re.sub(word, "", input_txt)#The re.sub() function is used to replace occurrences of a particular sub-string with another sub-string
    return input_txt
    
    
 #displaying
 df.head() #First five rows of the csv file imported is displayed
    
    
#cleaning
df['clean_message'] = np.vectorize(remove_pattern)(df['message'], "@[\w]*")#The vectorized version of the function takes a sequence of objects or NumPy arrays as input and evaluates the Python function over each element of the input sequence


# remove special characters, numbers and punctuations
df['clean_message'] = df['clean_message'].str.replace("[^a-zA-Z#]", " ")
df.head()


plt.rcParams["figure.figsize"] = [8,10] 
df.label.value_counts().plot(kind='pie', autopct='%1.0f%%')


# remove short words
df['clean_message'] = df['clean_message'].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))
df.head()


# individual words considered as tokens
tokenized_message = df['clean_message'].apply(lambda x: x.split())
tokenized_message.head()


# stem the words
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
stemmer = PorterStemmer()

tokenized_message = tokenized_message.apply(lambda sentence: [stemmer.stem(word) for word in sentence])
tokenized_message.head()


#A list of words to be stemmed
porter=PorterStemmer()
lancaster=LancasterStemmer()
word_list = ["friend", "friendship", "friends", "friendships","stabil","destabilize","misunderstanding","railroad","moonlight","football"]
print("{0:20}{1:20}{2:20}".format("Word","Porter Stemmer","lancaster Stemmer"))
for word in word_list:
    print("{0:20}{1:20}{2:20}".format(word,porter.stem(word),lancaster.stem(word)))
    
    
    # combine words into single sentence
for i in range(len(tokenized_message)):
    tokenized_message[i] = " ".join(tokenized_message[i])
    
df['clean_message'] = tokenized_message
df.head()


# visualize the frequent words
all_words = " ".join([sentence for sentence in df['clean_message']])

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=42,background_color="white_font_size=100).generate(all_words)

# plot the graph
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()




!pip install PILLOW
from PIL import Image
facebook= np.array(Image.open("/content/images.png"))
facebook



def transform_format(val):
    if val == 0:
        return 255
    else:
        return val
        
        
        
  # Create a word cloud image
wc = WordCloud(background_color="white", max_words=1000, mask=transformed_facebook,
                contour_width=3, contour_color='firebrick')

# Generate a wordcloud
wc.generate(all_words)

# store to file
wc.to_file("/content/images.png")

# show
plt.figure(figsize=[20,10])
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()




# frequent words visualization for +ve
all_words = " ".join([sentence for sentence in df['clean_message'][df['label']==0]])

wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

# plot the graph
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()



# frequent words visualization for -ve
all_words = " ".join([sentence for sentence in df['clean_message'][df['label']==1]])

wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

# plot the graph
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()



# extract the hashtag
def hashtag_extract(messages):
    hashtags = []
    # loop words in the messages
    for message in messages:
        ht = re.findall(r"#(\w+)", message)
        hashtags.append(ht)
    return hashtags
    
    
    # extract hashtags from non-racist/sexist tweets
ht_positive = hashtag_extract(df['clean_message'][df['label']==0])

# extract hashtags from racist/sexist tweets
ht_negative = hashtag_extract(df['clean_message'][df['label']==1])



ht_positive[:5]

# unnest list
ht_positive = sum(ht_positive, [])
ht_negative = sum(ht_negative, []

ht_positive[:5]


freq = nltk.FreqDist(ht_positive)
d = pd.DataFrame({'Hashtag': list(freq.keys()),
                 'Count': list(freq.values())})
d.head()




# select top 10 hashtags
d = d.nlargest(columns='Count', n=10)
plt.figure(figsize=(15,9))
sns.barplot(data=d, x='Hashtag', y='Count')
plt.show()




freq = nltk.FreqDist(ht_negative)
d = pd.DataFrame({'Hashtag': list(freq.keys()),
                 'Count': list(freq.values())})
d.head()



# select top 10 hashtags
d = d.nlargest(columns='Count', n=10)
plt.figure(figsize=(15,9))
sns.barplot(data=d, x='Hashtag', y='Count')
plt.show()


# feature extraction
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(df['clean_message'])


# bow[0].toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(bow, df['label'], random_state=42, test_size=0.25)

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=250, random_state=0) 
rf_clf.fit(x_train, y_train) 
y_pred = rf_clf.predict(x_test)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 

print(confusion_matrix(y_test,y_pred)) 
print(classification_report(y_test,y_pred)) 
print(accuracy_score(y_test,y_pred))




from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score


# training
model = LogisticRegression()
model.fit(x_train, y_train)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False
                   
# testing
pred = model.predict(x_test)


accuracy_score(y_test,pred)



# use probability to get output
pred_prob = model.predict_proba(x_test)
pred = pred_prob[:, 1] >= 0.3
pred = pred.astype(np.int)



accuracy_score(y_test,pred)


pred_prob[0][1] >= 0.3



    
    
    
    
