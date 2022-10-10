import pandas as pd
import numpy as np

# preprocessing dataset -RS
col_list = ["type", "rating", "title"]
dataframe = pd.read_csv("/Users/raymond/Desktop/Capstone Project/netflix_titles.csv", usecols=col_list)
ratings = ['TV-MA', 'TV-14', 'TV-PG', 'R', 'PG-13', 'TV-Y']
df = dataframe[dataframe['rating'].isin(ratings)]
# print(df.head())



print(df['type'].value_counts())

print(df['rating'].value_counts())

# !pip install neattext

import neattext.functions as nfx

# preprocess the NLP DATA -RS
# df['title'] = df['title'].str.lower()
df['title'] = df['title'].apply(lambda x: x.lower())
df['title'] = df['title'].apply(nfx.remove_stopwords)
df['title'] = df['title'].apply(nfx.remove_puncts)
df['title'] = df['title'].apply(nfx.remove_special_characters)
df['title'] = df['title'].apply(nfx.remove_emojis)

df.head()

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier

Xfeatures = df['title']
ylabels = df[['type','rating']]

x_train,x_test,y_train,y_test = train_test_split(Xfeatures,ylabels,test_size=0.3,random_state=7)

from sklearn.pipeline import Pipeline

pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),
                          ('lr_multi',MultiOutputClassifier(LogisticRegression(solver='lbfgs', max_iter=200)))])

pipe_lr.fit(x_train,y_train)


pipe_lr.score(x_test,y_test)
# print(pipe_lr.score(x_test,y_test))

typeres = []
ratingres = []
typetest = []
ratingtest = []
for i in range(len(x_test)):
    x = pipe_lr.predict([x_test.iloc[i]])
    typeres.append(x[0,0])
    typetest.append(y_test['type'].iloc[i])
    ratingres.append(x[0,1])
    ratingtest.append(y_test['rating'].iloc[i])



from sklearn.metrics import confusion_matrix
#Get the confusion matrix
cf_matrix = confusion_matrix(typeres, typetest)
print(cf_matrix)

cf_matrix2 = confusion_matrix(ratingres, ratingtest)
print(cf_matrix2)


# Adding Confusion Matrix visualization -RS
import seaborn as sns
import matplotlib.pyplot as plt

# # type confusion matrix -RS
# ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
# ax.set_title('Type Confusion Matrix\n\n');
# ax.set_xlabel('\nPredicted Type Category')
# ax.set_ylabel('Actual Type Category ');
# ## Ticket labels - List must be in alphabetical order
# ax.xaxis.set_ticklabels(['Movie','TV-Show'])
# ax.yaxis.set_ticklabels(['Movie','TV-Show'])
# ## Display the visualization of the Confusion Matrix.
# plt.show()

# rating confusion matrix -RS
ax = sns.heatmap(cf_matrix2, annot=True, cmap='Blues')
ax.set_title('Rating Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Rating Category')
ax.set_ylabel('Actual Rating Category ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['TV-MA', 'TV-14', 'TV-PG', 'R', 'PG-13', 'TV-Y'])
ax.yaxis.set_ticklabels(['TV-MA', 'TV-14', 'TV-PG', 'R', 'PG-13', 'TV-Y'])
## Display the visualization of the Confusion Matrix.
plt.show()