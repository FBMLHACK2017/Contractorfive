import nltk
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier            


    


## random forest model ##
data=pd.read_csv('/Users/jianjiey/Documents/countTB.csv',header=0,index_col=0)
dt=np.array(data)
pTag=pd.read_csv('/Users/jianjiey/pTag.csv',header=0,index_col=0)
pTag=np.array(pTag)

keyword=pd.read_csv('/Users/jianjiey/keywords.csv',header=0,index_col=0)
kw=np.array(keyword)

temporal=pd.read_csv('/Users/jianjiey/Documents/temporal_feature.csv',header=0,index_col=0)
temporal=np.array(temporal)

zipcode=pd.read_csv('/Users/jianjiey/Documents/zipcode_feature.csv',header=0,index_col=0)
zipcode=np.array(zipcode)


other=pd.read_csv('/Users/jianjiey/Documents/other_feature.csv',header=0,index_col=0)
other=np.array(other)


import random
feature=np.concatenate([dt,temporal,zipcode,other,lda_fe],axis=1)
#feature=np.concatenate([dt,lda_fe],axis=1)
arr = np.arange(len(pTag))
np.random.shuffle(arr)
fold=np.array(range(5)*11320)

score=[]
for i in range(5):
    train_fold=arr[np.where(fold!=i)]
    x_train=np.array(feature)[train_fold,]
    pTag_train = np.array(pTag)[train_fold]
    pv_sl=np.where(pv<0.1)[0]
    rf_model.fit(np.nan_to_num(x_train[:,pv_sl]),pTag_train.flatten())
    

    pTag_test = np.array(pTag)[[i for i in range(len(pTag))  if i not in train_fold]]

    x_test = np.array(feature)[[i for i in range(len(pTag))  if i not in train_fold],]
    y_pred_p = rf_model.predict(np.nan_to_num(x_test[:,pv_sl]))
    score.append(mean(np.sqrt(np.square(y_pred_p-pTag_test))))

mean(score)




#### variable selection ####
v = np.array(pTag).T
x=feature
pv=np.zeros(x.shape[1])
for i in range(x.shape[1]):
    z = np.array(x[:,i]).flatten()
    slope, intercept, r_value, p_value, std_err = stats.linregress(z,v)
    pv[i] = p_value


##varaible selection by RF importance ##
var_rf=np.argsort(rf_model.feature_importances_)[::-1][0:1000]

####LDA ####

import lda
import lda.datasets

##A GOOD SETTING ##n_topics=20;n_top_words = 12
## exclude  low frequency and high frequency word
model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
model.fit(dt)
topic_word = model.topic_word_ 
lda_fe=dt.dot(topic_word.T) 



from sklearn.ensemble import RandomForestRegressor            
rf_model = RandomForestRegressor(bootstrap=0.4,
             max_depth=None, max_features=20,
            max_leaf_nodes=None, min_samples_leaf=4,
            min_samples_split=4, n_estimators=2000, n_jobs=-1,
            oob_score=False, random_state=None, verbose=0)   

rf_model.fit(np.nan_to_num(x[:,pv_sl]),pTag.flatten())





import pickle

with open('/Users/jianjiey/Documents/rf2_model.dat', 'wb') as f:
    pickle.dump(rf_model, f)
    
np.save('/Users/jianjiey/Documents/topic_word.npy',topic_word)


np.save('/Users/jianjiey/Documents/pv.npy',np.where(pv<0.01)[0])
