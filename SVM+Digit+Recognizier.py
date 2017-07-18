
# coding: utf-8

# In[46]:


import pandas as pd;
import time;

from sklearn import svm;
from sklearn import model_selection;
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


# In[47]:


#Read data
start_time = time.time();

train_df = pd.read_csv("train.csv")
train_data = trainDataFrame.drop(trainDataFrame.columns[[0]],axis=1);
train_labels = trainDataFrame[['label']].values; 
r,c = train_labels.shape;
train_labels = train_labels.reshape(r,)

if(trainData.shape[0] != trainLabels.shape[0]):
    print "Warning! Train data may be inconsistent. Observations and Labels counts do not match."

test_data = pd.read_csv("test.csv")
    
print "Finished reading data..";
print "Total time taken --- %s seconds ---" % (time.time() - start_time);


# In[ ]:


#PCA grid search to get best number of components
start_time = time.time();
pipe = Pipeline([
    ('reduce_dim', PCA()),
    ('classify', svm.SVC())
])

N_FEATURES_OPTIONS = [8,16,32,46]
C_OPTIONS = [1, 10, 100, 1000]
param_grid = [
    {
        'reduce_dim': [PCA(iterated_power=7)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
]
reducer_labels = ['PCA']

grid = GridSearchCV(pipe, cv=3, n_jobs=2, param_grid=param_grid)
grid.fit(train_data, train_labels)

print "Finished PCA grid search in --- %s seconds---" %()(time.time()-start_time);


# In[ ]:


#Get mean scores from above grid search

mean_scores = np.array(grid.cv_results_['mean_test_score'])
# scores are in the order of param_grid iteration, which is alphabetical
mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
# select score for best C
print mean_scores


# In[37]:


#PCA on dataset
start_time = time.time();
pca = PCA(n_components=16);
train_transformed = pca.fit_transform(train_data);
test_transformed = pca.fit_transform(test_data);

print train_transformed.shape;

print("PCA applied on the dataset in --- %s seconde ---" %(time.time()-start_time));


# In[39]:


#SVM Grid Search 
start_time= time.time()
param_grid = {'C': [1,10,100,1000],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
              'kernel' : ['linear','rbf', 'poly'],
              'degree' :[2,3,4], }

clf = GridSearchCV(svm.SVC(), param_grid)
clf = clf.fit(train_transformed, train_labels.reshape(train_labels.shape[0]))
print(" Grid search done in %s" % (time.time() - start_time))

print("Best estimator found by grid search:")
print(clf.best_estimator_)


# In[ ]:





# In[27]:


#Train SVM model
print "Starting SVM.."
start_time = time.time();

classifier = svm.SVC(kernel='poly', degree =3);

classifier.fit(train_transformed, trainLabels);
print "Finished training"

print "Total time taken ---%s seconds ---"%(time.time()-start_time);


# In[28]:


#Test data 
start_time = time.time();
pred = classifier.predict(test_transformed);

'''
for i in range(0,len(pred)):
    print "%d" %(pred[i]);
    
index = 1;
for i in range(0,len(pred)):
    print "%d\t%d" %(index, pred[i]);
    index +=1; 
print "Total time taken ---%s seconds ---"%(time.time()-start_time);
'''
    


# In[29]:


#Generate output file
outputFile= open('output.csv','w+')
outputFile.write('ImageId,Label\n');

index = 1;
for i in range(0,len(pred)):
    outputFile.write("%d,%d"%(index,pred[i]))
    outputFile.write('\n');
    index +=1;

outputFile.close();

