from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import numpy as np
import os
from bz2 import BZ2File
import json
import sys 
import os.path




   #getting the dataset and cleaning data

def getValues():
    negReviews = list()
    posReviews=list()
    count = 0 
    reviews = []
    target =[]
    #data cleaning and getting data. 
    path_neg = 'review_polarity/txt_sentoken/neg/'
    path_pos = 'review_polarity/txt_sentoken/pos/'
    print(os.path.dirname(__file__))
    relative_path_neg = os.path.join(os.path.dirname(__file__), path_neg)
    print(relative_path_neg)
    relative_path_pos = os.path.join(os.path.dirname(__file__), path_pos)
    for i in os.listdir(relative_path_neg):
    #getting training set from files. 
        if i.endswith(".txt"):
          file = open(relative_path_neg+i, 'r')
          line = file.read()
          negReviews.append(line)
          file.close()
          continue
        else:
           continue

    for i in os.listdir(relative_path_pos):
     if i.endswith(".txt"):
         ##print i
         file = open(relative_path_pos+i, 'r')
         line = file.read()
         posReviews.append(line)
         file.close()
         continue
     else:
           continue

    test_pos = posReviews[len(posReviews)-10: len(posReviews)]
    test_neg = negReviews[len(negReviews)-10: len(negReviews)]
    posReviews = posReviews[:len(posReviews)-10] 
    negReviews = negReviews[:len(negReviews)-10]

    posReviewsTarget = [0] *len(posReviews) 
    negReviewsTarget = [1]*len(negReviews)

    testTargetPos = [0] *len(test_pos) 
    testTargetNeg = [1]*len(test_neg)

    reviews = posReviews+negReviews 
    target = posReviewsTarget+ negReviewsTarget
    test = test_pos+test_neg
    testTarget = np.array(testTargetPos+testTargetNeg) #should now be same size. 
    print("Returning")
    return reviews, target, test, testTarget




   
#analysng data, predicting if positive or negative review. 

[reviews, target, test, testTarget] = getValues()
tree_clf = Pipeline([('vect', CountVectorizer(decode_error ='ignore', tokenizer=lambda doc: doc, lowercase=False)),
                    ('clf', RandomForestClassifier()), 
    ])

tree_clf = tree_clf.fit(reviews, target)
prediction= tree_clf.predict(test) 
print (metrics.classification_report(testTarget, prediction) )


    






