import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn import svm,metrics


pkl = open('mlmodel.pickle', 'rb')
clf = pickle.load(pkl)   
vec = open('vectorizer.pickle', 'rb')
tf_vect = pickle.load(vec)

def test_string(s):
    X_test_tf = tf_vect.transform([s])
    y_predict = clf.predict(X_test_tf)
    return y_predict

output = test_string(sys.argv[1])[0]
outputstring= sys.argv[1]

print(output)

def fun():
    file=open("file1.txt","a")
    file.write(outputstring)
    file.write(" The Review is  ")
    file.write(output)
    file.write("\n \n")
    file.close()

fun()










