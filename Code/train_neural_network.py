import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.svm import SVC  
from names_vector import *
import numpy as np


f_out=open('output.txt');
f_out=f_out.read();
f_out=f_out.split('\n');
f_inp=open('input.txt');
f_inp=f_inp.read();
f_inp=f_inp.split('\n');



i=0;
l=len(f_inp);
vec_vect=[];
y_t=[];
while i<l:
    print f_inp[i];
    temp=glove_word2vec_vec(f_inp[i]);
    k=len(temp);
    if k==0:
        print "not found";
    else:
        y_t.append(int(f_out[i]));
        vec_vect.append(temp);
    i=i+1;
    print str(i)+" "+str(l);
    #print temp;
print "Input done";
f_inp_test=open('input_test.txt');
f_inp_test=f_inp_test.read();
f_inp_test=f_inp_test.split('\n');
i=0;
l=len(f_inp_test);
vec_vect_test=[];
while i<l:
    #print f_inp_test[i];
    temp=glove_word2vec_vec(f_inp_test[i]);
    k=len(temp);
    if k==0:
        print "not found";
    else:
        vec_vect_test.append(temp);
    i=i+1;
    #print temp;
x_train=np.array(vec_vect);
y_train=np.array(y_t);
x_test=np.array(vec_vect_test);
# print x_train;
# print y_train;

print "neural starts";
classifier = Sequential()
classifier.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu', input_dim = 300))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 32, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print "compiling done";

classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)
print "training done";
y_pred = classifier.predict(x_test)
print "predicting done";
y_pred = (y_pred > 0.5)

# svclassifier = SVC(kernel='rbf')  
# svclassifier.fit(x_train, y_train)  

# y_pred = svclassifier.predict(x_test)  
#print y_pred;
l=len(f_inp_test);
f_out_test=open('output_test_neural.txt',"w");
i=0;
cnt=0;
while i<l:
    temp=glove_word2vec_vec(f_inp_test[i]);
    k=len(temp);
    if k==0:
        print "not found";
        st=str(-1);
    else:
        st=str(y_pred[cnt]);
        cnt=cnt+1;
    if i!=l-1:  
        f_out_test.write(st+'\n');
    else:
        f_out_test.write(st);
    i=i+1;













