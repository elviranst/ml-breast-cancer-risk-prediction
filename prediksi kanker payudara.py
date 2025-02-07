#RANDOM FOREST
import pandas as pd
data = pd.read_excel("dataR2.xlsx")
print(data.head())

#Menghitung nilai Classification untuk melihat jumlah negatif=1 dan positif=2
sizes = data['Classification'].value_counts(sort = 1)
print(sizes)

#Siapkan Data X dan Y
Y = data["Classification"].values  
X = data.drop(labels = ["Classification"], axis=1)  
print(X.head())

#Bagi data menjadi data TRAIN dan TEST 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)
print(X_train)

#Definisikan model dan data training
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 26, random_state = 30)
# train model dari data training
model.fit(X_train, y_train)

#Model test dengan prediksi dari data test dan hitung skor akurasi
prediction_test = model.predict(X_test)
print(y_test, prediction_test)

from sklearn import metrics
print ("Accuracy(%) = ", metrics.accuracy_score(y_test, prediction_test)*100)

#Runtime
import timeit
print( "Program Execution Run Time :", timeit.timeit('''input_list=range(100)
def runtime(num):
    if num %5 ==0:
        return True
    else :
        return False
        
run_time = (i for i in input_list if runtime(i))

for i in run_time:
    x = i''', number= 10000))
    
# MODEL FEATURE
importances = list(model.feature_importances_)
feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp) 
#DECISION TREE
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

# Masukkan Data
bc = pd.read_excel("Breast Cancer.xlsx")
print(bc.head())

#Bagi dataset menjadi variabel bebas dan terikat
feature_cols = ['Usia', 'IMT', 'Glukosa', 'Resistin']
X = bc.drop(labels = ["Classification"], axis=1) 
print(X)
y = bc["Classification"].values 
print(y)

#Bagi data menjadi data test dan train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Buat Objek DT
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train DT
clf = clf.fit(X_train,y_train)

#prediksi respon data set
y_pred = clf.predict(X_test)

# Akurasi Model
print("Accuracy(%) :",metrics.accuracy_score(y_test, y_pred)*100)

#Visualisasi DT
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['Negative','Positive'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('breast_cancer.png')
Image(graph.create_png())

#Runtime
import timeit
print( "Program Execution Run Time :", timeit.timeit('''input_list=range(100)
def runtime(num):
    if num %5 ==0:
        return True
    else :
        return False
        
run_time = (i for i in input_list if runtime(i))

for i in run_time:
    x = i''', number= 10000))
