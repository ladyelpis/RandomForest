import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import Data_class as dc
import gensim
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix 

#Чтение данных
pd.set_option('display.max_colwidth', 100)

x_train, y_train = dc.Data_class.GetData('insults_train.jsonl')
x_test, y_test = dc.Data_class.GetData('insults_test.jsonl')

#Тренируем the word2vec model
w2v_model = gensim.models.Word2Vec(x_train,
                                   vector_size=100,
                                   window=5,
                                   min_count=2)

w2v_model.wv.index_to_key

#Ищем самые близкие слова к заданному слову из нашей модели
w2v_model.wv.most_similar('проститутка')

words = set(w2v_model.wv.index_to_key )
x_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in x_train])
x_test_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in x_test])

#Длина предложения отличается от длины вектора предложения
for i, v in enumerate(x_train_vect):
    print(len(x_train.iloc[i]), len(v))

#Вычисление векторов предложений путем усреднения векторов слов для слов, содержащихся в предложении
x_train_vect_avg = []
for v in x_train_vect:
    if v.size:
        x_train_vect_avg.append(v.mean(axis=0))
    else:
        x_train_vect_avg.append(np.zeros(100, dtype=float))
        
x_test_vect_avg = []
for v in x_test_vect:
    if v.size:
        x_test_vect_avg.append(v.mean(axis=0))
    else:
        x_test_vect_avg.append(np.zeros(100, dtype=float))

#Согласование длины векторов предложений
for i, v in enumerate(x_train_vect_avg):
    print(len(x_train.iloc[i]), len(v))

######## Прогулка по случайному лесу ######## 
#Создание и тренировка модели случайного леса
print("******************************************")
print("**************Случайный лес***************")
print("******************************************")
rf = RandomForestClassifier()
rf_model = rf.fit(x_train_vect_avg, y_train.values.ravel())

#Использование тренированной модели для создания предсказаний на основе тестовых данных 
y_pred = rf_model.predict(x_test_vect_avg)

from sklearn.metrics import precision_score, recall_score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
#Точность, Полнота, Вероятность
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round((y_pred==y_test).sum()/len(y_pred), 3)))
print("******************************************")
######## Логистическая регрессия ######## 
print("           ／＞　 フ")
print("　　　　　| 　_　 _|")
print("　 　　　／`ミ _x 彡")
print("　　 　 /　　　   |")
print("　　　 /　 ヽ　　 ﾉ")
print("　／￣|　　 |　|　|")
print("　| (￣ヽ＿_ヽ_)_)")
print("　＼二つ")

#Тренировка модели Логистической регрессии
classifier = LogisticRegression(random_state = 0) 
classifier.fit(x_train_vect_avg, y_train)
#Использование тренированной модели для создания предсказаний на основе тестовых данных 
y_pred = classifier.predict(x_test_vect_avg)

cm = confusion_matrix(y_test, y_pred)
print("******************************************")
print("*********Логистическая регрессия**********")
print("******************************************")
#Матрица ошибок (матрица неточностей) 
print ("Confusion Matrix : \n", cm)
#Вероятность
from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(y_test, y_pred))
print("******************************************")

