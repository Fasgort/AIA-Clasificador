import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from scipy.sparse import csc


from loader import filter_features, load_no_show_issue, load_no_show_issue_cache

#x_appointment, y_appointment, x_names, y_names = load_no_show_issue()

x_appointment, y_appointment, x_names, y_names = load_no_show_issue_cache()

features = ['Age', 'Gender', 'DayOfTheYe', 'DayOfTheWe', 'Diabetes', 'Alcoolism', 'HiperTensi', 'Handcap', 'Smokes', 'Scholarshi', 'Tuberculos', 'Sms_Remind', 'AwaitingTi']


x_data_filtered, x_names_filtered = filter_features(x_appointment, x_names, features)
x_train, x_test, y_train, y_test = train_test_split(x_data_filtered, y_appointment, test_size=0.25)

clasificador = DecisionTreeClassifier(criterion='entropy', max_depth=40, min_samples_leaf=10)
# entrenar
clasificador.fit(x_train, y_train)

y_test_pred = clasificador.predict(x_test)
exactitud = metrics.accuracy_score(y_test, y_test_pred)

print('Exactitud: {}'.format(exactitud))
print(metrics.classification_report(y_test, y_test_pred))
print('Matriz de confusión\n{}'.format(metrics.confusion_matrix(y_test, y_test_pred)))
export_graphviz(clasificador, feature_names=x_names_filtered,out_file="0all.dot")

modelo = Pipeline([('modelodecisiontree', DecisionTreeClassifier())])
modelo.fit(x_train, y_train)
modelo.predict(x_test)

kfold5 = KFold(x_data_filtered.shape[0], 5, shuffle=True)
valores = cross_val_score(modelo, x_data_filtered, y_appointment, cv=kfold5)
media_validacion = np.mean(valores)
np.std(valores)
print("Validación del modelo: {}".format(media_validacion))