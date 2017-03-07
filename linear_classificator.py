from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from loader import load_no_show_issue

x_appointment, y_appointment, x_names, y_names = load_no_show_issue()

x_train, x_test, y_train, y_test = train_test_split(x_appointment, y_appointment, test_size=0.25)

# normalizar
normalizador = StandardScaler().fit(x_train)
xn_train = normalizador.transform(x_train)

clasificador = SGDClassifier()
# entrenar
clasificador.fit(xn_train, y_train)


xn_test = normalizador.transform(x_test)
y_test_pred = clasificador.predict(xn_test)
exactitud = metrics.accuracy_score(y_test, y_test_pred)


print('Exactitud: {}'.format(exactitud))
print(metrics.classification_report(y_test, y_test_pred))
print('Matriz de confusión\n{}'.format(metrics.confusion_matrix(y_test, y_test_pred)))
