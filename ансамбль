from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Загрузка набора данных об ирисах
iris = load_iris()
X = iris.data
y = iris.target

# Разделяем данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем модель случайного леса
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Обучаем модель на обучающем наборе данных
model.fit(X_train, y_train)

# Делаем предсказания на тестовом наборе данных
y_pred = model.predict(X_test)

# Оцениваем точность модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy * 100:.2f}%")
