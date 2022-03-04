# Meters Object Detection
Вычислаются метрики F1-score, Recall, Precision, mAP, AP.
По файлам с детекциями и аннотациями для каждого класса строятся кривые зависимости Precision(Recall), F1-score(Threadshold), AUC-curve.
Значения метрик могут выводиться в терминал, сохряняться в файл .xlsx вместе с рисунками.

# Requirements and environment

### Linux/MacOS
Создайте виртуальное окружение python:
```console
virtualenv name_of_venv
```
Установите требуемые пакеты:
```console
pip3 install -r requirements.txt
```

# Run service
Запуск расчета метрик:
```console
python3 main.py
```
