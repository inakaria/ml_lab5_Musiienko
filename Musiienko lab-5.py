import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import NearestCentroid


print('1. Відкрити та зчитати файл з даними.')
data = pd.read_csv('dataset2_l4.txt', sep=',')

print(f'2. Визначити та вивести кількість записів: {data.shape[0]}')

print('3. Видалити атрибут Class.')
data.drop(columns=['Class'], inplace=True)

print(f'4. Вивести атрибути, що залишилися: {",".join(data.columns)}')

print('5. Використовуючи функцію KMeans бібліотеки scikit-learn, виконати розбиття', 
      'набору даних на кластери з випадковою початковою ініціалізацією і вивести координати центрів кластерів.', 
      'Оптимальну кількість кластерів визначити на основі початкового набору даних трьома різними способами.')
print('\n5.1. elbow method',
      '\n5.2 average silhouette method',
      '\n5.3 prediction strength method')
