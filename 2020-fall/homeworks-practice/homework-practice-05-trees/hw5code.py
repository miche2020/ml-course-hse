import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    feature_vector = np.array(feature_vector)
    target_vector = np. array(target_vector)
    n = len(feature_vector) # Кол-во объектов
    sort_ind = np.argsort(feature_vector) # Отсортированные индексы по возрастанию значения
    
    f_column_s = feature_vector[sort_ind] # Отсортированная колонка
    all_col = f_column_s[: -1] + f_column_s[1: ] # сложение 2 array для последующего нахождения среднего
    thresh, thresh_ind = np.unique(all_col, return_index=True) # получение комбинации порогов, нужно треш поделить на 2
    thresh_ind = (- thresh_ind + (n - 2))[::-1]
    thresh_ind = thresh_ind[f_column_s[thresh_ind]!= f_column_s[thresh_ind+1]]
    thresh = all_col[thresh_ind]
    thresh = thresh / 2
     
    t_columns_s = target_vector[sort_ind] # сортировка целевого столбца
    count_1_l = np.cumsum(t_columns_s)[: -1] # 257
    count_0_l = np.cumsum(np.ones(n) - t_columns_s)[: -1] #257
    count_1_r = np.sum(t_columns_s) - count_1_l # 257
    count_0_r = n - np.sum(t_columns_s) - count_0_l # 257
    
    R_l_size = np.arange(1, n)
    R_r_size = n - np.arange(1, n)
    
    H_R_l = np.ones(n - 1) - (count_1_l / R_l_size) ** 2 - (count_0_l / R_l_size) ** 2
    H_R_r = np.ones(n - 1) - (count_1_r / R_r_size) ** 2 - (count_0_r / R_r_size) ** 2
    
    gini = - R_l_size / n * H_R_l - R_r_size / n * H_R_r 
    gini = gini[thresh_ind]
    best = np.argmax(gini)
    thresh_best = thresh[best]
    gini_best = gini[best]
    

    return (thresh, gini, thresh_best,gini_best)
    
    

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type") # проверка на типы признаков

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y != sub_y[0]): # Критерия остановы
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]): 
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                feature_vector = np.array(map(lambda x: categories_map[x], sub_X[:, feature]))

            else:
                raise ValueError

            if len(np.unique(feature_vector)) == 1: #было 3if (feature_vector < threshold).sum() == sub_X.shape[0] or (feature_vector < threshold).sum() == 0:
                    continue
                

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:

        
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError
            

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[split], node["right_child"])

    def _predict_node(self, x, node):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
        if node['type'] == 'terminal':
            return node['class']
        
        feature = node['feature_split']
        if self._feature_types[feature] == 'real':
            direction = "left_child" if x[feature] < node["threshold"] else "right_child"
        else:
            direction = "left_child" if x[feature] in node["categories_split"] else "right_child"
        return self._predict_node(x, node[direction])
        

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)