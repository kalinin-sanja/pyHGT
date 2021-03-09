import numpy as np
import scipy.sparse as sp
import torch

# TODO: изучить эту метрику
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.

# TODO: изучить эту метрику
def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


# rs: список one-hot векторов размерности n_classes, в которых единицей обозначен positive-класс.
def mean_reciprocal_rank(rs):
    # Извлекаем из каждого one-hot векторов индекс элемента, равного 1.
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    # Добавляем к индексу 1, чтобы получить ранг (нулевого места не бывает, отсчёт должен быть с единицы).
    return [1. / (r[0] + 1) if r.size else 0. for r in rs]


def normalize(mx):
    """Row-normalize sparse matrix"""
    # Кол-во единиц в каждой строке матрицы смежности.
    # По сути, это степени вершин.
    rowsum = np.array(mx.sum(1))
    # Получаем обратные значения.
    r_inv = np.power(rowsum, -1).flatten()
    # В случае деления на ноль просто обнуляем коэффициенты.
    r_inv[np.isinf(r_inv)] = 0.
    # Создаём разреженную матрицу размерности [len(r_inv), len(r_inv)],
    # в которой элементы главной диагонали взяты из r_inv.
    # Элементы вне диагонали равны 0.
    r_mat_inv = sp.diags(r_inv)
    # Перемножаем.
    # TODO: матрицы имеют разную размерность! Как происходит умножение?
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def randint():
    return np.random.randint(2**32 - 1)



def feature_OAG(layer_data, graph):
    feature = {}
    times   = {}
    indxs   = {}  # Индексы вершин в исходном графе (graph).
    texts   = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs  = np.array(list(layer_data[_type].keys()))
        tims  = np.array(list(layer_data[_type].values()))[:,1]

        # Вектор признаков для вершины получается путём конкатенации трёх векторов.
        #
        # 1) Получаем эмбеддинги (размерности 400) для Venue/Field/Affiliation вершин.
        #    Подробнее в preprocess_OAG.py
        if 'node_emb' in graph.node_feature[_type]:
            feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, 'node_emb']), dtype=np.float)
        else:
            feature[_type] = np.zeros([len(idxs), 400])

        feature[_type] = np.concatenate((
            feature[_type],
            # 2) Эмбеддинги статей, а точнее их аннотаций (abstracts).
            #    Эмбеддинги остальных вершин вычислялись усреднением эмбеддингов смежных Paper-вершин
            #    (см. preprocess_OAG.py).
            list(graph.node_feature[_type].loc[idxs, 'emb']),
            # 3) Для каждой вершины вычислялось кол-во цитирований (см. preprocess_OAG.py).
            np.log10(np.array(list(graph.node_feature[_type].loc[idxs, 'citation'])).reshape(-1, 1) + 0.01)
        ), axis=1)
        
        times[_type]   = tims
        indxs[_type]   = idxs

        # Если это Paper, то считываем ещё заголовки статей.
        if _type == 'paper':
            texts = np.array(list(graph.node_feature[_type].loc[idxs, 'title']), dtype=np.str)

    return feature, times, indxs, texts
