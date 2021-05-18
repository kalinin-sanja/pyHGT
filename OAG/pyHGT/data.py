import json, os
import math, copy, time
import numpy as np
from collections import defaultdict
import pandas as pd
from .utils import *

import math
from tqdm import tqdm

import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import dill
from functools import partial
import multiprocessing as mp

class Graph():
    """
    Структура графа. Используем постфикс "gid" для того,
    чтобы указать уникальный идентификатор вершины в этой структуре.
    При этом, вершины разного типа имеют свою нумерацию, поэтому их индексы могут совпадать.
    """
    def __init__(self):
        super(Graph, self).__init__()
        '''
            node_forward and node_backward are only used when building the data. 
            Afterwards will be transformed into node_feature by DataFrame
            
            node_forward: name -> node_gid
            node_backward: node_id -> feature_dict
            node_feature: a DataFrame containing all features
        '''
        self.node_forward = defaultdict(lambda: {})  # [node_type][node_id] = node_gid
        self.node_backward = defaultdict(lambda: [])  # [node_type] = list of node objects
        # Задаётся извне класса, при считывании данных в preprocess_OAG.py.
        self.node_feature = defaultdict(lambda: [])

        '''
            edge_list: index the adjacancy matrix (time) by 
            <target_type, source_type, relation_type, target_id, source_id>
        '''
        self.edge_list = defaultdict( #target_type
                            lambda: defaultdict(  #source_type
                                lambda: defaultdict(  #relation_type
                                    lambda: defaultdict(  #target_gid
                                        lambda: defaultdict( #source_gid(
                                            lambda: int # time
                                        )))))
        self.times = {}

    # Возвращает индекс вершины (gid).
    # У каждого типа вершин своя нумерация!
    def add_node(self, node):
        node_id = node['id']
        node_type = node['type']
        id2idx = self.node_forward[node_type]

        if node_id not in id2idx:
            # Позиция вершины в массиве self.node_backward[node_type] есть её индекс node_gid.
            self.node_backward[node_type] += [node]
            node_gid = len(id2idx)
            id2idx[node_id] = node_gid
            return node_gid
        
        return id2idx[node_id]

    def add_edge(self, source_node, target_node, time = None, relation_type = None, directed = True):
        edge = [self.add_node(source_node), self.add_node(target_node)]
        '''
            Add bi-directional edges with different relation type
        '''
        self.edge_list[target_node['type']][source_node['type']][relation_type][edge[1]][edge[0]] = time
        if directed:
            # Если граф ориентированный, рассматриваем реверсию как отдельное отношение.
            self.edge_list[source_node['type']][target_node['type']]['rev_' + relation_type][edge[0]][edge[1]] = time
        else:
            # Если граф неориентированный, то рассматриваем одно отношение, просто меняем source и target местами.
            self.edge_list[source_node['type']][target_node['type']][relation_type][edge[0]][edge[1]] = time

        self.times[time] = True
        
    def update_node(self, node):
        nodes = self.node_backward[node['type']]
        node_gid = self.add_node(node)
        # Похоже, что это просто считывание атрибутов вершины.
        # Однако, `node` и `nodes[node_gid]` должны указывать на один объект.
        # Есть ли смысл в таких заморочках?
        for k in node:
            if k not in nodes[node_gid]:
                nodes[node_gid][k] = node[k]

    # Список кортежей - мета-триплетов.
    def get_meta_graph(self):
        types = self.get_types()
        metas = []
        for target_type in self.edge_list:
            for source_type in self.edge_list[target_type]:
                for r_type in self.edge_list[target_type][source_type]:
                    metas += [(target_type, source_type, r_type)]
        return metas
    
    def get_types(self):
        return list(self.node_feature.keys())



def sample_subgraph(graph, time_range, sampled_depth = 2, sampled_number = 8, inp = None, feature_extractor = feature_OAG):
    '''
        Sample Sub-Graph based on the connection of other nodes with currently sampled nodes
        We maintain budgets for each node type, indexed by <node_id, time>.
        Currently sampled nodes are stored in layer_data.
        After nodes are sampled, we construct the sampled adjacancy matrix.
    '''
    layer_data  = defaultdict( #target_type
                        lambda: {} # {target_id: [ser, time]}
                    )
    budget     = defaultdict( #source_type
                                    lambda: defaultdict(  #source_id
                                        lambda: [0., 0] #[sampled_score, time]
                            ))
    # TODO: неиспользуемая переменная
    new_layer_adj  = defaultdict( #target_type
                                    lambda: defaultdict(  #source_type
                                        lambda: defaultdict(  #relation_type
                                            lambda: [] #[target_id, source_id]
                                )))
    # TODO: Текст не дописан!
    '''
        For each node being sampled, we find out all its neighborhood, 
        adding the degree count of these nodes in the budget.
        Note that there exist some nodes that have many neighborhoods
        (such as fields, venues), for those case, we only consider 
    '''
    def add_budget(te, target_gid, target_time, layer_data, budget):
        # te = graph.edge_list[target_type]
        # Перечисляем триплеты, которые содержат target_gid (именно на позиции target-вершины).
        for source_type in te:
            tes = te[source_type]
            for relation_type in tes:
                if relation_type == 'self' or target_gid not in tes[relation_type]:
                    continue

                # Список source-вершин для данной target-вершины.
                adl = tes[relation_type][target_gid]
                # Выбираем из них только sampled_number вершин.
                if len(adl) < sampled_number:
                    sampled_gids = list(adl.keys())
                else:
                    sampled_gids = np.random.choice(list(adl.keys()), sampled_number, replace = False)

                # Перебираем их.
                for source_gid in sampled_gids:
                    source_time = adl[source_gid]
                    if source_time == None:
                        source_time = target_time

                    # Если вершина уже используется как target среди текущих, то не рассматриваем её.
                    if (source_time > np.max(list(time_range.keys()))) or (source_gid in layer_data[source_type]):
                        continue

                    # Степень вершины считают как len(sampled_gids) ???
                    # Кумулятивная нормализованная степень.
                    # `budget` не разделяется на отношения (relation),
                    # но для каждого отношения нормазилованная степень считается отдельно.
                    # Больше похоже на вероятность перехода в эту вершину из target-вершины.
                    budget[source_type][source_gid][0] += 1. / len(sampled_gids)
                    budget[source_type][source_gid][1] = source_time

    '''
        First adding the sampled nodes then updating budget.
    '''
    # Инициализация `layer_data`.
    # Изначально `layer_data` содержит seed-target вершины.
    for node_type in inp.keys():
        for node_gid, _time in inp[node_type]:
            # Индексируем вершины.
            # _time - время триплета, в котором данная вершина является target.
            # Получается, что, если вершина встречалась как target в нескольких триплетах,
            # то время возьмётся только с последнего, а также перезапишется индекс!
            # Допустим, сначала встрели пару [5, time1], записали её как [0, time1].
            # Затем встретили пару [5, time2], записали её как [1, time2].
            # Т.е., мы потеряем значение, которое хранилось под индексом 0!
            # НО! Если мы рассматриваем связь Venue-Paper, то paper не должен встречаться больше одного раза.
            layer_data[node_type][node_gid] = [len(layer_data[node_type]), _time]

    # Выбираем вершины
    for node_type in inp.keys():
        te = graph.edge_list[node_type]
        # Перебираем seed-target вершины.
        for node_gid, _time in inp[node_type]:
            # Добавляем в `budget` вершины (sources), смежные с node_gid (target).
            add_budget(te, node_gid, _time, layer_data, budget)
    '''
        We recursively expand the sampled graph by sampled_depth.
        Each time we sample a fixed number of nodes for each budget,
        based on the accumulated degree.
    '''
    # Теперь проходимся по собранным в `budget` вершинам.
    # `layer` нигде не используется. Просто повторяем алгоритм `sampled_depth` раз.
    for layer in range(sampled_depth):
        source_types = list(budget.keys())
        for source_type in source_types:
            # Первое измерение `graph.edge_list` - это обращение по target_type,
            # а здесь обращение по source_type. Т.е., теперь для вершин в `budget`
            # мы извлекаем триплеты, в которых эти вершины являются target.
            # Из этих триплетов берём source-вершины.
            te = graph.edge_list[source_type]

            # source_ids
            gids  = np.array(list(budget[source_type].keys()))
            if sampled_number > len(gids):
                '''
                    Directly sample all the nodes
                '''
                sampled_idxs = np.arange(len(gids))  # Это индексы!
            else:
                '''
                    Sample based on accumulated degree
                '''
                # Сначала считаем вероятности для каждого элемента.
                # Возводим кумулятивные нормализованные степени в квадрат.
                score = np.array(list(budget[source_type].values()))[:,0] ** 2
                # Нормализуем каждый коэффициент делением на сумму полученных квадратов.
                score = score / np.sum(score)
                # А потом случайно выбираем вершины согласно получившимся вероятностям.
                sampled_idxs = np.random.choice(len(score), sampled_number, p = score, replace = False)

            sampled_gids = gids[sampled_idxs]
            '''
                First adding the sampled nodes then updating budget.
            '''
            for gid in sampled_gids:
                layer_data[source_type][gid] = [len(layer_data[source_type]), budget[source_type][gid][1]]
            for gid in sampled_gids:
                add_budget(te, gid, budget[source_type][gid][1], layer_data, budget)
                # Удаляем вершину из `budget`, но в `layer_data` она остаётся.
                budget[source_type].pop(gid)
    '''
        Prepare feature, time and adjacency matrix for the sampled graph
    '''
    feature, times, indxs, texts = feature_extractor(layer_data, graph)

    # Формат отличается от edge_list класса Graph.
    sampled_edge_list = defaultdict( #target_type
                        lambda: defaultdict(  #source_type
                            lambda: defaultdict(  #relation_type
                                lambda: []  # [target_id, source_id]
                                    )))
    # Добавляем петли (relation_type = 'self').
    for node_type in layer_data:
        for node_gid in layer_data[node_type]:
            # Индекс вершины в сэмплированном подграфе.
            node_sid = layer_data[node_type][node_gid][0]
            sampled_edge_list[node_type][node_type]['self'] += [[node_sid, node_sid]]
    '''
        Reconstruct sampled adjacancy matrix by checking whether each
        link exist in the original graph
    '''
    for target_type in graph.edge_list:
        te = graph.edge_list[target_type]
        tld = layer_data[target_type]
        for source_type in te:
            tes = te[source_type]
            sld  = layer_data[source_type]
            for relation_type in tes:
                tesr = tes[relation_type]
                for target_gid in tld:
                    if target_gid not in tesr:
                        continue
                    target_sid = tld[target_gid][0]
                    for source_gid in tesr[target_gid]:
                        '''
                            Check whether each link (target_id, source_id) exist in original adjacancy matrix
                        '''
                        if source_gid in sld:
                            source_sid = sld[source_gid][0]
                            # Новые индексы вершин в выделенном подграфе.
                            sampled_edge_list[target_type][source_type][relation_type] += [[target_sid, source_sid]]

    return feature, times, sampled_edge_list, indxs, texts

# Первые три аргумента посчитаны для сэмплированного подграфа.
def to_torch(feature, time, edge_list, graph):
    '''
        Transform a sampled sub-graph into pytorch Tensor
        node_dict: {node_type: <node_number, node_type_ID>} node_number is used to trace back the nodes in original graph.
        edge_dict: {edge_type: edge_type_ID}

        feature: {node_type: ndarray} - двумерная матрица для каждого типа вершины, которая содержит векторы вершин.
    '''
    node_dict = {}
    node_feature = []
    node_type    = []
    node_time    = []
    edge_index   = []
    edge_type    = []
    edge_time    = []
    
    node_num = 0  # Кол-во вершин.
    types = graph.get_types()  # ['paper', 'venue', 'field', 'author', 'affiliation']
    # Пронумеруем типы вершин и запишем для них offsets.
    for t in types:
        node_dict[t] = [node_num, len(node_dict)]
        node_num     += len(feature[t])

    # Преобразуем словари, которые группируют элементы по типу вершин, в "плоские" списки.
    # А идентифицировать элементы будем по offsets, которые мы записали выше.
    for t in types:
        node_feature += list(feature[t])
        node_time    += list(time[t])
        node_type    += [node_dict[t][1] for _ in range(len(feature[t]))]

    # Пронумеруем отношения.
    edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
    # Добавим также отношение "Петля".
    edge_dict['self'] = len(edge_dict)

    # Перебираем триплеты в сэмплированном подграфе.
    for target_type in edge_list:
        for source_type in edge_list[target_type]:
            for relation_type in edge_list[target_type][source_type]:
                for ti, si in edge_list[target_type][source_type][relation_type]:
                    # Добавляем к индексам offsets.
                    tid, sid = ti + node_dict[target_type][0], si + node_dict[source_type][0]
                    edge_index += [[sid, tid]]
                    edge_type  += [edge_dict[relation_type]]
                    '''
                        Our time ranges from 1900 - 2020, largest span is 120.

                        https://github.com/acbull/pyHGT/issues/7
                        Разность между датами может быть отрицательной,
                        поэтому к выражению добавляется разница между наименьшей и наибольшей датами (120 лет).
                    '''
                    edge_time  += [node_time[tid] - node_time[sid] + 120]

    node_feature = torch.FloatTensor(node_feature)  # [n_sampled_nodes, n_features]
    node_type    = torch.LongTensor(node_type)  # [n_sampled_nodes,]
    edge_time    = torch.LongTensor(edge_time)  # [n_sampled_edges,]

    # Graph connectivity in COO format with shape [2, num_edges] and type torch.long
    edge_index   = torch.LongTensor(edge_index).t()

    edge_type    = torch.LongTensor(edge_type)  # [n_sampled_nodes,]

    return node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict
    

    
class RenameUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "GPT_GNN.data" or module == 'data':
            renamed_module = "pyHGT.data"
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()
