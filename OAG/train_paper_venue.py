"""
    Задача восстановления связей между триплетами <Paper, journal, Venue>.
    Необходимо определить, в каком из журналов опубликована статья.
    Задача решается как классификация: на вход подаётся Paper, на выходе - Softmax-вероятности принадлежности к Venue.
    Для оценивания используются метрики ранжирования (MRR), для этого полученные вероятности рассматриваются
    как scores триплетов и преобразуются в ранги.
"""
import sys
from pyHGT.data import *
from pyHGT.model import *
from warnings import filterwarnings
filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(description='Training GNN on Paper-Venue (Journal) classification task')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='./dataset/oag_output',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default='./model_save',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--task_name', type=str, default='PV',
                    help='The name of the stored models and optimization results.')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--domain', type=str, default='_CS',
                    help='CS, Medicion or All: _CS or _Med or (empty)')         
'''
   Model arguments 
'''
parser.add_argument('--conv_name', type=str, default='hgt',
                    choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=400,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=4,
                    help='Number of GNN layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout ratio')
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_width', type=int, default=128,
                    help='How many nodes to be sampled per layer per type')

'''
    Optimization arguments
'''
parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--data_percentage', type=float, default=1.0,
                    help='Percentage of training and validation data to use')
parser.add_argument('--n_epoch', type=int, default=200,
                    help='Number of epoch to run')
parser.add_argument('--n_pool', type=int, default=4,
                    help='Number of process to sample subgraph')    
parser.add_argument('--n_batch', type=int, default=32,
                    help='Number of batch (sampled graphs) for each epoch')   
parser.add_argument('--repeat', type=int, default=2,
                    help='How many time to train over a singe batch (reuse data)') 
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of output nodes for training')    
parser.add_argument('--clip', type=float, default=0.25,
                    help='Gradient Norm Clipping') 


args = parser.parse_args()

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

graph = renamed_load(open(os.path.join(args.data_dir, 'graph%s.pk' % args.domain), 'rb'))

train_range = {t: True for t in graph.times if t != None and t < 2015}
valid_range = {t: True for t in graph.times if t != None and t >= 2015  and t <= 2016}
test_range  = {t: True for t in graph.times if t != None and t > 2016}

types = graph.get_types()

'''
    cand_list stores all the Journal, which is the classification domain.
    
    Paper может быть связан с Venue одним из четырёх отношений: 'conference', 'journal', 'repository', 'patent'.
    В этой задаче рассматривается именно journal!
    Кол-во журналов равно 3514. Т.е., задача ставится как 3514-классовая классификация.
'''
cand_list = list(graph.edge_list['venue']['paper']['PV_Journal'].keys())  # id venue-вершин

'''
Use CrossEntropy (log-softmax + NLL) here, since each paper can be associated with one venue.

Многоклассовая классификация.
'''
criterion = nn.NLLLoss()

# Сэмплирование батча, который формируется случайным образом.
def node_classification_sample(seed, pairs, time_range, batch_size):
    '''
        sub-graph sampling and label preparation for node classification:
        (1) Sample batch_size number of output nodes (papers) and their time.
    '''
    np.random.seed(seed)
    # paper ids, являются seed-вершинами.
    target_ids = np.random.choice(list(pairs.keys()), batch_size, replace = False)
    # Получаем пары paper-time.
    target_info = []
    for target_idx in target_ids:
        # time - год
        _, _time = pairs[target_idx]
        target_info += [[target_idx, _time]]

    '''
        (2) Based on the seed nodes, sample a subgraph with 'sampled_depth' and 'sampled_number'
        
        Сэмплируем подграф, в котором содержатся seed-target вершины (paper) и их source-соседи (в т.ч., venue).
        Вершины в получившемся edge_list имеют другую нумерацию!
        Далее, мы обучаем модель на получившемся подграфе.
        
        Полученные структуры данных сгруппированы по типу вершин. Вершины каждого типа имеют свою нумерацию.
        Т.е., две вершины разного типа могут иметь одинаковый индекс.
    '''
    feature, times, edge_list, _, _ = sample_subgraph(graph, time_range, \
                inp = {'paper': np.array(target_info)}, \
                sampled_depth = args.sample_depth, sampled_number = args.sample_width)

    # Для наглядности:
    # edge_list = defaultdict( #target_type
    #                     lambda: defaultdict(  #source_type
    #                         lambda: defaultdict(  #relation_type
    #                             lambda: [] # [target_id, source_id]
    #                                 )))

    '''
        (3) Mask out the edge between the output target nodes (paper) with output source nodes (Journal)
        
        В новом edge_list, полученным после сэмплирования подграфа, вершины имеют новую нумерацию.
        Seed-вершины будут иметь номера < batch_size, т.к. они добавлялись первыми по порядку.
        Удаляем триплеты Paper-Venue, которые содержат seed-paper вершины.
        TODO: но зачем это делать?
    '''
    masked_edge_list = []
    # Префикс "rev" означает реверсивную связь.
    for target_source_pair in edge_list['paper']['venue']['rev_PV_Journal']:
        target_idx = target_source_pair[0]
        if target_idx >= batch_size:
            masked_edge_list += [target_source_pair]
    edge_list['paper']['venue']['rev_PV_Journal'] = masked_edge_list

    masked_edge_list = []
    for target_source_pair in edge_list['venue']['paper']['PV_Journal']:
        source_idx = target_source_pair[1]
        if source_idx >= batch_size:
            masked_edge_list += [target_source_pair]
    edge_list['venue']['paper']['PV_Journal'] = masked_edge_list
    
    '''
        (4) Transform the subgraph into torch Tensor (edge_index is in format of pytorch_geometric)
    '''
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
            to_torch(feature, times, edge_list, graph)
    '''
        (5) Prepare the labels for each output target node (paper), and their index in sampled graph.
            (node_dict[type][0] stores the start index of a specific type of nodes)
    '''
    y_label = torch.zeros(batch_size, dtype = torch.long)
    for idx, target_id in enumerate(target_ids):
        # Индекс source-вершины (Venue) для данной target (Paper).
        source_id = pairs[target_id][0]
        y_label[idx] = cand_list.index(source_id)

    # Видимо, это индексы seed-paper вершин в сэмплированном подграфе.
    # node_dict: {node_type: <node_number, node_type_ID>}
    x_ids = np.arange(batch_size) + node_dict['paper'][0]

    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, y_label
    
def prepare_data(pool):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    for batch_id in np.arange(args.n_batch):
        p = pool.apply_async(node_classification_sample, args=(randint(), \
            sel_train_pairs, train_range, args.batch_size))
        jobs.append(p)
    p = pool.apply_async(node_classification_sample, args=(randint(), \
            sel_valid_pairs, valid_range, args.batch_size))
    jobs.append(p)
    return jobs

# Ключами таких словарей будут являться id paper-вершин, связанных с venue-вершинами.
# Значение - список из двух элементов: venue id и время триплета.
train_pairs = {}
valid_pairs = {}
test_pairs  = {}
'''
    Prepare all the source nodes (Journal) associated with each target node (paper) as dict
'''

# Добавление source-вершин (Venue) из реверсивных триплетов.
# Почему смотрим только именно на реверсивные?!
# Видимо, это связано со структурой данных, которая представляет граф.
# Иначе, было бы неудобно.
# Paper-Venue - это связь 1-1.
# Venue-Paper - 1-М
for target_id in graph.edge_list['paper']['venue']['rev_PV_Journal']:
    # По идее, здесь всегда должна быть одна итерация, т.к. один Venue для одного Paper.
    for source_id in graph.edge_list['paper']['venue']['rev_PV_Journal'][target_id]:
        _time = graph.edge_list['paper']['venue']['rev_PV_Journal'][target_id][source_id]
        if _time in train_range:
            if target_id not in train_pairs:
                # Добавляем Venue к парам.
                train_pairs[target_id] = [source_id, _time]
        elif _time in valid_range:
            if target_id not in valid_pairs:
                valid_pairs[target_id] = [source_id, _time]
        else:
            if target_id not in test_pairs:
                test_pairs[target_id]  = [source_id, _time]


np.random.seed(43)
'''
    Only train and valid with a certain percentage of data, if necessary.
'''
# Словари имеют тот же формат, что и train_pairs, valid_pairs. Просто урезаны до нужного размера.
sel_train_pairs = {p : train_pairs[p] for p in np.random.choice(list(train_pairs.keys()), int(len(train_pairs) * args.data_percentage), replace = False)}
sel_valid_pairs = {p : valid_pairs[p] for p in np.random.choice(list(valid_pairs.keys()), int(len(valid_pairs) * args.data_percentage), replace = False)}

            
'''
    Initialize GNN (model is specified by conv_name) and Classifier
'''
# Что за 401? Каждая вершина представлена в виде вектора, который получен с помощью конкатенации трёх разных векторов
# разной длины, что можно посмотреть в utils.feature_OAG().
# Конкретно в данной задаче размерность вектора равна 1169.
dim = len(graph.node_feature['paper']['emb'].values[0]) + 401
# На самом деле, это кол-во мета-триплетов. Т.е., предполагается,
# что, если между между двумя типами вершин определено отношение,
# то это отношение имеет уникальное имя и не может быть определено
# между парой с другими типами вершин (учитывая их порядок).
# Также здесь учитываются реверсивные связи.
# +1 для учёта петель.
# TODO: но вот одно отношние "Петля" добавляется между любыми типами вершин.
#  При этом, между одним и тем же типом вершин может быть уже и так определено отношение (Paper->cite->Paper).
num_relations = len(graph.get_meta_graph()) + 1
gnn = GNN(conv_name = args.conv_name, in_dim = dim, \
          n_hid = args.n_hid, n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout,\
          num_types = len(graph.get_types()), num_relations = num_relations).to(device)
# После получения эмбеддингов вершин классифицируем их с помощью линейного слоя + Softmax.
classifier = Classifier(args.n_hid, len(cand_list)).to(device)
# По идее, весь ансамбль (gnn+classifier) "захаркоден" под классификацию вершины на n_venue классов.
# Подаём на вход вектор Paper-вершины и получаем на выходе её Venue (вероятности).
# Такое возможно благодаря тому, что каждая статья может быть опубликована только в одном журнале.
# Мы просто выбираем Venue с максимальной вероятностью.
model = nn.Sequential(gnn, classifier)


if args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters())
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters())
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(model.parameters())

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-6)

stats = []
res = []
best_val   = 0
train_step = 1500

pool = mp.Pool(args.n_pool)
st = time.time()
jobs = prepare_data(pool)

for epoch in np.arange(args.n_epoch) + 1:
    # На каждой эпохе мы заново формируем новые `n_batch` случайных батчей
    # и проходим по ним.
    '''
        Prepare Training and Validation Data
    '''
    train_data = [job.get() for job in jobs[:-1]]
    valid_data = jobs[-1].get()
    pool.close()
    pool.join()
    '''
        After the data is collected, close the pool and then reopen it.
    '''
    # Зачем опять это делать?
    # Видимо, для следующей эпохи.
    # Получается, что на последней эпохе это вычисление будет излишним.
    pool = mp.Pool(args.n_pool)
    jobs = prepare_data(pool)
    et = time.time()
    print('Data Preparation: %.1fs' % (et - st))
    
    '''
        Train (time < 2015)
    '''
    model.train()
    train_losses = []
    torch.cuda.empty_cache()

    # Каждый батч обрабатываем `repeat` раз.
    for _ in range(args.repeat):
        for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel in train_data:
            # Через трансформер прогоняем все вершины сэмплированного подграфа.
            # node_rep: [n_sampled_nodes, n_hid]
            node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                   edge_time.to(device), edge_index.to(device), edge_type.to(device))
            # Но через слой-классификатор пропускаем только исходные Paper-вершины
            # и смотрим на ошибку их классификации.
            # res: [batch_size, n_venue]
            res  = classifier.forward(node_rep[x_ids])
            # ylabel: [batch_size,]
            loss = criterion(res, ylabel.to(device))

            optimizer.zero_grad() 
            torch.cuda.empty_cache()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]
            train_step += 1
            scheduler.step(train_step)
            del res, loss

    '''
        Valid (2015 <= time <= 2016)
    '''
    model.eval()
    with torch.no_grad():
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = valid_data
        node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                   edge_time.to(device), edge_index.to(device), edge_type.to(device))
        res  = classifier.forward(node_rep[x_ids])
        loss = criterion(res, ylabel.to(device))
        
        '''
            Calculate Valid NDCG. Update the best model based on highest NDCG score.
            
            По идее, здесь можно и accuracy посчитать. Но, очевидно, при таком большом кол-ве классов,
            значение accuracy будет очень низким. Поэтому авторы применяют метрики из задач ранжирования: NDCG и MRR.
            Softmax-вероятности рассматриваются как scores триплетов <Paper->journal->Venue>,
            что позволяет ранжировать их между собой.
            Таким образом, метрики считаются только на корректных (согласно онтологии) триплетах.
        '''
        valid_res = []
        for ai, bi in zip(ylabel, res.argsort(descending = True)):
            valid_res += [(bi == ai).int().tolist()]
        valid_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in valid_res])
        
        if valid_ndcg > best_val:
            best_val = valid_ndcg
            torch.save(model, os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
            print('UPDATE!!!')
        
        st = time.time()
        print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid NDCG: %.4f") % \
              (epoch, (st-et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
                    loss.cpu().detach().tolist(), valid_ndcg))
        stats += [[np.average(train_losses), loss.cpu().detach().tolist()]]
        del res, loss
    del train_data, valid_data


'''
    Evaluate the trained model via test set (time > 2016)
'''

with torch.no_grad():
    test_res = []
    # Почему 10?
    for _ in range(10):
        # Формируем случайный батч из тестовой выборки.
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = \
                    node_classification_sample(randint(), test_pairs, test_range, args.batch_size)

        # Скармливаем его модели.
        paper_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                    edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]
        # res: [batch_size, n_venue]
        res = classifier.forward(paper_rep)

        # Делаем оценку.
        # res.argsort(descending=True) преобразует каждый вектор Softmax-вероятностей в вектор индексов,
        # упорядоченных по убыванию значений вероятностей.
        for ai, bi in zip(ylabel, res.argsort(descending = True)):
            # Преобразуем каждый вектор в one-hot вектор. 1 - индекс реального класса, 0 - все остальные.
            # Идеальная ситуация - это когда первый элемент вектора равен единице.
            test_res += [(bi == ai).int().tolist()]

    # Считаем метрики для каждого батча, а затем усредняем.
    test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
    print('Last Test NDCG: %.4f' % np.average(test_ndcg))
    # Это ещё не mean, mean он станет на следующей строчке после усреднения.
    test_mrr = mean_reciprocal_rank(test_res)
    print('Last Test MRR:  %.4f' % np.average(test_mrr))

# Проверка лучшей модели, отобранной на valid-выборке.
best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
best_model.eval()
gnn, classifier = best_model
with torch.no_grad():
    test_res = []
    for _ in range(10):
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = \
                    node_classification_sample(randint(), test_pairs, test_range, args.batch_size)
        paper_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                    edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]
        res = classifier.forward(paper_rep)
        for ai, bi in zip(ylabel, res.argsort(descending = True)):
            test_res += [(bi == ai).int().tolist()]
    test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
    print('Best Test NDCG: %.4f' % np.average(test_ndcg))
    test_mrr = mean_reciprocal_rank(test_res)
    print('Best Test MRR:  %.4f' % np.average(test_mrr))
