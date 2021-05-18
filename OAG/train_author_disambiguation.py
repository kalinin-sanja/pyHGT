"""
    Задача различия авторов. Имея статью и нескольких авторов с одинаковыми именами,
    необходимо определить действительного автора статьи, т.е., при ранжировании поместить его на первое место.

    Несовсем понятен прикладной смысл. Получается, что мы уже должны знать имя автора, чтобы определить его вершину?

    Используем следующие постфиксы:
    '_gid' - для обозначения индекса сущности в графе data.py/Graph;
    '_sid' - для обозначения индекса сущности в сэмплированном подграфе.
"""
import sys
from pyHGT.data import *
from pyHGT.model import *
from warnings import filterwarnings

filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(description='Training GNN on Author Disambiguation task')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='./dataset/oag_output',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default='./model_save',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--task_name', type=str, default='AD',
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
parser.add_argument('--n_layers', type=int, default=3,
                    help='Number of GNN layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout ratio')
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_width', type=int, default=128,
                    help='How many `nodes to be sampled per layer per type')

'''
    Optimization arguments
'''
parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--data_percentage', type=float, default=1.0,
                    help='Percentage of training and validation data to use')
parser.add_argument('--n_epoch', type=int, default=100,
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
graph = renamed_load(open(args.data_dir + '/graph%s.pk' % args.domain, 'rb'))

train_range = {t: True for t in graph.times if t != None and t < 2015}
valid_range = {t: True for t in graph.times if t != None and t >= 2015 and t <= 2016}
test_range = {t: True for t in graph.times if t != None and t > 2016}

types = graph.get_types()

# Выбираем авторов, которые являются первыми авторами не менее двух статей.
apd = graph.edge_list['author']['paper']['rev_AP_write_first']
first_author_dict = {author_gid: True for author_gid in apd if len(apd[author_gid]) >= 2}

# Группируем авторов по одинаковому имени.
name_count = defaultdict(lambda: [])
for author_gid, row in tqdm(graph.node_feature['author'].iterrows(), total=len(graph.node_feature['author'])):
    if author_gid in first_author_dict:
        # Записываем индексы авторов из исходного графа.
        name_count[row['name']] += [author_gid]
# Затем отбрасываем тех, чьи имена встречались менее 4 раз.
name_count = {name: name_count[name] for name in name_count if len(name_count[name]) >= 4}


# CrossEntropyLoss
def mask_softmax(pred, size):
    # На вход Matcher'у мы подаём два вектора: Paper и Author.
    # При этом эти пары изначально сгруппированы по Paper.
    # Для каждой Paper имеем несколько Author: один из них (первый) действительный,
    # остальные - ложные, но с тем же именем, что и действительный.
    # Например:
    # paper_1: [0, 1, 6] - истинный автор 0.
    # paper_2: [3, 2, 4, 5] - истинный автор 3.
    # При скармливании модели мы преобразуем все эти списки в два "flatten", для примера выше:
    # paper_keys: [1, 1, 1, 2, 2, 2, 2]
    # author_keys: [0, 1, 6, 3, 2, 4, 5]
    # Список `size` содержит кол-во Author для каждой Paper, для примера выше: [3, 4].
    # С помощью `size` мы можем посчитать Softmax отдельно для каждой Paper.
    # По идее, `pred` содержит score для каждой пары Paper-Author.
    loss = 0
    stx = 0
    for l in size:
        # TODO: Зачем делить на логарифм?
        loss += torch.log_softmax(pred[stx: stx + l], dim=-1)[0] / np.log(l)
        stx += l
    return -loss


def author_disambiguation_sample(seed, pairs, time_range, batch_size):
    '''
        sub-graph sampling and label preparation for author disambiguation:
        (1) Sample batch_size // 4 number of names
    '''
    # `pairs` - это для каждого уникального имени автора список пар [Author, Paper],
    # в которых имя автора совпадает с данным.
    # По логике кол-во статей должно быть меньше кол-ва авторов, и у каждой статьи может быть только один первый автор.
    # Т.е., среди всех пар статья должна встретиться только один раз, в отличие от авторов.
    np.random.seed(seed)
    # Уменьшает размер батча в 4 раза, т.к., одно имя может принадлежать сразу нескольким авторам.
    names = np.random.choice(list(pairs.keys()), batch_size // 4, replace=False)
    '''
        (2) Get all the papers written by these same-name authors, and then prepare the label
    '''

    # Ключ: индекс автора в исходном графе, Значение: новый индекс.
    # Кол-во ключей равно кол-ву уникальных авторов (n_author) в `pairs`.
    author_dict = {}
    author_info = []  # shape=[n_author,] - список пар [индекс автора в исходном графе, максимальное время в выборке].
    paper_info = []  # shape=[n_paper,] - список пар [индекс статьи в исходном графе, время триплета].
    name_label = []  # shape=[n_paper,] - список массивов авторов с индексами из `author_dict`.
    max_time = np.max(list(time_range.keys()))

    for name in names:
        # Индексы авторов в исходном графе.
        author_list = name_count[name]
        for author_gid in author_list:
            if author_gid not in author_dict:
                author_dict[author_gid] = len(author_dict)
                # TODO: зачем max_time?
                author_info += [[author_gid, max_time]]
        
        for paper_gid, author_idx, _time in pairs[name]:
            paper_info += [[paper_gid, _time]]
            '''
                For each paper, create a list: the first entry is the true author's id, 
                while the others are negative samples (id of authors with same name)
            '''
            # author_idx - индекс автора в списке авторов с одинаковым именем (`author_list`).
            # author_list[author_idx] - индекс автора в исходном графе (`graph`).
            # [author_dict[author_list[author_idx]]] - новый индекс автора.
            name_label += [[author_dict[author_list[author_idx]]] + \
                           [author_dict[author_gid] for (x_id, author_gid) in enumerate(author_list)
                            if x_id != author_idx]]

    '''
        (3) Based on the seed nodes, sample a subgraph with 'sampled_depth' and 'sampled_number'

        Здесь два множества seed-вершин: одно для Paper и одно для Author.
    '''
    # В подграф попадут также другие авторы и статьи, а не только те, которые в pairs.
    feature, times, edge_list, _, _ = sample_subgraph(graph, time_range, \
                                                      inp={'paper': np.array(paper_info),
                                                           'author': np.array(author_info)}, \
                                                      sampled_depth=args.sample_depth, sampled_number=args.sample_width)

    '''
        (4) Mask out the edge between the output target nodes (paper) with output source nodes (author)

        В edge_list сэмплированного подграфа вершины имеют новую нумерацию.
        Seed-вершины будут иметь номера < batch_size, т.к. они добавлялись первыми по порядку.
        Удаляем триплеты Author-Paper, которые содержат seed-Paper вершины.
        TODO: но зачем это делать?
         В статье написано:
         "To avoid data leakage, we remove out the links we aim to predict
         (e.g. the Paper-Field link as the label) from the sub-graph."
    '''
    masked_edge_list = []
    for target_source_pair in edge_list['paper']['author']['AP_write_first']:
        target_sid = target_source_pair[0]
        if target_sid >= batch_size:
            masked_edge_list += [target_source_pair]
    edge_list['paper']['author']['AP_write_first'] = masked_edge_list

    masked_edge_list = []
    for target_source_pair in edge_list['author']['paper']['rev_AP_write_first']:
        source_sid = target_source_pair[1]
        if source_sid >= batch_size:
            masked_edge_list += [target_source_pair]
    edge_list['author']['paper']['rev_AP_write_first'] = masked_edge_list

    '''
        (5) Transform the subgraph into torch Tensor (edge_index is in format of pytorch_geometric)
    '''
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
        to_torch(feature, times, edge_list, graph)
    '''
        (6) Prepare the labels for each output target node (paper), and their index in sampled graph.
            (node_dict[type][0] stores the start index of a specific type of nodes)
    '''
    ylabel = {}  # кол-во ключей равно кол-ву статей
    # По сути, для каждой Paper мы получаем тот же name_label, но с преобразованными индексами авторов.
    for x_id, author_ids in enumerate(name_label):
        # Ключ - статья.
        # Значение - список авторов.
        ylabel[x_id + node_dict['paper'][0]] = np.array(author_ids) + node_dict['author'][0]
    return node_feature, node_type, edge_time, edge_index, edge_type, ylabel


def prepare_data(pool):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    for batch_id in np.arange(args.n_batch):
        p = pool.apply_async(author_disambiguation_sample, args=(randint(), \
                                                                 sel_train_pairs, train_range, args.batch_size))
        jobs.append(p)
    p = pool.apply_async(author_disambiguation_sample, args=(randint(), \
                                                             sel_valid_pairs, valid_range, args.batch_size))
    jobs.append(p)
    return jobs


train_pairs = {}
valid_pairs = {}
test_pairs = {}
'''
    Prepare all the author with same name and their written papers.
'''

for name in name_count:
    same_name_author_list = np.array(name_count[name])
    # Перебираем всех авторов с этим именем, а затем их статьи.
    for author_idx, author_gid in enumerate(same_name_author_list):
        for paper_gid in graph.edge_list['author']['paper']['rev_AP_write_first'][author_gid]:
            _time = graph.edge_list['author']['paper']['rev_AP_write_first'][author_gid][paper_gid]
            # Жестокое условие
            if type(_time) != int:
                continue
            # Сохраняем пары и их время.
            if _time in train_range:
                if name not in train_pairs:
                    train_pairs[name] = []
                train_pairs[name] += [[paper_gid, author_idx, _time]]
            elif _time in valid_range:
                if name not in valid_pairs:
                    valid_pairs[name] = []
                valid_pairs[name] += [[paper_gid, author_idx, _time]]
            else:
                if name not in test_pairs:
                    test_pairs[name] = []
                test_pairs[name] += [[paper_gid, author_idx, _time]]

np.random.seed(43)
'''
    Only train and valid with a certain percentage of data, if necessary.
'''
sel_train_pairs = {p: train_pairs[p] for p in
                   np.random.choice(list(train_pairs.keys()), int(len(train_pairs) * args.data_percentage),
                                    replace=False)}
sel_valid_pairs = {p: valid_pairs[p] for p in
                   np.random.choice(list(valid_pairs.keys()), int(len(valid_pairs) * args.data_percentage),
                                    replace=False)}

'''
    Initialize GNN (model is specified by conv_name) and Classifier
'''
# Что за 401? Каждая вершина представлена в виде вектора, который получен с помощью конкатенации трёх разных векторов
# разной длины, что можно посмотреть в utils.feature_OAG().
# Конкретно в данной задаче размерность вектора равна 1169.
dim = len(graph.node_feature['paper']['emb'].values[0]) + 401
# На самом деле, это кол-во мета-триплетов. Т.е., предполагается,
# что, если между двумя типами вершин определено отношение,
# то это отношение имеет уникальное имя и не может быть определено
# между парой с другими типами вершин (учитывая их порядок).
# Также здесь учитываются реверсивные связи.
# +1 для учёта петель.
num_relations = len(graph.get_meta_graph()) + 1

gnn = GNN(conv_name=args.conv_name, in_dim=dim, \
          n_hid=args.n_hid, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout, \
          num_types=len(graph.get_types()), num_relations=num_relations).to(device)
matcher = Matcher(args.n_hid).to(device)

model = nn.Sequential(gnn, matcher)

if args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters())
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters())
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(model.parameters())

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-6)

stats = []
res = []
best_val = 0
train_step = 1500

pool = mp.Pool(args.n_pool)
st = time.time()
jobs = prepare_data(pool)

# TODO: для чего +1?
for epoch in np.arange(args.n_epoch) + 1:
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
    for _ in range(args.repeat):
        for node_feature, node_type, edge_time, edge_index, edge_type, ylabel in train_data:
            node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                   edge_time.to(device), edge_index.to(device), edge_type.to(device))

            author_key = []
            paper_key = []
            key_size = []  # shape: len(ylabel)
            
            for paper_sid in ylabel:
                author_sids = ylabel[paper_sid]
                paper_key += [np.repeat(paper_sid, len(author_sids))]
                author_key += [author_sids]
                key_size += [len(author_sids)]

            # Объединяем список массивов в один массив (тензор).
            # paper_key[i] - статья "A".
            # author_key[i] - реальный или нереальный автор статьи "А".
            # Соответственно, оба массива имеют одинаковую размерность.
            paper_key = torch.LongTensor(np.concatenate(paper_key)).to(device)
            author_key = torch.LongTensor(np.concatenate(author_key)).to(device)

            train_paper_vecs = node_rep[paper_key]  # [len(paper_key), n_hid]
            train_author_vecs = node_rep[author_key]  # [len(paper_key), n_hid]
            # shape: [len(paper_key), n_hid]
            res = matcher.forward(train_author_vecs, train_paper_vecs, pair=True)
            loss = mask_softmax(res, key_size)

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
        node_feature, node_type, edge_time, edge_index, edge_type, ylabel = valid_data
        node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                               edge_time.to(device), edge_index.to(device), edge_type.to(device))

        author_key = []
        paper_key = []
        key_size = []
        for paper_sid in ylabel:
            author_sids = ylabel[paper_sid]
            paper_key += [np.repeat(paper_sid, len(author_sids))]
            author_key += [author_sids]
            key_size += [len(author_sids)]

        paper_key = torch.LongTensor(np.concatenate(paper_key)).to(device)
        author_key = torch.LongTensor(np.concatenate(author_key)).to(device)

        valid_paper_vecs = node_rep[paper_key]
        valid_author_vecs = node_rep[author_key]
        # shape: [len(paper_key), h_hid]
        res = matcher.forward(valid_author_vecs, valid_paper_vecs, pair=True)
        loss = mask_softmax(res, key_size)
        '''
            Calculate Valid NDCG. Update the best model based on highest NDCG score.
        '''
        valid_res = []
        ser = 0
        for author_count in key_size:
            scores = res[ser: ser + author_count]  # список score возможных авторов для данной статьи
            ohe_vector = torch.zeros(author_count)
            ohe_vector[0] = 1  # первый элемент - истинный автор
            ohe_vector = ohe_vector[scores.argsort(descending=True)]  # упорядочиваем векторы по убыванию score
            # Для данной статьи имеем One-Hot вектор, где единицей обозначен реальный автор.
            valid_res += [ohe_vector.cpu().detach().tolist()]
            ser += author_count
        valid_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in valid_res])
        valid_mrr = np.average(mean_reciprocal_rank(valid_res))

        if valid_ndcg > best_val:
            best_val = valid_ndcg
            torch.save(model, os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
            print('UPDATE!!!')

        st = time.time()
        print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid NDCG: %.4f  Valid MRR: %.4f") % \
              (epoch, (st - et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
               loss.cpu().detach().tolist(), valid_ndcg, valid_mrr))
        stats += [[np.average(train_losses), loss.cpu().detach().tolist()]]
        del res, loss
    del train_data, valid_data

'''
    Evaluate the trained model via test set (time > 2016)
'''

best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
best_model.eval()
gnn, matcher = best_model
with torch.no_grad():
    test_res = []
    for _ in range(10):
        node_feature, node_type, edge_time, edge_index, edge_type, ylabel = \
            author_disambiguation_sample(randint(), test_pairs, test_range, args.batch_size)
        node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                               edge_time.to(device), edge_index.to(device), edge_type.to(device))

        author_key = []
        paper_key = []
        key_size = []
        for paper_sid in ylabel:
            author_sids = ylabel[paper_sid]
            paper_key += [np.repeat(paper_sid, len(author_sids))]
            author_key += [author_sids]
            key_size += [len(author_sids)]

        paper_key = torch.LongTensor(np.concatenate(paper_key)).to(device)
        author_key = torch.LongTensor(np.concatenate(author_key)).to(device)

        test_paper_vecs = node_rep[paper_key]
        test_author_vecs = node_rep[author_key]
        res = matcher.forward(test_author_vecs, test_paper_vecs, pair=True)

        ser = 0
        for author_count in key_size:
            scores = res[ser: ser + author_count]  # список score возможных авторов для данной статьи
            ohe_vector = torch.zeros(author_count)
            ohe_vector[0] = 1  # первый элемент - истинный автор
            ohe_vector = ohe_vector[scores.argsort(descending=True)]  # упорядочиваем векторы по убыванию score
            # Для данной статьи имеем One-Hot вектор, где единицей обозначен реальный автор.
            test_res += [ohe_vector.cpu().detach().tolist()]
            ser += author_count

    test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
    print('Test NDCG: %.4f' % np.average(test_ndcg))
    # TODO: для разных статей разное кол-во возможных авторов?
    test_mrr = mean_reciprocal_rank(test_res)
    print('Test MRR:  %.4f' % np.average(test_mrr))
