import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import math

class HGTConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = True, use_RTE = True, **kwargs):
        super(HGTConv, self).__init__(aggr='add', **kwargs)

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations  # В Paper-Venue задаче: 16*2 + 1 = 33
        self.total_rel     = num_types * num_relations * num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.use_norm      = use_norm
        self.use_RTE       = use_RTE
        self.att           = None
        
        
        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()

        # Каждое отношение имеет свои матрицы Query, Key, Value.
        # На самом деле, GNN создаёт HGTConv с аргументами in_dim = out_dim = h_dim.
        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        '''
            TODO: make relation_pri smaller, as not all <st, rt, tt> pair exist in meta relation list.
            
            "Since not all the relationships contribute equally to the target nodes, we add a prior tensor
            to denote the general significance of each meta relation triplet,
            serving as an adaptive scaling to the attention".
        '''
        self.relation_pri   = nn.Parameter(torch.ones(num_relations, self.n_heads))

        # "We need to calculate the similarity between the Query vector Qi (t) and Key vector Ki (s).
        # One unique characteristic of heterogeneous graphs is that there may exist different edge types (relations)
        # between a node type pair, e.g., type(s) and type(t). Therefore, unlike the vanilla Transformer that
        # directly calculates the dot product between the Query and Key vectors, we keep a distinct
        # edge-based matrix for each edge type. In doing so, the model can capture different semantic relations
        # even between the same node type pairs."
        self.relation_att   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))

        self.skip           = nn.Parameter(torch.ones(num_types))
        self.drop           = nn.Dropout(dropout)
        
        if self.use_RTE:
            self.emb            = RelTemporalEncoding(in_dim)

        # Инициализация тензоров.
        glorot(self.relation_att)
        glorot(self.relation_msg)
        
    def forward(self, node_inp, node_type, edge_index, edge_type, edge_time):
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type, \
                              edge_type=edge_type, edge_time = edge_time)

        # Основная суть метода propagate:
        #     out = self.message(*message_args)
        #     # default aggr: 'add'.
        #     out = scatter_(self.aggr, out, edge_index[i], dim, dim_size=size[i])
        #     out = self.update(out, *update_args)
        #
        #     return out

    # See Figure 2 in the paper.
    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_time):
        '''
            Вызывается методом self.propagate.
            j: source, i: target; <j, i>
        '''
        # Кол-во рёбер в сэмплированном подграфе.
        data_size = edge_index_i.size(0)
        '''
            Create Attention and Message tensor beforehand.
        '''
        res_att     = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg     = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)

        # Начинаем проход по "мета-триплетам".
        # Проходимся по типам вершин для source.
        for source_type in range(self.num_types):
            sb = (node_type_j == int(source_type))
            # В слоях Key и Value храним векторы для Source.
            # Key используется в (1) Heterogeneous Mutual Attention.
            k_linear = self.k_linears[source_type]
            # Value используется в (2) Heterogeneous Message Passing.
            v_linear = self.v_linears[source_type]
            # Проходимся по типам вершин для target.
            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                # В Query храним векторы для target.
                # Query используется в (1) Heterogeneous Mutual Attention.
                q_linear = self.q_linears[target_type]
                # Проходимся по отношениям.
                for relation_type in range(self.num_relations):
                    '''
                        idx is all the edges with meta relation <source_type, relation_type, target_type>
                    '''
                    idx = (edge_type == int(relation_type)) & tb
                    if idx.sum() == 0:
                        continue
                    '''
                        Get the corresponding input node representations by idx.
                        Add tempotal encoding to source representation (j)
                    '''
                    target_node_vec = node_inp_i[idx]
                    source_node_vec = node_inp_j[idx]

                    # Пока пропустим изучение этого.
                    if self.use_RTE:
                        # "Для того, чтобы модель понимала порядок слов, мы добавляем векторы позиционного кодирования,
                        # значения которых следуют определенному шаблону."
                        source_node_vec = self.emb(source_node_vec, edge_time[idx])
                    '''
                        Step 1: Heterogeneous Mutual Attention
                    '''
                    # Получаем векторы Query и Key умножением эмбеддингов вершин на соответствующие матрицы.
                    # Сами матрицы обновляются в процессе обучения.
                    # Новые векторы меньше в размере (`d_k`), чем исходные (`d`).
                    # Как говорится в одном блоге про архитектуру Трансформер (https://habr.com/ru/post/486358/):
                    # "Это абстракции, которые оказываются весьма полезны для понимания и вычисления внимания."
                    # "Они не обязаны быть меньше, но в нашем случае выбор данной
                    # архитектуры модели обусловлен желанием сделать вычисления в слое
                    # множественного внимания (multi-head attention) более стабильными".
                    # Мы имеем разные векторы для каждой из "голов внимания", количество которых `n_heads`
                    # Размерности: [idx.sum(), n_heads, d_k]
                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    # Необходимо перемножить векторы `q` и `k`, чтобы получить score.
                    # Т.е., аналогично работе с текстом, мы оцениваем каждое слово (source) во входящем предложении
                    # по отношению к данному слову (target).
                    # Score определяет, насколько нужно сфокусироваться на других частях
                    # входящего предложения во время кодирования слова в конкретной позиции.
                    # TODO: что делает эта строка? Зачем тут relation_att
                    k_mat = torch.bmm(k_mat.transpose(1,0), self.relation_att[relation_type]).transpose(1,0)
                    # Разделим эти scores на квадратный корень размерности векторов Keys.
                    # Данное значение обеспечивает более стабильные градиенты.
                    # TODO: зачем здесь сумма? Что такое relation_pri?
                    res_att[idx] = (q_mat * k_mat).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk
                    '''
                        Step 2: Heterogeneous Message Passing
                    '''
                    # Получаем векторы Value для source, той же размерности, что Query и Key.
                    # Одна вершина имеет по одному вектору для каждой головы внимания.
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    # TODO: непонятно.
                    res_msg[idx] = torch.bmm(v_mat.transpose(1,0), self.relation_msg[relation_type]).transpose(1,0)
        '''
            Softmax based on target node's id (edge_index_i). Store attention value in self.att for later visualization.
        '''
        # Получаем attention weights.
        # "Полученный софтмакс-коэффициент (softmax score) определяет, в какой мере каждое из слов предложения
        # будет выражено в определенной позиции. Очевидно, что слово в своей позиции получит
        # наибольший софтмакс-коэффициент, но иногда полезно учитывать и другое слово, релевантное к рассматриваемому".
        # self.att: [n_edges, n_heads]
        # TODO: разобрать, как здесь считается softmax.
        self.att = softmax(res_att, edge_index_i)
        # Агрегированные векторы, полученные с Heterogeneous Message Passing,
        # взвешенные согласно полученным attention weights.
        # В res_msg сообщение от каждой "головы внимания" представлено вектором размерности d_k.
        # В self.att хранится коэффициент для каждого вектора, элементы которого умножаются на этот коэффициент.
        # res_msg: [n_edges, n_heads, d_k]
        # self.att.view(-1, self.n_heads, 1): [n_edges, n_heads, 1]
        # res: [n_edges, n_heads, d_k]
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        return res.view(-1, self.out_dim)
        # Затем все векторы source-вершин группируются по target-вершине и суммируются между собой.
        # https://raw.githubusercontent.com/rusty1s/pytorch_scatter/master/docs/source/_figures/add.svg


    # aggr_out: [n_sampled_nodes, dim]
    def update(self, aggr_out, node_inp, node_type):
        '''
            Step 3: Target-specific Aggregation
            x = W[node_type] * gelu(Agg(x)) + x

            "With the heterogeneous multi-head attention and message calculated,
            we need to aggregate them from the source nodes to the target node (See Figure 2 (3)).
            Note that the softmax procedure in Eq. 2 has made the sum of each target node t’s attention vectors
            to one, we can thus simply use the attention vector as the weight to average the corresponding messages
            from the source nodes and get the updated vector H(l)[t]."
        '''
        # В статье этого нет.
        aggr_out = F.gelu(aggr_out)

        # Инициализируем матрицу размерности batch_size строк.
        # Далее заполняем её векторами вершин подграфа, каждый из которых вычисляем согласно типу вершины.
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)

        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue

            # Прогоняем векторы через ещё один слой, ассоциированный с типом target-вершины.
            # Применяем drop-out.
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx]))

            '''
                Add skip connection with learnable weight self.skip[t_id]
            '''
            # `self.skip` - вектор единиц размерности `num_types`.
            # Чем alpha больше, тем больше учитываются новые векторы.
            # Чем alpha меньше, тем тем больше учитываются исходные векторы.
            # Очень странно применяется сигмоида.
            # Похоже, что это формула 4 статьи.
            # "The final step is to map target node t’s vector back to its type-
            # specific distribution, indexed by its node type. To do so, we
            # apply a linear projection A-Linear to the updated vector H(l)[t],
            # followed by a non-linear activation and residual connection [5]".
            alpha = torch.sigmoid(self.skip[target_type])
            # Добавление к новому вектору старого (node_inp[idx]) - это residual connection.
            if self.use_norm:
                # Для каждого типа вершины также свой слой нормализации.
                res[idx] = self.norms[target_type](trans_out * alpha + node_inp[idx] * (1 - alpha))
            else:
                res[idx] = trans_out * alpha + node_inp[idx] * (1 - alpha)
        return res

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)
    
    
    
class DenseHGTConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = True, use_RTE = True, **kwargs):
        super(DenseHGTConv, self).__init__(aggr='add', **kwargs)

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.total_rel     = num_types * num_relations * num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.use_norm      = use_norm
        self.use_RTE       = use_RTE
        self.att           = None
        
        
        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()

        
        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        '''
            TODO: make relation_pri smaller, as not all <st, rt, tt> pair exist in meta relation list.
        '''
        self.relation_pri   = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.drop           = nn.Dropout(dropout)
        
        if self.use_RTE:
            self.emb            = RelTemporalEncoding(in_dim)
        
        glorot(self.relation_att)
        glorot(self.relation_msg)
        
        
        self.mid_linear  = nn.Linear(out_dim,  out_dim * 2)
        self.out_linear  = nn.Linear(out_dim * 2,  out_dim)
        self.out_norm    = nn.LayerNorm(out_dim)
        
    def forward(self, node_inp, node_type, edge_index, edge_type, edge_time):
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type, \
                              edge_type=edge_type, edge_time = edge_time)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_time):
        '''
            j: source, i: target; <j, i>
        '''
        data_size = edge_index_i.size(0)
        '''
            Create Attention and Message tensor beforehand.
        '''
        res_att     = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg     = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        
        for source_type in range(self.num_types):
            sb = (node_type_j == int(source_type))
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type] 
            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]
                for relation_type in range(self.num_relations):
                    '''
                        idx is all the edges with meta relation <source_type, relation_type, target_type>
                    '''
                    idx = (edge_type == int(relation_type)) & tb
                    if idx.sum() == 0:
                        continue
                    '''
                        Get the corresponding input node representations by idx.
                        Add tempotal encoding to source representation (j)
                    '''
                    target_node_vec = node_inp_i[idx]
                    source_node_vec = node_inp_j[idx]
                    if self.use_RTE:
                        source_node_vec = self.emb(source_node_vec, edge_time[idx])
                    '''
                        Step 1: Heterogeneous Mutual Attention
                    '''
                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(k_mat.transpose(1,0), self.relation_att[relation_type]).transpose(1,0)
                    res_att[idx] = (q_mat * k_mat).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk
                    '''
                        Step 2: Heterogeneous Message Passing
                    '''
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(v_mat.transpose(1,0), self.relation_msg[relation_type]).transpose(1,0)   
        '''
            Softmax based on target node's id (edge_index_i). Store attention value in self.att for later visualization.
        '''
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        return res.view(-1, self.out_dim)


    def update(self, aggr_out, node_inp, node_type):
        '''
            Step 3: Target-specific Aggregation
            x = W[node_type] * Agg(x) + x
        '''
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx])) + node_inp[idx]
            '''
                Add skip connection with learnable weight self.skip[t_id]
            '''
            if self.use_norm:
                trans_out = self.norms[target_type](trans_out)
                
            '''
                Step 4: Shared Dense Layer
                x = Out_L(gelu(Mid_L(x))) + x
            '''
                
            trans_out     = self.drop(self.out_linear(F.gelu(self.mid_linear(trans_out)))) + trans_out
            res[idx]      = self.out_norm(trans_out)
        return res

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''
    def __init__(self, n_hid, max_len = 240, dropout = 0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)
    def forward(self, x, t):
        return x + self.lin(self.emb(t))
    
    
    
class GeneralConv(nn.Module):
    def __init__(self, conv_name, in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm = True, use_RTE = True):
        super(GeneralConv, self).__init__()
        self.conv_name = conv_name
        if self.conv_name == 'hgt':
            self.base_conv = HGTConv(in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm, use_RTE)
        elif self.conv_name == 'dense_hgt':
            self.base_conv = DenseHGTConv(in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm, use_RTE)
        elif self.conv_name == 'gcn':
            self.base_conv = GCNConv(in_hid, out_hid)
        elif self.conv_name == 'gat':
            self.base_conv = GATConv(in_hid, out_hid // n_heads, heads=n_heads)
    def forward(self, meta_xs, node_type, edge_index, edge_type, edge_time):
        if self.conv_name == 'hgt':
            return self.base_conv(meta_xs, node_type, edge_index, edge_type, edge_time)
        elif self.conv_name == 'gcn':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'gat':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'dense_hgt':
            return self.base_conv(meta_xs, node_type, edge_index, edge_type, edge_time)
    
