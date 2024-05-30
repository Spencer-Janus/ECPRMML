import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_mean



def softmax(X):
   X_exp = torch.exp(X)
   partition = X_exp.sum(1, keepdim=True)
   return X_exp / partition
def SumFusion(PF,TF,A,B):
    return PF(A)+TF(B)
def ConcatFusion(PTF,A,B):
    output = torch.cat((A, B), dim=1)
    return PTF(output)
def GatedFusion(fc_x,fc_y,fc_out,x,y,x_gate):
    x_gate  = x_gate  # whether to choose the x to obtain the gate
    sigmoid = nn.Sigmoid()
    out_x = fc_x(x)
    out_y = fc_y(y)
    if x_gate:
            gate   = sigmoid(out_x)
            output = fc_out(torch.mul(gate, out_y))
    else:
            gate   = sigmoid(out_y)
            output = fc_out(torch.mul(out_x, gate))
    return output
def AttentionSum(PF,TF,a,b,w):
        out_x=PF(a)
        out_y=TF(b)
        a=F.relu(w(out_x))
        b=F.relu(w(out_y))

        # with torch.no_grad():
        c=torch.cat((a,b),1)
        c=softmax(c)

        output = c[:,0].reshape(c[:,0].shape[0],1)*out_x+c[:,1].reshape(c[:,1].shape[0],1)*out_y
        return output



class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self, n_users):
        super(Aggregator, self).__init__()
        self.n_users = n_users


    def forward(self, entity_emb, edge_index, edge_type, weight):
        # f=open("up.txt","w",encoding='utf-8')
        # g=open("pr.txt","w",encoding='utf-8')
        n_entities = entity_emb.shape[0]
        # print(entity_emb.shape[0])
        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = weight[edge_type - 1]  
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        return entity_agg
        # return entity_emb


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users,
                 n_factors, n_relations, ind, node_dropout_rate=0.5, mess_dropout_rate=0.1,):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_factors = n_factors
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind

        self.temperature = 0.2

        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        disen_weight_att = initializer(torch.empty(n_factors, n_relations - 1))
        self.disen_weight_att = nn.Parameter(disen_weight_att)

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))
    def forward(self,entity_emb,edge_index, edge_type,
                mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)


        entity_res_emb = entity_emb  # [n_entity, channel]


        for i in range(len(self.convs)):
            entity_emb = self.convs[i](entity_emb, edge_index, edge_type, self.weight)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)

            entity_emb = F.normalize(entity_emb)


            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)


        return entity_res_emb


class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph,agg,picture,text):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities
        self.agg=agg
        self.PictureFeature=torch.tensor(picture).reshape(-1,2048).to("cuda:"'0')
        self.TextFeature=torch.tensor(text).to("cuda:"'0')
        self.w=nn.Linear(args_config.dim, 1,False)
        self.PF=nn.Linear(2048, args_config.dim)
        self.TF=nn.Linear(768, args_config.dim)
        self.PTF=nn.Linear(2816, args_config.dim)
        
        # self.fc_x = nn.Linear(2048, 512)
        # self.fc_y = nn.Linear(768, 512)
        # self.fc_out  = nn.Linear(512, args_config.dim)

        # self.item_emb_last=nn.Embedding(self.n_items, args_config.dim)
        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")

        self.graph = graph
        self.edge_index, self.edge_type = self._get_edges(graph)

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)


        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_entities, self.emb_size))

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         n_factors=self.n_factors,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, batch=None):
        # self.item_emb_last=self.PF(self.PictureFeature)+self.TF(self.TextFeature) #PT
        # self.item_emb_last=self.PF(self.PictureFeature)#P
        # self.item_emb_last=self.TF(self.TextFeature) #T
        # self.item_emb_last=ConcatFusion(self.PTF,self.PictureFeature,self.TextFeature)
        # self.item_emb_last= GatedFusion(self.fc_x,self.fc_y,self.fc_out,self.PictureFeature,self.TextFeature,False)
        self.item_emb_last=AttentionSum(self.PF,self.TF,self.PictureFeature,self.TextFeature,self.w)
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']
        entity_emb = self.all_embed
        entity_gcn_emb= self.gcn(
                                                    entity_emb,                                   
                                                     self.edge_index,
                                                     self.edge_type,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)

        u_e = entity_gcn_emb[user]
        pos_e, neg_e = self.item_emb_last[pos_item], self.item_emb_last[neg_item]
        # pos_e, neg_e = self.item_emb_last(pos_item), self.item_emb_last(neg_item)
        return self.create_bpr_loss(u_e, pos_e, neg_e)

    def generate(self):
        item_emb = self.all_embed
        return self.gcn(
                        item_emb,
                        self.edge_index,
                        self.edge_type,
                        mess_dropout=False, node_dropout=False)[:],self.item_emb_last,self.w 

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss , mf_loss, emb_loss
