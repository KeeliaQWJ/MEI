import torch
from tqdm import tqdm
import torch.nn.functional as F

'''
这段代码主要是用于计算一组数据点（例如蛋白质序列）的聚类中心和两组数据点之间的距离矩阵。以下是各个函数的功能解释：
 1. `get_cluster_center`  函数:计算每个聚类的中心。输入是模型嵌入model_emb,和一个字典,字典的键是聚类的标识符,例如EC号,
    值是属于该聚类的数据点的ID列表。输出是一个字典,键是聚类的标识符，值是聚类中心的向量。
 2. `dist_map_helper_dot`  函数:计算两组数据点之间的距离矩阵,使用点积距离。输入是两组数据点的键,例如EC号或ID,和它们对应的嵌入向量。输出是一个嵌套字典，表示每对数据点之间的距离。
 3. `dist_map_helper`  函数：与  `dist_map_helper_dot`  类似，但使用欧几里得距离来计算距离矩阵。
 4. `get_dist_map`  函数:计算训练集中所有可能的EC聚类中心之间的距离矩阵。输入是EC-ID字典、ESM嵌入、设备、数据类型和模型,可选。输出是距离矩阵。
 5. `get_dist_map_test`  函数:计算测试集数据点和训练集EC聚类中心之间的距离矩阵。输入是训练集和测试集的模型嵌入、训练集的EC-ID字典、测试集的ID-EC字典、设备和数据类型。输出是距离矩阵。
 6. `get_random_nk_dist_map`  函数:计算从训练集中随机选择的nk个数据点与所有EC聚类中心之间的距离矩阵。
    输入是训练集嵌入、随机选择的nk个训练集嵌入、训练集的EC-ID字典、随机选择的nk个ID、设备和数据类型。输出是距离矩阵。
 总之，这段代码的主要目的是计算聚类中心和不同数据集之间的距离矩阵，以便进行聚类分析和模型评估。
'''

def get_cluster_center(model_emb, ec_id_dict):
    cluster_center_model = {}
    id_counter = 0
    with torch.no_grad():
        for ec in tqdm(list(ec_id_dict.keys())): # list(ec_id_dict.keys())是将字典ec_id_dict中的所有键转换为一个列表，tqdm来显示循环处理ec_id_dict字典中的所有键的进度条
            ids_for_query = list(ec_id_dict[ec])
            id_counter_prime = id_counter + len(ids_for_query)
            emb_cluster = model_emb[id_counter: id_counter_prime] # 作获取该聚类对应的嵌入向量
            cluster_center = emb_cluster.mean(dim=0) # mean方法计算嵌入向量的平均值，即该聚类的中心
            cluster_center_model[ec] = cluster_center.detach().cpu() # 将该聚类的中心存储在cluster_center_model字典中
            id_counter = id_counter_prime # 更新id_counter的值
    return cluster_center_model # 返回cluster_center_model字典，其中包含了每个聚类的中心


def dist_map_helper_dot(keys1, lookup1, keys2, lookup2): # 计算两个向量集合之间的距离
    dist = {} # 存储距离信息
    lookup1 = F.normalize(lookup1, dim=-1, p=2) # 对lookup1和lookup2中的向量进行归一化处理，以便进行距离计算
    lookup2 = F.normalize(lookup2, dim=-1, p=2)
    for i, key1 in tqdm(enumerate(keys1)):
        current = lookup1[i].unsqueeze(0) # lookup1[i]表示从lookup1中获取第i个元素，unsqueeze(0)则是在该元素的第0个维度上增加一个维度，使其成为一个单独的张量。这个操作的结果是将原本的一维张量变成了二维张量
        dist_norm = (current - lookup2).norm(dim=1, p=2) # 计算current与lookup2中所有向量的距离。距离计算使用的是欧几里得距离公式，并将结果存储在dist_norm中
        dist_norm = dist_norm**2
        #dist_norm = (current - lookup2).norm(dim=1, p=2)
        dist_norm = dist_norm.detach().cpu().numpy() #使用detach方法和cpu方法将结果从计算图中分离出来，并转换为numpy数组
        dist[key1] = {}
        for j, key2 in enumerate(keys2):
            dist[key1][key2] = dist_norm[j] # 将当前元素的键值作为dist字典的键，再创建一个空字典作为值，并将当前元素与keys2列表中的每个元素的距离作为值存储在这个空字典中
    return dist # 返回dist字典，其中包含了keys1和keys2两个向量集合之间的距离信息


def dist_map_helper(keys1, lookup1, keys2, lookup2):
    dist = {}
    for i, key1 in tqdm(enumerate(keys1)):
        current = lookup1[i].unsqueeze(0)
        dist_norm = (current - lookup2).norm(dim=1, p=2)
        dist_norm = dist_norm.detach().cpu().numpy()
        dist[key1] = {}
        for j, key2 in enumerate(keys2):
            dist[key1][key2] = dist_norm[j]
    return dist


def get_dist_map(ec_id_dict, esm_emb, device, dtype, model=None, dot=False):
    '''
    Get the distance map for training, size of (N_EC_train, N_EC_train)
    between all possible pairs of EC cluster centers
    '''
    # inference all queries at once to get model embedding
    if model is not None:
        model_emb = model(esm_emb.to(device=device, dtype=dtype))
    else:
        # the first distance map before training comes from ESM
        model_emb = esm_emb
    # calculate cluster center by averaging all embeddings in one EC
    cluster_center_model = get_cluster_center(model_emb, ec_id_dict)
    # organize cluster centers in a matrix
    total_ec_n, out_dim = len(ec_id_dict.keys()), model_emb.size(1)
    model_lookup = torch.zeros(total_ec_n, out_dim, device=device, dtype=dtype)
    ecs = list(cluster_center_model.keys())
    for i, ec in enumerate(ecs):
        model_lookup[i] = cluster_center_model[ec]
    model_lookup = model_lookup.to(device=device, dtype=dtype)
    # calculate pairwise distance map between total_ec_n * total_ec_n pairs
    print(f'Calculating distance map, number of unique EC is {total_ec_n}')
    if dot:
        model_dist = dist_map_helper_dot(ecs, model_lookup, ecs, model_lookup)
    else:
        model_dist = dist_map_helper(ecs, model_lookup, ecs, model_lookup)
    return model_dist


def get_dist_map_test(model_emb_train, model_emb_test,
                      ec_id_dict_train, id_ec_test,
                      device, dtype, dot=False):
    '''
    Get the pair-wise distance map for test queries and train EC cluster centers
    map is of size of (N_test_ids, N_EC_train)
    '''
    print("The embedding sizes for train and test:",
          model_emb_train.size(), model_emb_test.size())
    # get cluster center for all EC appeared in training set
    cluster_center_model = get_cluster_center(
        model_emb_train, ec_id_dict_train)
    total_ec_n, out_dim = len(ec_id_dict_train.keys()), model_emb_train.size(1)
    model_lookup = torch.zeros(total_ec_n, out_dim, device=device, dtype=dtype)
    ecs = list(cluster_center_model.keys())
    for i, ec in enumerate(ecs):
        model_lookup[i] = cluster_center_model[ec]
    model_lookup = model_lookup.to(device=device, dtype=dtype)
    # calculate distance map between n_query_test * total_ec_n (training) pairs
    ids = list(id_ec_test.keys())
    print(f'Calculating eval distance map, between {len(ids)} test ids '
          f'and {total_ec_n} train EC cluster centers')
    if dot:
        eval_dist = dist_map_helper_dot(ids, model_emb_test, ecs, model_lookup)
    else:
        eval_dist = dist_map_helper(ids, model_emb_test, ecs, model_lookup)
    return eval_dist


def get_random_nk_dist_map(emb_train, rand_nk_emb_train,
                           ec_id_dict_train, rand_nk_ids,
                           device, dtype, dot=False):
    '''
    Get the pair-wise distance map between 
    randomly chosen nk ids from training and all EC cluster centers 
    map is of size of (nk, N_EC_train)
    '''
    cluster_center_model = get_cluster_center(emb_train, ec_id_dict_train)
    total_ec_n, out_dim = len(ec_id_dict_train.keys()), emb_train.size(1)
    model_lookup = torch.zeros(total_ec_n, out_dim, device=device, dtype=dtype)
    ecs = list(cluster_center_model.keys())
    for i, ec in enumerate(ecs):
        model_lookup[i] = cluster_center_model[ec]
    model_lookup = model_lookup.to(device=device, dtype=dtype)
    if dot:
        random_nk_dist_map = dist_map_helper_dot(
            rand_nk_ids, rand_nk_emb_train, ecs, model_lookup)
    else:
        random_nk_dist_map = dist_map_helper(
            rand_nk_ids, rand_nk_emb_train, ecs, model_lookup)
    return random_nk_dist_map
