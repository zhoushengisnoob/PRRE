import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from numpy.linalg import inv
import random

def jaccard(edge_path):
    neighbor_set_dict={}
    node_set=set()
    edge_num=0
    with open(edge_path) as fp:
        for line in fp.readlines():
            edge_num+=1
            node1=int(line.strip('\n\r').split()[0])
            node2=int(line.strip('\n\r').split()[1])
            node_set.add(node1)
            node_set.add(node2)
            if node1 not in neighbor_set_dict:
                neighbor_set_dict[node1]=set()
                neighbor_set_dict[node1].add(node2)
            else:
                neighbor_set_dict[node1].add(node2)
            if node2 not in neighbor_set_dict:
                neighbor_set_dict[node2]=set()
                neighbor_set_dict[node2].add(node1)
            else:
                neighbor_set_dict[node2].add(node1)
    node_num=len(node_set)
    print('Node number:',node_num)
    print('Edge number:',edge_num)
    num=0
    sim_mat=np.zeros((node_num,node_num))
    row=[]
    col=[]
    data=[]
    for i in range(node_num):
        for j in range(node_num):
            i_nbr=neighbor_set_dict[i]
            j_nbr=neighbor_set_dict[j]
            inter=len(i_nbr.intersection(j_nbr))
            union=len(i_nbr.union(j_nbr))
            score=float(inter)/union
            sim_mat[i,j]=score
            if i!=j and score>0:
                num+=1
                row.append(i)
                col.append(j)
                data.append(score)
    M=csr_matrix((data, (row, col)), shape=(node_num, node_num))
    print('Jaccard simiarity finished!')
    print(float(num)/(node_num*node_num))
    return M.toarray(),node_num

def katz(edge_path,beta):
    '''
    S=(I-beta*A)^-1 * beta*A
    :param edge_path:
    :param beta:
    :return:
    '''
    node_set = set()
    row=[]
    col=[]
    G=nx.Graph()
    with open(edge_path) as fp:
        for line in fp.readlines():
            node1 = int(line.strip('\n\r').split()[0])
            node2 = int(line.strip('\n\r').split()[1])
            G.add_edge(node1,node2)
            row.append(node1)
            row.append(node2)
            col.append(node2)
            col.append(node1)
            node_set.add(node1)
            node_set.add(node2)
    node_num=len(node_set)
    print('node num:',node_num,G.number_of_nodes())
    A=np.zeros((node_num,node_num))
    for i in range(len(col)):
        A[row[i],col[i]]=1.0
    S=np.dot(inv(np.identity(node_num)-beta*A),beta*A)
    return S,node_num

def RPR(edge_path,alpha):
    '''
    S=(I-alpha*P)^-1 * (1-alpha)* I
    :param edge_path:
    :param alpha:
    :return:
    '''
    node_set = set()
    row = []
    col = []
    G = nx.Graph()
    with open(edge_path) as fp:
        for line in fp.readlines():
            node1 = int(line.strip('\n\r').split()[0])
            node2 = int(line.strip('\n\r').split()[1])
            G.add_edge(node1, node2)
            row.append(node1)
            row.append(node2)
            col.append(node2)
            col.append(node1)
            node_set.add(node1)
            node_set.add(node2)
    node_num = len(node_set)
    print('node num:', node_num, G.number_of_nodes())
    A = np.zeros((node_num, node_num))
    for i in range(len(col)):
        A[row[i], col[i]] = 1.0
    P=np.zeros((node_num,node_num))
    for i in range(node_num):
        row_sum=np.sum(A[i,:])
        P[i,:]=A[i,:]/row_sum
    S=np.dot(inv(np.identity(node_num)-alpha*P),(1-alpha)*np.identity(node_num))
    return S,node_num

def CN(edge_path):
    '''
    S=I^-1 * A^2
    :param edge_path:
    :return: integer similarity
    '''
    node_set = set()
    row = []
    col = []
    G = nx.Graph()
    with open(edge_path) as fp:
        for line in fp.readlines():
            node1 = int(line.strip('\n\r').split()[0])
            node2 = int(line.strip('\n\r').split()[1])
            G.add_edge(node1, node2)
            row.append(node1)
            row.append(node2)
            col.append(node2)
            col.append(node1)
            node_set.add(node1)
            node_set.add(node2)
    node_num = len(node_set)
    print('node num:', node_num, G.number_of_nodes())
    A = np.zeros((node_num, node_num))
    for i in range(len(col)):
        A[row[i], col[i]] = 1
    S=np.dot(inv(np.identity(node_num)),np.dot(A,A))
    return S

def AA(edge_path):
    '''
    S=I^-1 * (A*D*A)
    :param edge_path:
    :return: similarity matrix
    '''
    node_set = set()
    row = []
    col = []
    G = nx.Graph()
    with open(edge_path) as fp:
        for line in fp.readlines():
            node1 = int(line.strip('\n\r').split()[0])
            node2 = int(line.strip('\n\r').split()[1])
            G.add_edge(node1, node2)
            row.append(node1)
            row.append(node2)
            col.append(node2)
            col.append(node1)
            node_set.add(node1)
            node_set.add(node2)
    node_num = len(node_set)
    print('node num:', node_num, G.number_of_nodes())
    A = np.zeros((node_num, node_num))
    for i in range(len(col)):
        A[row[i], col[i]] = 1.0
    D=np.zeros((node_num,node_num))
    for i in range(node_num):
        D[i,i]=0.5/np.sum(A[i,:])
    S=np.dot(inv(np.identity(node_num)),np.dot(np.dot(A,D),A))
    return S,node_num

def PPMI(edge_path,window_size):
    G_dic = {}
    max_node = 0
    with open(edge_path) as f:
        lines = f.readlines()
        print('Edge Number:',len(lines))
        for line in lines:
            items = line.strip('\n').split()
            a = int(items[0])
            b = int(items[1])
            #if a == b:
            #    continue
            max_node = max(max_node, a)
            max_node = max(max_node, b)
            if a in G_dic:
                G_dic[a].append(b)
            else:
                G_dic[a] = [b]
            if b in G_dic:
                G_dic[b].append(a)
            else:
                G_dic[b] = [a]
    G = [[] for _ in range(max_node + 1)]
    for k, v in G_dic.items():
        G[k] = v
    node_num=len(G_dic.items())
    print('Node num:',node_num)
    walk_length=80
    walk_num=20
    walks = []
    for cnt in range(walk_num):
        for node in range(node_num):
            path = [node]
            while len(path) < walk_length:
                cur = path[-1]
                if len(G[cur]) > 0:
                    path.append(random.choice(G[cur]))
                else:
                    break
            walks.append(path)

    vocab = np.zeros(node_num)
    for walk in walks:
        for node in walk:
            vocab[node] += 1
    pair_num_dict={}
    for walk in walks:
        for i in range(len(walk)):
            source_node = walk[i]
            left_window = max(i - window_size, 0)
            right_window = min(i + window_size, len(walk))
            for j in range(left_window, right_window):
                target_node=walk[j]
                if source_node!=target_node:
                    if (source_node,target_node) not in pair_num_dict:
                        pair_num_dict[(source_node,target_node)]=1
                    else:
                        pair_num_dict[(source_node, target_node)] += 1
    PPMI_matrix=np.zeros((node_num,node_num))
    len_D=node_num*walk_length*walk_num
    for key in pair_num_dict:
        node1=key[0]
        node2=key[1]
        co_occurance=pair_num_dict[key]
        frequency_1=vocab[node1]
        frequency_2=vocab[node2]
        res=np.log(1.0*co_occurance*len_D/(frequency_1*frequency_2))
        if res<0:
            res=0
        PPMI_matrix[node1,node2]=res
    return PPMI_matrix,node_num

