#__author__:zhousheng

from __future__ import print_function
import numpy as np
from scipy.special import expit as sigmoid
from copy import deepcopy
from classify import read_node_label, eval, link_cut, link_prediction
from graph_distance import PPMI,jaccard
from sklearn.metrics.pairwise import cosine_similarity
import sys
from scipy.io import mmread
from sklearn.preprocessing import normalize


class Graph():
    def __init__(self, feature_path, edge_path, label_path, embedding_size, lambda_h, lambda_theta_attr,lambda_theta_net, step_size,
                 step_size_theta_attr,step_size_theta_net,feature_sparse):
        self.label_path = label_path
        self.feature_path = feature_path
        self.edge_path = edge_path
        [sim_mat_graph, self.node_num] = jaccard(self.edge_path)
        self.sim_mat_graph=sim_mat_graph
        #self.sim_mat_graph=self.norm_sim_mat(sim_mat_graph,self.node_num)
        print('------Using Jaccard similarity measure------')
        self.feature_sim_mat(feature_sparse)
        self.embedding_size = embedding_size
        self.embedding_mat = np.random.normal(loc=0, scale=0.1, size=(self.node_num, self.embedding_size))
        self.context_mat = np.random.normal(loc=0, scale=0.1, size=(self.node_num, self.embedding_size))
        self.lambda_h = lambda_h
        self.lambda_theta_attr = lambda_theta_attr
        self.lambda_theta_net=lambda_theta_net
        self.step_size = step_size
        self.step_size_theta_attr = step_size_theta_attr
        self.step_size_theta_net=step_size_theta_net
        self.theta_graph = np.mean(self.sim_mat_graph)
        self.theta_attr = np.mean(self.sim_mat_attr)
        print('Theta graph:',self.theta_graph,'Theta attr:',self.theta_attr)
        self.batch_size = 256
        self.b = np.random.normal(loc=0, scale=0.1, size=(1, self.node_num))
        self.loss = 1

    def norm_sim_mat(self,M,node_num):
        MM=deepcopy(M)
        for i in range(node_num):
            MM[i,i]=0
        print('Using L1 norm')
        return normalize(MM,'l1')

    def feature_sim_mat(self,feature_sparse):
        if feature_sparse==False:
            with open(self.feature_path) as fp:
                lines = fp.readlines()
                node_num = len(lines)
                line = lines[0]
                attr_num = len(line.strip('\n\r').split())
            print('Node number:', node_num, 'Attribute dimension:', attr_num)
            self.node_num = node_num
            A = np.zeros((node_num, attr_num))
            with open(self.feature_path) as fp:
                line_num = 0
                for line in fp.readlines():
                    A[line_num, :] = line.strip('\n\r').split()
                    line_num += 1
        else:
            A=mmread(feature_path).todense()
            self.node_num=A.shape[0]
        self.A = A
        A_sim = cosine_similarity(A)
        self.sim_mat_attr=A_sim
        #self.sim_mat_attr = self.norm_sim_mat(A_sim,self.node_num)
        print('Average Attribute Similarity:',np.mean(A_sim))


    def judge_pos_neg(self, M, theta):
        sim_mat = M
        pos_neg_mat = np.zeros((self.node_num, self.node_num))
        pos_neg_mat[sim_mat >= theta] = 1
        pos_neg_mat[sim_mat < theta] = 0
        for i in range(self.node_num):
            pos_neg_mat[i, i] = 666
        return pos_neg_mat

    def final_judge(self):
        pos_neg_mat_graph = self.judge_pos_neg(self.sim_mat_graph, self.theta_graph)
        pos_neg_mat_attr = self.judge_pos_neg(self.sim_mat_attr, self.theta_attr)
        final_pos_neg_mat = np.zeros((self.node_num, self.node_num))
        final_pos_neg_mat[np.where(pos_neg_mat_graph == 1)] += 1
        final_pos_neg_mat[np.where(pos_neg_mat_graph == 0)] -= 1
        final_pos_neg_mat[np.where(pos_neg_mat_attr == 1)] += 1
        final_pos_neg_mat[np.where(pos_neg_mat_attr == 0)] -= 1
        for i in range(self.node_num):
            final_pos_neg_mat[i, i] = 666
        return final_pos_neg_mat

    def sampling(self):
        u_list = range(self.node_num)
        final_pos_neg_mat = self.final_judge()
        P = float(len(np.where(final_pos_neg_mat == 2)[0]))
        A = float(len(np.where(final_pos_neg_mat == 0)[0]))
        N = float(len(np.where(final_pos_neg_mat == -2)[0]))
        sum = P + A + N
        sampled_list = []
        np.random.shuffle(u_list)
        for u in u_list:
            pos_neg_vec = final_pos_neg_mat[u, :]
            p_list = np.where(pos_neg_vec == 2)[0]
            a_list = np.where(pos_neg_vec == 0)[0]
            n_list = np.where(pos_neg_vec == -2)[0]
            # if len(p_list) > 0 and len(a_list) > 0 and len(n_list) > 0:
            #     for i in range(100):
            #         p = np.random.choice(p_list)
            #         a = np.random.choice(a_list)
            #         n = np.random.choice(n_list)
            #         sampled_list.append([u, p, a, n])
            #
            for i in range(20):
                if len(p_list) > 0 and len(a_list) > 0 and len(n_list) > 0:
                    p = np.random.choice(p_list)
                    a = np.random.choice(a_list)
                    n = np.random.choice(n_list)
                    sampled_list.append([u, p, a, n])
                elif len(p_list) > 0 and len(a_list) == 0 and len(n_list) > 0:
                    p=np.random.choice(p_list)
                    a=p
                    n = np.random.choice(n_list)
                    sampled_list.append([u, p, a, n])
                elif len(p_list) == 0 and len(a_list) > 0 and len(n_list) > 0:
                    p = np.random.choice(a_list)
                    a = p
                    n = np.random.choice(n_list)
                    sampled_list.append([u, p, a, n])



        np.random.shuffle(sampled_list)
        self.sampled_list = sampled_list
        print('-----------New Sampling--------')
        print(len(sampled_list), 'triplets sampled')

    def mini_batch(self):
        sampled_list = self.sampled_list
        np.random.shuffle(sampled_list)
        batch_num = len(sampled_list) // self.batch_size
        if len(sampled_list) % self.batch_size == 0:
            for i in range(batch_num):
                yield sampled_list[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            for i in range(batch_num + 1):
                if i < batch_num:
                    yield sampled_list[i * self.batch_size: (i + 1) * self.batch_size]
                else:
                    yield sampled_list[i * self.batch_size:]

    def g_theta(self):
        pos_neg_mat_graph = self.judge_pos_neg(self.sim_mat_graph, self.theta_graph)
        pos_neg_mat_attr = self.judge_pos_neg(self.sim_mat_attr, self.theta_attr)
        if len(np.where(pos_neg_mat_graph == 0)[0]) > 0:
            t_neg_graph = np.mean(self.sim_mat_graph[np.where(pos_neg_mat_graph == 0)])
        else:
            t_neg_graph = 0
            print('T graph negative is 0!!!!')
        self.t_neg_graph = t_neg_graph
        if len(np.where(pos_neg_mat_graph == 1)[0]) > 0:
            t_pos_graph = np.mean(self.sim_mat_graph[np.where(pos_neg_mat_graph == 1)])
        else:
            t_pos_graph = self.theta_graph
            print('Theta graph is setted to 1!!!!!')
        self.t_pos_graph = t_pos_graph
        if len(np.where(pos_neg_mat_attr == 0)[0]) > 0:
            t_neg_attr = np.mean(self.sim_mat_attr[np.where(pos_neg_mat_attr == 0)])
        else:
            t_neg_attr = 0
            print('T attribute negative is 0!!!!')
        self.t_neg_attr = t_neg_attr
        if len(np.where(pos_neg_mat_attr == 1)[0]) > 0:
            t_pos_attr = np.mean(self.sim_mat_attr[np.where(pos_neg_mat_attr == 1)])
        else:
            t_pos_attr = self.theta_attr
            print('Theta attribute is setted to 1!!!!!')
        self.t_pos_attr = t_pos_attr
        return (t_pos_graph - self.theta_graph) * (self.theta_graph - t_neg_graph) + (t_pos_attr - self.theta_attr) * (
                self.theta_attr - t_neg_attr)

    def Estep(self):
        g_theta = self.g_theta()
        H = deepcopy(self.embedding_mat)
        HH = sigmoid(np.dot(H, H.T))
        SHH = HH * (1 - HH)
        grad_mat = np.zeros((self.node_num, self.embedding_size))
        g_theta_plus_one_inv = 1.0 / (g_theta + 1)
        for pair in self.sampled_list:
            u = pair[0]
            p = pair[1]
            a = pair[2]
            n = pair[3]
            h_u = H[u, :]
            h_p = H[p, :]
            h_a = H[a, :]
            h_n = H[n, :]
            up = HH[u, p]
            ua = HH[u, a]
            un = HH[u, n]
            sup = SHH[u, p]
            sua = SHH[u, a]
            sun = SHH[u, n]
            upua = 1 + up - ua
            uaun = 1 + ua - un
            if upua != 0 and uaun != 0:
                grad_mat[u, :] += (sup * h_p - sua*h_a) / upua + (sua*h_a - sun * h_n) / uaun
                grad_mat[p, :] += (sup * h_u) / upua
                grad_mat[a, :] += (-sua*h_u) / upua + (sua*h_u) / uaun
                grad_mat[n, :] += (-sun * h_u) / uaun
        grad_mat *= g_theta_plus_one_inv
        grad_mat -= self.lambda_h * H
        H += self.step_size * grad_mat
        self.embedding_mat = H

    def Mstep(self):
        H = deepcopy(self.embedding_mat)
        HH = sigmoid(np.dot(H, H.T) )
        grad_theta_graph = 0
        grad_theta_attr = 0
        g_theta = self.g_theta()
        for pair in self.sampled_list:
            u = pair[0]
            p = pair[1]
            a = pair[2]
            n = pair[3]
            grad_theta_graph += (self.t_pos_graph + self.t_neg_graph - 2 * self.theta_graph) * (
                    np.log((HH[u, p] - HH[u, a] + 1) / 2) + np.log((HH[u, a] - HH[u, n] + 1) / 2)) * (
                                        -1.0 / np.power(1.0 + g_theta, 2))
            grad_theta_attr += (self.t_pos_attr + self.t_neg_attr - 2 * self.theta_attr) * (
                    np.log((HH[u, p] - HH[u, a] + 1) / 2) + np.log((HH[u, a] - HH[u, n] + 1) / 2)) * (
                                       -1.0 / np.power(1.0 + g_theta, 2))
        grad_theta_graph /= len(self.sampled_list)
        grad_theta_attr /= len(self.sampled_list)
        grad_theta_graph -= self.lambda_theta_net * self.theta_graph
        grad_theta_attr -= self.lambda_theta_attr * self.theta_attr
        self.theta_graph += self.step_size_theta_net * grad_theta_graph
        self.theta_attr += self.step_size_theta_attr * grad_theta_attr
        print( 't_graph:', self.theta_graph, 't_attr:', self.theta_attr)

    def run(self, task):
        for i in range(200):
            self.sampling()
            print(i + 1, 'epoch generated')
            self.Estep()
            self.Mstep()
            self.output(task)
        #np.savetxt('res500.tsv', self.embedding_mat[:500,:], delimiter='\t')
        #np.savetxt('res1000.tsv', self.embedding_mat[:1000, :], delimiter='\t')
        np.savetxt('resall.tsv', self.embedding_mat, delimiter='\t')



    def output(self, task):
        X = self.embedding_mat
        node_num = self.node_num
        if task == 'class':
            Y = read_node_label(self.label_path, node_num)
            eval(X, Y)
        else:
            link_prediction(X, test_path)


if __name__ == '__main__':
    data = 'blogcatalog'#sys.argv[1]
    task = 'class'
    split='1'#sys.argv[1]#str(1)
    print(data,task,split)
    print('PRRE both')
    feature_sparse = True
    edge_path = '../../data/' + data + '/' + data + '.edgelist'
    label_path = '../../data/' + data + '/' + data + '.label'
    if feature_sparse==False:
        feature_path = '../../data/' + data + '/' + data + '.feature'
    else:
        feature_path='../../data/' + data + '/' + data + '_feature.mtx'
    train_path = '../../data/' + data + '/' + data + '.train'+split
    test_path = '../../data/' + data + '/' + data + '.test'+split

    lambda_h =1#float(sys.argv[1])
    lambda_theta_attr = 0
    lambda_theta_net=0
    step_size = 0.1
    step_size_theta_attr = 0.1
    step_size_theta_net=0.1
    print(lambda_h, lambda_theta_attr,lambda_theta_net, step_size, step_size_theta_attr,step_size_theta_net)
    if task == 'class':
        G = Graph(feature_path=feature_path, edge_path=edge_path, label_path=label_path, embedding_size=128,
                  lambda_h=lambda_h, lambda_theta_attr=lambda_theta_attr,lambda_theta_net=lambda_theta_net,
                  step_size=step_size, step_size_theta_attr=step_size_theta_attr,step_size_theta_net=step_size_theta_net,feature_sparse=feature_sparse)
        G.run(task)
    else:
        G = Graph(feature_path=feature_path, edge_path=train_path, label_path=None, embedding_size=128,
                  lambda_h=lambda_h, lambda_theta_attr=lambda_theta_attr,lambda_theta_net=lambda_theta_net,
                  step_size=step_size, step_size_theta_attr=step_size_theta_attr,step_size_theta_net=step_size_theta_net,feature_sparse=feature_sparse)
        G.run(task)
