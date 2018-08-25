import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from time import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import roc_auc_score,average_precision_score
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity



# class TopKRanker(OneVsRestClassifier):
#     def predict(self, X, top_k_list):
#         probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
#         all_labels = []
#         for i, k in enumerate(top_k_list):
#             probs_ = probs[i, :]
#             labels = self.classes_[probs_.argsort()[-k:]].tolist()
#             probs_[:] = 0
#             probs_[labels] = 1
#             all_labels.append(probs_)
#         return numpy.asarray(all_labels)
#
#
# class Classifier(object):
#
#     def __init__(self, vectors, clf):
#         self.embeddings = vectors
#         self.clf = TopKRanker(clf)
#         self.binarizer = MultiLabelBinarizer(sparse_output=True)
#
#     def train(self, X, Y, Y_all):
#         self.binarizer.fit(Y_all)
#         X_train = [self.embeddings[x] for x in X]
#         Y = self.binarizer.transform(Y)
#         self.clf.fit(X_train, Y)
#
#     def evaluate(self, X, Y):
#         top_k_list = [len(l) for l in Y]
#         Y_ = self.predict(X, top_k_list)
#         Y = self.binarizer.transform(Y)
#         averages = ["micro", "macro", "samples", "weighted"]
#         results = {}
#         for average in averages:
#             results[average] = f1_score(Y, Y_, average=average)
#         # print 'Results, using embeddings of dimensionality', len(self.embeddings[X[0]])
#         # print '-------------------'
#         print results
#         return results
#         # print '-------------------'
#
#     def predict(self, X, top_k_list):
#         X_ = numpy.asarray([self.embeddings[x] for x in X])
#         Y = self.clf.predict(X_, top_k_list=top_k_list)
#         return Y
#
#     def split_train_evaluate(self, X, Y, train_precent, seed=0):
#         state = numpy.random.get_state()
#
#         training_size = int(train_precent * len(X))
#         numpy.random.seed(seed)
#         shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
#         X_train = [X[shuffle_indices[i]] for i in range(training_size)]
#         Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
#         X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
#         Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]
#
#         self.train(X_train, Y_train, Y)
#         numpy.random.set_state(state)
#         return self.evaluate(X_test, Y_test)



# def load_embeddings(filename):
#     fin = open(filename, 'r')
#     node_num, size = [int(x) for x in fin.readline().strip().split()]
#     vectors = {}
#     while 1:
#         l = fin.readline()
#         if l == '':
#             break
#         vec = l.strip().split(' ')
#         assert len(vec) == size+1
#         vectors[vec[0]] = [float(x) for x in vec[1:]]
#     fin.close()
#     assert len(vectors) == node_num
#     return vectors

# def read_node_label(filename):
#     fin = open(filename, 'r')
#     X = []
#     Y = []
#     while 1:
#         l = fin.readline()
#         if l == '':
#             break
#         vec = l.strip().split(' ')
#         X.append(vec[0])
#         Y.append(vec[1:])
#     fin.close()
#     return X, Y

def load_embeddings(filename):
    fin = open(filename, 'r')
    node_num, size = [int(x) for x in fin.readline().strip().split()]
    print(node_num,size)
    X=np.zeros((node_num,size))
    while 1:
        l = fin.readline()
        if l == '':
            break
        node = int(l.strip('\n\r').split()[0])
        embedding=l.strip('\n\r').split()[1:]
        X[node,:]=[float(x) for x in embedding]
    fin.close()
    return X,node_num

def load_embeddings2(filename):
    fin = open(filename, 'r')
    node_num, size = [int(x) for x in fin.readline().strip().split()]
    X=np.zeros((node_num,size*2))
    while 1:
        l = fin.readline()
        if l == '':
            break
        node = int(l.strip('\n\r').split()[0])
        embedding=l.strip('\n\r').split()[1:]
        X[node,:]=[float(x) for x in embedding]
    fin.close()
    return X,node_num



def read_node_label(filename,node_num):
    Y = np.zeros(node_num)
    label_path = filename
    with open(label_path) as fp:
        for line in fp.readlines():
            node = line.strip('\n\r').split()[0]
            label = line.strip('\n\r').split()[1]
            Y[int(node)] = int(label)
    return Y

def eval(X,Y,train_percent=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=train_percent, test_size=1 - train_percent,random_state=666)
    #clf = SVC(C=20)
    clf=LinearSVC()
    clf.fit(X_train, y_train)
    res = clf.predict(X_test)
    accuracy = accuracy_score(y_test, res)
    macro = f1_score(y_test, res, average='macro')
    micro = f1_score(y_test, res, average='micro')
    print(micro,macro)

def link_cut(edge_path,rate):
    all_edge = []
    node_edge_num_dict = {}
    node_set = set()
    with open(edge_path) as fp:
        for line in fp.readlines():
            node1 = int(line.strip('\n\r').split()[0])
            node2 = int(line.strip('\n\r').split()[1])
            node_set.add(node1)
            node_set.add(node2)
            all_edge.append((node1, node2))
            if node1 not in node_edge_num_dict:
                node_edge_num_dict[node1] = 1
            else:
                node_edge_num_dict[node1] += 1
            if node2 not in node_edge_num_dict:
                node_edge_num_dict[node2] = 1
            else:
                node_edge_num_dict[node2] += 1
    seperatable_edge = []
    for edge in all_edge:
        node1 = edge[0]
        node2 = edge[1]
        if node_edge_num_dict[node1] > 1 and node_edge_num_dict[node2] > 1:
            seperatable_edge.append(edge)
    print('Number of nodes:',len(node_set))
    print('Number of edges:', len(all_edge))
    print('Number of seperatable edges:', len(seperatable_edge))
    test_edges = []
    train_edges = []
    if len(all_edge) * rate > len(seperatable_edge):
        print('Not so many edges to be sampled!')
    else:
        np.random.shuffle(seperatable_edge)
        for i in range(int(len(all_edge) * rate)):
            test_edges.append(seperatable_edge[i])
        for edge in all_edge:
            if edge not in test_edges:
                train_edges.append(edge)
        for i in range(len(node_set)):
            flag=0
            for pair in train_edges:
                if i in pair:
                    flag+=1
            if flag==0:
                train_edges.append((i,i))
        train_set=set()
        with open('training_graph.txt', 'w') as wp:
            for edge in train_edges:
                node1 = edge[0]
                node2 = edge[1]
                train_set.add(node1)
                train_set.add(node2)
                wp.write(str(node1) + '\t' + str(node2) + '\n')
        with open('test_graph.txt', 'w') as wp:
            for edge in test_edges:
                node1 = edge[0]
                node2 = edge[1]
                wp.write(str(node1) + '\t' + str(node2) + '\n')
        print('Training graph node number:',len(train_set))


def link_prediction(embedding,test_path):
    embedding_sim =cosine_similarity(embedding)
    y_predict=[]
    y_gt=[]
    with open(test_path) as fp:
        for line in fp.readlines():
            node1=int(line.strip('\n\r').split()[0])
            node2 = int(line.strip('\n\r').split()[1])
            label=int(line.strip('\n\r').split()[2])
            y_gt.append(label)
            y_predict.append(embedding_sim[node1,node2])
        roc=roc_auc_score(y_gt,y_predict)
        ap=average_precision_score(y_gt,y_predict)
        if roc<0.5:
            roc=1-roc
        print('ROC:',roc,'AP:',ap)