import sys
import os
from pathlib import Path

CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture  
from munkres import Munkres, make_cost_matrix
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import random
from tqdm import tqdm

def set_global_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def get_label_dict(tru:list,pred:list):
    """Because cluster id is different from class id, thus we need to use 
    Munkres algorithm to find the max match between the ground truth and predicted labels

    Parameters
    ----------
    tru : ground truth

    pred : predicted labels

    Returns
    -------
    label_dict : the dict that converts class id to cluster id

    Examples
    --------
    >>> y_true = [0, 1, 2, 3] # cluster id
    >>> y_pred = [2, 3, 1, 0] # class id
    >>> get_label_dict(y_true, y_pred)
    {0: 3, 1: 2, 2: 0, 3: 1}
    """

    cnt= max(len(set(tru)),len(set(pred)))
    per_conf=np.zeros([cnt,cnt],dtype=int)
    for i,j in zip(pred,tru):
        per_conf[i][j]+=1
    maxsize=max(max(i) for i in per_conf)
    cost_matrix = []
    for row in per_conf:
        cost_row = []
        for col in row:
            cost_row += [maxsize - col]
        cost_matrix+= [cost_row]
    m = Munkres()
    indexes = m.compute(cost_matrix)
    total=0
    label_dict={}
    for row, column in indexes:
        value = per_conf[row][column]
        label_dict[row]=column 
        total += value
    y_pre_trans=[]
    for i in pred:
        y_pre_trans.append(label_dict[i])
    return label_dict 


def cal_k_star(means:np.ndarray):
    """solve k*(e) in equal_5  
    """
    assert means.shape[0]==means.shape[1] and means.ndim==2 
    cost_matrix = make_cost_matrix(means, lambda x :  - x) 
    m = Munkres()
    indexes = m.compute(cost_matrix)
    k_star = np.array([column for _, column in indexes])
    return k_star

def CLA(means, k_star):
    N = means.shape[0]
    return sum([means[i][k_star[i]] for i in range(N)])


def PROCA(logits):
    N_feature = logits.shape[1]  # the number of classes

    ## equation 3
    logits = np.exp(logits)
    logits = logits/np.sum(logits,axis=-1,keepdims=True)
    distribution = np.log(logits)
    ##
    seed_start = 1600 
    models = [] # the GMM model of different seeds
    max_index = -1
    current_max_cla = -1*(1e10)
    N_iter = 100  # repeat N_iter(100) times (paragraph 3, Section 4.1)
    for idx,seed in tqdm(enumerate(range(seed_start,N_iter+seed_start)), total=N_iter):
        set_global_random_seed(seed)

        ## perform GMM
        model = GaussianMixture(n_components=N_feature,max_iter=100, tol=1e-3, init_params="kmeans")
        model.fit(distribution)
        ##

        means = model.means_  # μ in the paper (paragraph 1, Section 3.2)
        k_star = cal_k_star(means) # solve k*(e)
        cla = CLA(means, k_star) # The CLA of k*(e)
        models.append({
                'model':model,
                'cla':cla,
                'k_star':k_star
            })
        if cla>current_max_cla:
            current_max_cla = cla
            max_index = idx

    # To Select model according to CLA of different seeds
    target = models[max_index]
    target['model'].weights_ = np.array([1.0 for _ in range(len(target['model'].weights_))])  # Set alpha to one, discarding the mixing coefficient(Section 3.4)
    y_pred = target['model'].fit_predict(distribution)
    predicted_labels = np.array([target['k_star'][n_] for n_ in y_pred]) # k*˜n(e*)  (paragraph 1, Section 3.4)
    return predicted_labels


if __name__=="__main__":
    """Test PROCA"""
    # Since PROCA is a clustering model, we generate cluster data to test it.
    centers = [[0,1,0], [0, -1,0],[1,0,0]] # the center of 3 clusters
    logits, gt = make_blobs(n_samples=300, centers=centers, cluster_std=0.2, random_state=0)
    labels = PROCA(logits)

    # visualize PROCA's prediction
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    colors = {0:"red",1:"green",2:"blue"}
    for xn,lb in zip(logits,labels):
        x,y,z = xn
        ax.scatter(x, y, x, c=colors[lb])
    ax.view_init(60, 60)

    # converts class id to cluster id
    label_dict = get_label_dict(gt.tolist(),labels.tolist())
    labels = [label_dict[item] for item in labels]
    f1 = f1_score(gt, labels, average = 'macro')    
    acc = accuracy_score(gt, labels)
    info = "Macro_F1 : {:.4f}, acc= {:.4f}".format(f1,acc)
    print(info) 
    plt.title(info)
    plt.savefig("test.png")

