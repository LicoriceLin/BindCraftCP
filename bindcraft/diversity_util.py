from ._import import *
from itertools import product, combinations
from sklearn.cluster import SpectralClustering

def _simple_identity(x:str,y:str):
    m=0
    for i,j in zip(x,y):
        if i==j:
            m+=1
    return m/len(x)
    
def simple_diversity(df:pd.DataFrame):
    '''
    same-length simple diversity
    '''
    odf=pd.DataFrame(columns=df.index,index=df.index)
    for i, j in combinations(df['seq'].index, 2):
        odf.at[i, j] = odf.at[j, i] = 1/(_simple_identity(df['seq'][i], df['seq'][j])+ 1e-3)
    odf=odf.fillna(1.)
    return odf

def cluster_and_get_medoids(distance_matrix, num_clusters=5):
    spectral = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=42)
    labels = spectral.fit_predict(distance_matrix)

    medoids = []
    for cluster_id in range(num_clusters):
        cluster_points = np.where(labels == cluster_id)[0]  
        # if len(cluster_points) == 0:
        #     continue  

        # cluster_distances = distance_matrix[np.ix_(cluster_points, cluster_points)]
        
        # medoid_index = cluster_points[np.argmin(cluster_distances.sum(axis=1))]
        medoids.append(cluster_points[0])

    return medoids, labels

def mmseqs2_diversity():
    pass

def foldseek_diversity():
    pass
    

