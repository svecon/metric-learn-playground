from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sortedTechniques = [
    ('kNN', 'Euclidean'),
    ('stand+kNN', 's:Euclidean'),
    
    ('Cov+kNN', 'Covariance'),
    ('stand+Cov+kNN', 's:Covariance'),
    
    ('LMNN+kNN', 'LMNN'),
    ('stand+LMNN+kNN', 's:LMNN'),
    
    ('NCA+kNN', 'NCA'),
    ('stand+NCA+kNN', 's:NCA'),
    
    ('LFDA+kNN', 'LFDA'),
    ('stand+LFDA+kNN', 's:LFDA'),
    
    ('CMAES+kNN', 'CMAES.kNN'),
    ('stand+CMAES+kNN', 's:CMAES.kNN'),
    
    ('CMAESFme+kNN', 'CMAES.fMe'),
    ('stand+CMAESFme+kNN', 's:CMAES.fMe'),
    
    ('JDE+kNN', 'jDE.fMe'),
    ('stand+JDE+kNN', 's:jDE.fMe'),

    ('JDEkNN+kNN', 'jDE.kNN'),
    ('stand+JDEkNN+kNN', 's:jDE.kNN'),

#     ('JDEPur+kNN', 'jDE Pur'),
#     ('stand+JDEPur+kNN', 's:jDE.Pur'),
    
#     ('ITML+kNN', 'ITML'),
#     ('stand+ITML+kNN', 's:ITML'),
    
#     ('LSML+kNN', 'LSML'),
#     ('stand+LSML+kNN', 's:LSML'),
    
#     ('RCA+kNN', 'RCA'),
#     ('stand+RCA+kNN', 's:RCA'),
    
#     ('SDML+kNN', 'SDML'),
#     ('stand+SDML+kNN', 's:SDML'),
]

sortedDatasets = [
    ('balance-scale', 'balance-scale'),
    ('breast-cancer-wisconsin', 'breast-cancer'),
    ('digits6', 'digits6'),
    ('digits10', 'digits10'),
    ('gaussians', 'gaussians'),
    ('iris', 'iris'),
    ('mice-protein', 'mice-protein'),
    ('pima-indians-diabetes', 'pima-indians'),
    ('sonar', 'sonar'),
    ('wine', 'wine'),
]
