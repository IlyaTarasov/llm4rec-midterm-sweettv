"""Nonnegative Poisson/KL count factorization with sparse multiplicative updates."""
import time
import numpy as np
from scipy import sparse

class PoissonMF:
    def fit(self,X,k=8,l1=0.,iterations=100,seed=42,budget_seconds=90):
        started=time.perf_counter();X=sparse.coo_matrix(X,dtype=float);X.sum_duplicates()
        if np.any(X.data<0):raise ValueError('counts must be nonnegative')
        active_u=np.unique(X.row[X.data>0]);active_i=np.unique(X.col[X.data>0])
        rng=np.random.default_rng(seed);scale=np.sqrt(max(X.sum(),1)/(max(len(active_u)*len(active_i)*k,1)))
        self.U=np.zeros((X.shape[0],k));self.V=np.zeros((X.shape[1],k))
        self.U[active_u]=rng.uniform(.5,1.5,(len(active_u),k))*scale
        self.V[active_i]=rng.uniform(.5,1.5,(len(active_i),k))*scale
        self.losses=[]
        def ratio():
            prediction=np.maximum(np.einsum('ij,ij->i',self.U[X.row],self.V[X.col]),1e-12)
            return sparse.csr_matrix((X.data/prediction,(X.row,X.col)),shape=X.shape)
        for iteration in range(iterations):
            if time.perf_counter()-started>budget_seconds:raise TimeoutError('declared fit budget exceeded')
            self.U*=ratio().dot(self.V)/np.maximum(self.V.sum(axis=0)[None,:]+l1,1e-12)
            self.V*=ratio().T.dot(self.U)/np.maximum(self.U.sum(axis=0)[None,:]+l1,1e-12)
            pred=np.maximum(np.einsum('ij,ij->i',self.U[X.row],self.V[X.col]),1e-12)
            loss=float(self.U.sum(axis=0).dot(self.V.sum(axis=0))-X.data.dot(np.log(pred))+l1*(self.U.sum()+self.V.sum()))
            if not np.isfinite(loss):raise RuntimeError('nonfinite Poisson objective')
            self.losses.append(loss)
        return self
    def score(self,users,items):return self.U[users]@self.V[items].T
