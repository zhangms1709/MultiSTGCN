import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.cluster import KMeans
import torch.optim as optim
from random import shuffle
import pandas as pd
import numpy as np
import scanpy as sc
from . layers import GraphConvolution
from scipy.stats import wasserstein_distance
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scanpy as sc
from .layers import GraphConvolution
from sklearn.cluster import KMeans

class Generator(nn.Module):
    def __init__(self, latent_dim, mapping_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        ) # Where is the GCN?

    def forward(self, x):
        img = self.model(x)
        img = img.view(img.shape[0], self.img_shape)
        return img

    # def forward(self, x):
    #     x = torch.relu(self.fc1(x))
    #     embeddings = torch.tanh(x)  # Low-dimensional embeddings
    #     mapping_matrix = torch.sigmoid(self.fc2(embeddings))  # Mapping matrix
    #     return embeddings, mapping_matrix

class Discriminator(nn.Module):
    def __init__(self, input_dim, latent_dim, img_shape):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

class simple_GC_DEC(nn.Module):
    def __init__(self, nfeat, nhid, output_dim, mapping_dim, alpha=0.2):
        super(simple_GC_DEC, self).__init__()
        self.generator = Generator(nfeat, output_dim, mapping_dim)
        self.discriminator = Discriminator(nfeat)
        self.nhid = nhid
        self.alpha = alpha

    def forward(self, x1, x2, adj1, adj2):
        # Forward pass through generator and discriminator
        embeddings1, mapping_matrix1 = self.generator(x1)
        embeddings2, mapping_matrix2 = self.generator(x2)
        # Pass embeddings through discriminator (not shown here)
        return embeddings1, embeddings2, mapping_matrix1, mapping_matrix2

    def loss_function(self, embeddings1, embeddings2, mapping_matrix1, mapping_matrix2):
        # Calculate the loss function (e.g., Wasserstein distance)
        return -torch.mean(embeddings1) + torch.mean(embeddings2)

    def fit(self, X,adj, X2, adj2, lr=0.001, max_epochs=5000, update_interval=3, trajectory_interval=50,weight_decay=5e-4,opt="sgd",init="louvain",n_neighbors=10,res=0.4,n_clusters=10,init_spa=True,tol=1e-3):
        self.trajectory=[]
        if opt=="sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt=="admin":
            optimizer = optim.Adam(self.parameters(),lr=lr, weight_decay=weight_decay)

        features= self.gc(torch.FloatTensor(X),torch.FloatTensor(adj))
        #----------------------------------------------------------------        
        if init=="kmeans":
            print("Initializing cluster centers with kmeans, n_clusters known")
            self.n_clusters=n_clusters
            kmeans = KMeans(self.n_clusters, n_init=20)
            if init_spa:
                #------Kmeans use exp and spatial
                y_pred = kmeans.fit_predict(features.detach().numpy())
            else:
                #------Kmeans only use exp info, no spatial
                y_pred = kmeans.fit_predict(X)  #Here we use X as numpy
        elif init=="louvain":
            print("Initializing cluster centers with louvain, resolution = ", res)
            if init_spa:
                adata=sc.AnnData(features.detach().numpy())
            else:
                adata=sc.AnnData(X)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)
            sc.tl.louvain(adata,resolution=res)
            y_pred=adata.obs['louvain'].astype(int).to_numpy()
            self.n_clusters=len(np.unique(y_pred))
        #----------------------------------------------------------------
        y_pred_last = y_pred
        self.mu = Parameter(torch.Tensor(self.n_clusters, self.nhid))
        X=torch.FloatTensor(X)
        adj=torch.FloatTensor(adj)
        self.trajectory.append(y_pred)
        features=pd.DataFrame(features.detach().numpy(),index=np.arange(0,features.shape[0]))
        Group=pd.Series(y_pred,index=np.arange(0,features.shape[0]),name="Group")
        Mergefeature=pd.concat([features,Group],axis=1)
        cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
        
        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        for epoch in range(max_epochs):
            if epoch%update_interval == 0:
                _, q = self.forward(X,adj)
                p = self.target_distribution(q).data
            if epoch%10==0:
                print("Epoch ", epoch) 
            optimizer.zero_grad()
            z,q = self(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()
            if epoch%trajectory_interval == 0:
                self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())

            #Check stop criterion
            y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / X.shape[0]
            y_pred_last = y_pred
            if epoch>0 and (epoch-1)%update_interval == 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print("Reach tolerance threshold. Stopping training.")
                print("Total epoch:", epoch)
                break

    def fit_with_init(self, X,adj, X2, adj2, init_y, lr=0.001, max_epochs=5000, update_interval=1, weight_decay=5e-4,opt="sgd"):
        print("Initializing cluster centers with kmeans.")
        if opt=="sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt=="admin":
            optimizer = optim.Adam(self.parameters(),lr=lr, weight_decay=weight_decay)
        X=torch.FloatTensor(X)
        adj=torch.FloatTensor(adj)
        features, _ = self.forward(X,adj)
        features=pd.DataFrame(features.detach().numpy(),index=np.arange(0,features.shape[0]))
        Group=pd.Series(init_y,index=np.arange(0,features.shape[0]),name="Group")
        Mergefeature=pd.concat([features,Group],axis=1)
        cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        for epoch in range(max_epochs):
            if epoch%update_interval == 0:
                _, q = self.forward(torch.FloatTensor(X),torch.FloatTensor(adj))
                p = self.target_distribution(q).data
            X=torch.FloatTensor(X)
            adj=torch.FloatTensor(adj)
            optimizer.zero_grad()
            z,q = self(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()

    def predict(self, X, adj):
        # Generate low-dimensional embeddings and mapping matrix
        embeddings, mapping_matrix = self.generator(X)
        return embeddings, mapping_matrix

# class simple_GC_DEC(nn.Module):
#     def __init__(self, nfeat, nhid, alpha=0.2):
#         super(simple_GC_DEC, self).__init__()
#         self.gc = GraphConvolution(nfeat, nhid)
#         self.nhid=nhid
#         #self.mu determined by the init method
#         self.alpha=alpha

#     def forward(self, x, adj):
#         x=self.gc(x, adj)
#         q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha) + 1e-8)
#         q = q**(self.alpha+1.0)/2.0
#         q = q / torch.sum(q, dim=1, keepdim=True)
#         print("x: ", x)
#         print("q: ", q)
#         return x, q

#     def loss_function(self, p, q):
#         def kld(target, pred):
#             return -torch.mean(target) + torch.mean(pred)
#         #wasserstein_distance(target.detach().numpy(), pred.detach().numpy()) #torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
#         loss = kld(p, q)
#         print("l (non-WGAN): ", loss)
#         return loss

#     def target_distribution(self, q):
#         #weight = q ** 2 / q.sum(0)
#         #return torch.transpose((torch.transpose(weight,0,1) / weight.sum(1)),0,1)e
#         p = q**2 / torch.sum(q, dim=0)
#         p = p / torch.sum(p, dim=1, keepdim=True)
#         print("p: ", p)
#         return p

#     def fit(self, X,adj,  lr=0.001, max_epochs=5000, update_interval=3, trajectory_interval=50,weight_decay=5e-4,opt="sgd",init="louvain",n_neighbors=10,res=0.4,n_clusters=10,init_spa=True,tol=1e-3):
#         self.trajectory=[]
#         if opt=="sgd":
#             optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
#         elif opt=="admin":
#             optimizer = optim.Adam(self.parameters(),lr=lr, weight_decay=weight_decay)

#         features= self.gc(torch.FloatTensor(X),torch.FloatTensor(adj))
#         #----------------------------------------------------------------        
#         if init=="kmeans":
#             print("Initializing cluster centers with kmeans, n_clusters known")
#             self.n_clusters=n_clusters
#             kmeans = KMeans(self.n_clusters, n_init=20)
#             if init_spa:
#                 #------Kmeans use exp and spatial
#                 y_pred = kmeans.fit_predict(features.detach().numpy())
#             else:
#                 #------Kmeans only use exp info, no spatial
#                 y_pred = kmeans.fit_predict(X)  #Here we use X as numpy
#         elif init=="louvain":
#             print("Initializing cluster centers with louvain, resolution = ", res)
#             if init_spa:
#                 adata=sc.AnnData(features.detach().numpy())
#             else:
#                 adata=sc.AnnData(X)
#             sc.pp.neighbors(adata, n_neighbors=n_neighbors)
#             sc.tl.louvain(adata,resolution=res)
#             y_pred=adata.obs['louvain'].astype(int).to_numpy()
#             self.n_clusters=len(np.unique(y_pred))
#         #----------------------------------------------------------------
#         y_pred_last = y_pred
#         self.mu = Parameter(torch.Tensor(self.n_clusters, self.nhid))
#         X=torch.FloatTensor(X)
#         adj=torch.FloatTensor(adj)
#         self.trajectory.append(y_pred)
#         features=pd.DataFrame(features.detach().numpy(),index=np.arange(0,features.shape[0]))
#         Group=pd.Series(y_pred,index=np.arange(0,features.shape[0]),name="Group")
#         Mergefeature=pd.concat([features,Group],axis=1)
#         cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
        
#         self.mu.data.copy_(torch.Tensor(cluster_centers))
#         self.train()
#         for epoch in range(max_epochs):
#             if epoch%update_interval == 0:
#                 _, q = self.forward(X,adj)
#                 p = self.target_distribution(q).data
#             if epoch%10==0:
#                 print("Epoch ", epoch) 
#             optimizer.zero_grad()
#             z,q = self(X, adj)
#             loss = self.loss_function(p, q)
#             loss.backward()
#             optimizer.step()
#             if epoch%trajectory_interval == 0:
#                 self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())

#             #Check stop criterion
#             y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
#             delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / X.shape[0]
#             y_pred_last = y_pred
#             if epoch>0 and (epoch-1)%update_interval == 0 and delta_label < tol:
#                 print('delta_label ', delta_label, '< tol ', tol)
#                 print("Reach tolerance threshold. Stopping training.")
#                 print("Total epoch:", epoch)
#                 break


#     def fit_with_init(self, X,adj, init_y, lr=0.001, max_epochs=5000, update_interval=1, weight_decay=5e-4,opt="sgd"):
#         print("Initializing cluster centers with kmeans.")
#         if opt=="sgd":
#             optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
#         elif opt=="admin":
#             optimizer = optim.Adam(self.parameters(),lr=lr, weight_decay=weight_decay)
#         X=torch.FloatTensor(X)
#         adj=torch.FloatTensor(adj)
#         features, _ = self.forward(X,adj)
#         features=pd.DataFrame(features.detach().numpy(),index=np.arange(0,features.shape[0]))
#         Group=pd.Series(init_y,index=np.arange(0,features.shape[0]),name="Group")
#         Mergefeature=pd.concat([features,Group],axis=1)
#         cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
#         self.mu.data.copy_(torch.Tensor(cluster_centers))
#         self.train()
#         for epoch in range(max_epochs):
#             if epoch%update_interval == 0:
#                 _, q = self.forward(torch.FloatTensor(X),torch.FloatTensor(adj))
#                 p = self.target_distribution(q).data
#             X=torch.FloatTensor(X)
#             adj=torch.FloatTensor(adj)
#             optimizer.zero_grad()
#             z,q = self(X, adj)
#             loss = self.loss_function(p, q)
#             loss.backward()
#             optimizer.step()

#     def predict(self, X, adj):
#         z,q = self(torch.FloatTensor(X),torch.FloatTensor(adj))
#         print("z: ", z)
#         print("q: ", q)
#         return z, q

# class GC_DEC(nn.Module):
#     def __init__(self, nfeat, nhid1,nhid2, n_clusters=None, dropout=0.5,alpha=0.2):
#         super(GC_DEC, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid1)
#         self.gc2 = GraphConvolution(nhid1, nhid2)
#         self.dropout = dropout
#         self.mu = Parameter(torch.Tensor(n_clusters, nhid2))
#         self.n_clusters=n_clusters
#         self.alpha=alpha

#     def forward(self, x, adj):
#         x=self.gc1(x, adj) 
#         x = F.relu(x)
#         x = F.dropout(x, self.dropout, training=True)
#         x = self.gc2(x, adj)
#         q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha) + 1e-6)
#         q = q**(self.alpha+1.0)/2.0
#         q = q / torch.sum(q, dim=1, keepdim=True)
#         print("x: ", x)
#         print("q: ", q)
#         return x, q

# # two slices, wasserstein gan loss, try to add, wgan loss can give you n1 and n2 relationship info, look at wgan paper
# # diff words are diff locations, belong to diff slices. Maybe adding wgan loss. wassestein based distance
#     def loss_function(self, p, q):
#         def kld(target, pred):
#             return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1)) #return wasserstein_distance(target, pred) #torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
#         loss = kld(p, q)
#         print(loss)
#         return loss

#     def target_distribution(self, q):
#         #weight = q ** 2 / q.sum(0)
#         #return torch.transpose((torch.transpose(weight,0,1) / weight.sum(1)),0,1)e
#         p = q**2 / torch.sum(q, dim=0)
#         p = p / torch.sum(p, dim=1, keepdim=True)
#         print("p: ", p)
#         return p

#     def fit(self, X,adj, lr=0.001, max_epochs=10, update_interval=5, weight_decay=5e-4,opt="sgd",init="louvain",n_neighbors=10,res=0.4):
#         self.trajectory=[]
#         print("Initializing cluster centers with kmeans.")
#         if opt=="sgd":
#             optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
#         elif opt=="admin":
#             optimizer = optim.Adam(self.parameters(),lr=lr, weight_decay=weight_decay)

#         features, _ = self.forward(torch.FloatTensor(X),torch.FloatTensor(adj))
#         #----------------------------------------------------------------
        
#         if init=="kmeans":
#             #Kmeans only use exp info, no spatial
#             #kmeans = KMeans(self.n_clusters, n_init=20)
#             #y_pred = kmeans.fit_predict(X)  #Here we use X as numpy
#             #Kmeans use exp and spatial
#             kmeans = KMeans(self.n_clusters, n_init=20)
#             y_pred = kmeans.fit_predict(features.detach().numpy())
#         elif init=="louvain":
#             adata=sc.AnnData(features.detach().numpy())
#             sc.pp.neighbors(adata, n_neighbors=n_neighbors)
#             sc.tl.louvain(adata,resolution=res)
#             y_pred=adata.obs['louvain'].astype(int).to_numpy()
#         #----------------------------------------------------------------
#         X=torch.FloatTensor(X)
#         adj=torch.FloatTensor(adj)
#         self.trajectory.append(y_pred)
#         features=pd.DataFrame(features.detach().numpy(),index=np.arange(0,features.shape[0]))
#         Group=pd.Series(y_pred,index=np.arange(0,features.shape[0]),name="Group")
#         Mergefeature=pd.concat([features,Group],axis=1)
#         cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
        
#         self.mu.data.copy_(torch.Tensor(cluster_centers))
#         self.train()
#         for epoch in range(max_epochs):
#             if epoch%update_interval == 0:
#                 _, q = self.forward(X,adj)
#                 p = self.target_distribution(q).data
#             if epoch%100==0:
#                 print("Epoch ", epoch) 
#             optimizer.zero_grad()
#             z,q = self(X, adj)
#             loss = self.loss_function(p, q)
#             loss.backward()
#             optimizer.step()
#             self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())

#     def fit_with_init(self, X,adj, init_y, lr=0.001, max_epochs=10, update_interval=1, weight_decay=5e-4,opt="sgd"):
#         print("Initializing cluster centers with kmeans.")
#         if opt=="sgd":
#             optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
#         elif opt=="admin":
#             optimizer = optim.Adam(self.parameters(),lr=lr, weight_decay=weight_decay)
#         X=torch.FloatTensor(X)
#         adj=torch.FloatTensor(adj)
#         features, _ = self.forward(X,adj)
#         features=pd.DataFrame(features.detach().numpy(),index=np.arange(0,features.shape[0]))
#         Group=pd.Series(init_y,index=np.arange(0,features.shape[0]),name="Group")
#         Mergefeature=pd.concat([features,Group],axis=1)
#         cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
#         self.mu.data.copy_(torch.Tensor(cluster_centers))
#         self.train()
#         for epoch in range(max_epochs):
#             if epoch%update_interval == 0:
#                 _, q = self.forward(torch.FloatTensor(X),torch.FloatTensor(adj))
#                 p = self.target_distribution(q).data
#             X=torch.FloatTensor(X)
#             adj=torch.FloatTensor(adj)
#             optimizer.zero_grad()
#             z,q = self(X, adj)
#             loss = self.loss_function(p, q)
#             loss.backward()
#             optimizer.step()

#     def predict(self, X, adj):
#         z,q = self(torch.FloatTensor(X),torch.FloatTensor(adj))
#         print("z: ", z)
#         print("q: ", q)
#         return z, q


