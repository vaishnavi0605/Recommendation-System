import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time

dataset=pd.read_csv('data.csv')
G=nx.DiGraph()
for index, row in dataset.iterrows():
    for Impression in dataset.columns[1:]:
        if row[Impression]!=' ':
            G.add_edge(row['Email Address'],row[Impression])

#find the adjacency matrix of the graph and maintain a dictionary of nodes and their indexes
nodes={}
index=0
for i in G.nodes():
    nodes[i]=int(index)
    index+=1
#print(nodes)
#create an adjacency matrix of the graph with nodes in ascending order of their indexes
adjacency_matrix=np.zeros((143,143))
for i in G.edges():
    adjacency_matrix[nodes[i[0]],nodes[i[1]]]=1

for i in range(143):
    for j in range(143):
        if adjacency_matrix[i,j]==1 and adjacency_matrix[j,i]==0:
            adjacency_matrix[j,i]= -1
np.savetxt("adjacency_matrix.csv", adjacency_matrix, delimiter=",", fmt='%d')
#print(adjacency_matrix)

#nx.draw(G, with_labels=True, node_size=50, node_color="skyblue", pos=nx.spring_layout(G))
#plt.show()

def missing_links(adjacency_matrix,learning_rate,regularization_parameter,epochs,k):
    #now i will be factorizing the adjacency matrix into two matrices of order 143*2 and 2*143
    #let them be U(143*2) and V(2*143)
    #initializing U and V with random values between 0 and 1
    U = np.random.rand(adjacency_matrix.shape[0],k)
    V = np.random.rand(k,adjacency_matrix.shape[1])
    #iterating over the epochs
    print("Finding Missing links ...Wait till the timer reaches 1000")
    for epoch in range(epochs):
        #now multiply these to get a predicted matrix P of order 143*143
        P = np.dot(U,V)
        #now we will compare the predicted matrix with the adjacency matrix
        #we will be comparing element wise and updating the U and V matrices
        #we will be using gradient descent to update the matrices
        for row in range(adjacency_matrix.shape[0]):
            for col in range(adjacency_matrix.shape[1]):
                if adjacency_matrix[row,col]!=0:
                    #finding the error
                    error=(adjacency_matrix[row,col]-P[row,col])
                    for i in range(k):
                        #finding the gradient
                        gradient_U= -2*(error)*V[i,col]
                        gradient_V= -2*(error)*U[row,i]
                        #updating the matrices along with the regularizaton parameter
                        U[row,i]=U[row,i]-learning_rate*(gradient_U+regularization_parameter*U[row,i])
                        V[i,col]=V[i,col]-learning_rate*(gradient_V+regularization_parameter*V[i,col])

        print(f'Timer: {epoch}',end='\r')
        time.sleep(0.1)
    P=np.dot(U,V)

    #applying a threshold of 0.8 for making a link in graph
    #iterate over  and convert all entries greater than 0.8 to 1 and else to 0
    count=0
    for row in range(143):
        for col in range(143):
            if P[row,col]>0.8:
                P[row,col]=1
                count+=1
                
            else:
                P[row,col]=0
    print(f'The number of links in the predicted matrix are: {count-3480}')
    return P

P=missing_links(adjacency_matrix,0.001,0.001,1000,50)
np.savetxt("predicted_matrix.csv", P, delimiter=",", fmt='%f')

#finding the RMSE between the predicted matrix and the adjacency matrix
def RMSE(adjacency_matrix,P):
    RMSE=0
    count=0
    for row in range(adjacency_matrix.shape[0]):
        for col in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[row,col]!=0:
                RMSE+=(adjacency_matrix[row,col]-P[row,col])**2
                count+=1
    RMSE=np.sqrt(RMSE/(count))
    return RMSE
#RMSE=RMSE(adjacency_matrix,P)

#now use this predicted matrix and make the graph with nodes corresponding to index stored in the nodes dictionary
def graph_from_matrix(P):
    G_predicted=nx.DiGraph()
    for row in range(143):
        for col in range(143):
            if P[row,col]==1:
                G_predicted.add_edge(list(nodes.keys())[list(nodes.values()).index(row)],list(nodes.keys())[list(nodes.values()).index(col)])
    np.savetxt("predicted_matrix.csv", P, delimiter=",", fmt='%f')
    
    return G_predicted
G_predicted=graph_from_matrix(P)
#print(pagerank(G_predicted,1000000))
nx.draw(G_predicted, with_labels=True, node_size=50, node_color="skyblue", pos=nx.spring_layout(G_predicted))
plt.show()
#print each node and its number of inlinks in the G and G_predicted graph
#for i in G.nodes():
#    print(i,len(list(G.in_edges(i))),len(list(G_predicted.in_edges(i))))
