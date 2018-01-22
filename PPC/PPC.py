from config import setup
setup()


import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances
Data, y = make_blobs(n_samples=100, centers=3, n_features=2,random_state=0)
ml=[(1,5),(5,6)]
cl=[(1,3),(4,7)]
print(Data[:5])


from docplex.cp.model import CpoModel


def create_csp_clustering(P): #P is the set of data and constraints
    mdl = CpoModel(name='Clustering constraints')
    print("Calculating distances...")
    Dist = pairwise_distances(P)

    k_min = 3
    k_max = 3
    N = len(P)

    #Creating vars

    k=mdl.integer_var(k_min,k_max,"k")
    # D=mdl.float_var(min(Dist),+inf)
    # S=mdl.float_var(-inf, max(Dist))
    # V=mdl.float_var(0,+inf)
    Labels = mdl.integer_var_list(N, 1, k_max, "x") # xi = n if xi is in cluster n

    #Partition constraints

    ##k must grow with the number of clusters
    for i in range(N):
        mdl.add(Labels[i] <= k)

    ##break cluster symmetry by forcing order(Use precede if found)
    mdl.add(Labels[0] == 1) # First cluster is labbelled 1
    for i in range(1,N):
        mdl.add(Labels[i] <= mdl.max(Labels[:i]) + 1)


    #User constraints
    ##Must-link
    #for i in range(N):
    #    for j in range(N):
    #        if Mustlink[i,j]:
    #            mdl.add(Labels[i]==Labels[j])
    for i,j in ml:
         mdl.add(Labels[i]==Labels[j])
    # mdl.add(D > Dist(i,j))
    ##Cannot-link
    #for i in range(N):
    #    for j in range(N):
    #        if Cannotlink[i,j]:
    #            mdl.add(Labels[i]!=Labels[j])
    for i,j in cl:
         mdl.add(Labels[i]!=Labels[j])
    # mdl.add(S < Dist(i,j))

    #Optimization constraint
    # mdl.add(D==diameter(Labels,Dists))
    # mdl.add(S==split(Labels,Dists))
    # mdl.add(V==wcsd(Labels,Dists))


    #D diameter
    mdl.add(mdl.minimize( \
        mdl.max([(Labels[i] == Labels[j]) * Dist[i,j] for i in range(N) for j in range(N)])))

    # mdl.add(mdl.maximize(score))
    print("Done.")
    return mdl,Labels

mdl,lbls = create_csp_clustering(Data)

#print(mdl.propagate())
print(mdl.solve())
sol=mdl.solve()
Attrib=[sol.get_value(lbl) for lbl in lbls]

import collections

def cons_linking(cons):
    dico = collections.defaultdict(set) # pt => list of pts
    # linked = collections.defaultdict(set) # nb => list of pt
    asso = dict() # pt => nb
    for pt1, pt2 in cons:
        dico[pt1].add(pt2)
        dico[pt1].add(pt1)
        dico[pt2].add(pt1)
        dico[pt2].add(pt2)
        asso[pt1] = pt1
        asso[pt2] = pt2

    pts = set(dico.keys())
    for pt in pts:
        for pt2 in pts :
            # if the intersection is not empty
            if asso[pt] != asso[pt2] \
            and not not dico[asso[pt]].intersection(dico[asso[pt2]]):
                dico[asso[pt]].update(dico[asso[pt2]])
                asso[pt2] = pt
                del dico[pt2]

    return list(dico.values())

new_cons = cons_linking(ml)
# Plotting the constraints
# Plotting the equality constraints
for cons in new_cons:
    new_X_x = []
    new_X_y = []
    for pt in cons :
        new_X_x.append(Data[pt, 0])
        new_X_y.append(Data[pt, 1])

    plt.plot(new_X_x,new_X_y,
             'r^', linewidth=50)

#plotting Cannnotlink
for pt1, pt2 in cl:
    plt.plot([Data[pt1,0],Data[pt2,0]],
             [Data[pt1,1],Data[pt2,1]],
             'x', linewidth=50)

#plotting Mustlink (done above)
#for pt1, pt2 in ml:
#    plt.plot([Data[pt1,0],Data[pt2,0]],
#             [Data[pt1,1],Data[pt2,1]],
#             '^', linewidth=50)

plt.scatter(Data[:,0],Data[:,1],c=Attrib)
plt.title("Attributions:")
plt.show()
