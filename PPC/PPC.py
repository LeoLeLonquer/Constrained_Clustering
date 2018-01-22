from config import setup
setup()


import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances


from docplex.cp.model import CpoModel


def create_csp_clustering(data,ml,cl,k): #P is the set of data and constraints
    mdl = CpoModel(name='Clustering constraints')
    print("Calculating distances...")
    Dist = pairwise_distances(data)

    k_min = k
    k_max = k
    N = len(data)

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


def call_CSP(data,ml,cl,k):
        mdl,lbls = create_csp_clustering(data,ml,cl,k)
        sol=mdl.solve()
        Attrib=[sol.get_value(lbl) for lbl in lbls]
        score=sol.get_objective_values()
        tt=sol.get_solve_time()
        return tt,Attrib