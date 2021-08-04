# %cd code
# %% a
# %%

# %load_ext autoreload
# %autoreload 2
# %cd code
import numpy as np

from primitives import value_iter, visitation_iter, Reward, fwd
from environs import Env#, Ebigd
from rv import Variable as Var

# E = Ebigd
E = Env.generate(Var.alph("A", 3), Var.alph("S", 8), spec="random.det")

R0 = np.zeros(E.Sshape)
R0[5] = 1
V0 = value_iter(E, R0, 0.99)


# time to visualize
import itertools
import networkx as nx

#%#% One way to create it is by adding edges alltogether.
G = nx.DiGraph()
G.add_nodes_from(E.S)
G.add_edges_from([(s,t,{'A':a}) for (s,a),t in E.T.idxmax(axis=1).items()])

# way 2: this is equivalent.
"""
for s, a in itertools.product(E.S, E.A):
    sprime = E.T.loc[[(s, a)]].idxmax(axis=1).values.item()
    G.add_edge(s, sprime, A=a)
"""
# list(enumerate(E.S.ordered))
#%#% Another way is to directly create a graph for each action
AGs = {
    a: nx.relabel_nodes(
        nx.from_numpy_array(E.TT[:, i, :], create_using=nx.DiGraph),
        {i: n for i, n in enumerate(E.S.ordered)},
    )
    for i, a in enumerate(E.A.ordered)
}
UAGs = nx.DiGraph()
UAGs.add_edges_from(
    list(
        (s,t, {"A": a}) for a, gr in AGs.items() for s, t in gr.edges()
    )
)
# list(('s%d'%s,'s%d'%t,{'A':a}) for a,gr in AGs.items()  for s,t in gr.edges())

# nx.drawing.draw_networkx(AGs['a0'])
len(set(UAGs.edges.data("A")) - set(G.edges.data("A")))
set(UAGs.edges.data("A")) & set(G.edges.data("A"))

len(sorted(UAGs.edges.data("A")))
len(sorted(G.edges.data("A")))


Q = [(s,t,a) for (s,a), t in E.T.idxmax(axis=1).items()]
len(Q)

set(Q) - set(UAGs.edges.data("A"))
len(E.T.idxmax(axis=1))
sorted(UAGs.edges.data("A"))
sorted(G.edges.data("A"))


nx.drawing.draw_networkx(UAGs)
nx.drawing.draw_networkx(G)


# drawing
# G.edge
from matplotlib import pyplot as plt


fig, ax = plt.subplots(figsize=(12, 8))
# pos = nx.spring_layout(G)
pos = nx.kamada_kawai_layout(G)
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=550)
nx.draw_networkx_labels(G, pos, font_color='white')

for a, c in zip(E.A, ["r", "b", "g", "y", "k"]):
    nx.draw_networkx_edges(AGs[a], pos, width=2, edge_color=c,
        arrowstyle="-|>",arrowsize=20,arrows=True,
        ax=ax, alpha=0.5, node_size=650)
# nx.draw_networkx_edges(G, pos, edgelist=[(u,v) for (u,v,a) in G.edges], width=4.0, alpha=0.5)
plt.show()
nx.draw(G)


# Good.
np.allclose(V0.T, V0.reshape(1, 1, -1))

R = Reward(R0)
πopt = fwd(E, R0, 0.9)
πrand = E.random_policy()

D = visitation_iter(E, πopt)
Drand = visitation_iter(E, πrand, None, 1000)


# These numbers are typical.
R.value_variance(D)
R.value_variance(Drand)


R.diff(E)
