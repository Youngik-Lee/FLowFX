import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import imageio
from fx_flow_model import build_country_graph, compute_flows
from fx_utils import fetch_rates_yahoo, CURRENCIES  # import currencies and fetch function
def draw_frame(G, flow, pos, filename):
    
    plt.figure(figsize=(10,7))
    nx.draw_networkx_nodes(G,pos,node_size=1500,node_color=(flow-1),cmap="coolwarm")
    nx.draw_networkx_labels(G,pos)
    for u,v in G.edges():
        fu = flow[CURRENCIES.index(u)]
        fv = flow[CURRENCIES.index(v)]
        if fu < fv:
            start,end=u,v; mag=fv-fu
        else:
            start,end=v,u; mag=fu-fv

        nx.draw_networkx_edges(
            G,pos,edgelist=[(start,end)],width=3*mag,
            arrows=True  # remove arrowstyle warning
        )
    plt.axis("off")
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    G = build_country_graph()
    rates = fetch_rates_yahoo(CURRENCIES)  # use the function from fx_utils
    flows = compute_flows(rates)

    pos = nx.spring_layout(G, seed=2)

    frames=[]
    for i in range(len(flows)):
        f = flows.iloc[i].values
        fname = f"frame_{i}.png"
        draw_frame(G,f,pos,fname)
        frames.append(imageio.imread(fname))

    imageio.mimsave("flow_animation.gif", frames, fps=2)
    print("Saved animation: flow_animation.gif")
