# %%
# Create network
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os
import collections
import numpy as np
import re
import os
import glob

plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'

dir1 = glob.glob('./networks/edgelists/*')
dir2 = glob.glob('./networks/hashtag_list/*')

for edgelist,hashtag_text in zip(sorted(dir1),sorted(dir2)):
    name = re.search('ANJA_B_(.*)_combined_edgelist.csv', edgelist)
    name = name.group(1)
    print(f"[INFO] Processing {name}...")
    edgelist = pd.DataFrame(pd.read_csv(edgelist))

    # Here there will not always be a connection to the main hashtag
    high_weight = edgelist.sort_values("count", ascending = False).head(200) 
    # high_weight = edgelist.loc[(edgelist.iloc[:,0]==name) | (edgelist.iloc[:,1]==name)].sort_values("count", ascending = False).head(300)
    test = edgelist.sort_values("count", ascending = False).head(200) 
    mask = pd.DataFrame(np.sort(test[['0','1']], axis=1), index=test.index).duplicated()    
    print(len(test[~mask]))
    
    
    high_weight["count_divided"] = high_weight["count"]/200 # was 200

    # high_weight = high_weight.iloc[:,1:]

    text = pd.read_csv(hashtag_text)
    g_high_weight = nx.from_pandas_edgelist(high_weight, 
                                            source='0', 
                                            target='1', 
                                            edge_attr='count_divided')

    # Count number of tweets hashtag occurs in
    import ast
    h_list = [ast.literal_eval(h) for h in text.hashtags]
    flat_list = [item for sublist in h_list for item in sublist]
    count = collections.Counter(flat_list)
    
    # Getting all highweight nodes
    hmm = high_weight.iloc[:,1]
    hmm = hmm.append(high_weight.iloc[:,2])

    # Taking only high weight nodes
    # nodelist_w_size = {key: count[key] for key in count.keys() & set(high_weight.iloc[:,0])}
    nodelist_w_size = {key: count[key] for key in count.keys() & set(hmm)}

    # Adding attributes
    nx.set_node_attributes(g_high_weight, values = nodelist_w_size, name='size')
    # Edge weights
    widths = nx.get_edge_attributes(g_high_weight, 'count_divided')
    # Node size
    node_size = nx.get_node_attributes(g_high_weight, 'size')
    # Node list
    nodelist = g_high_weight.nodes()

    # Dictionary changes
    # The network needs a list with nodes / edges and an array with values
    # Dividing values with 5 to minimize node size
    size_node_dict = {k:(float(v)/50) for k, v in nodelist_w_size.items()}
 
    key_list_nodes = list(size_node_dict.keys()) 
    size_node_array = np.array(list(size_node_dict.values()))

    widths_dict = {k:(float(v)) for k, v in widths.items()}
    widths_list = list(widths_dict.keys()) 
    widths_node_array = np.array(list(widths_dict.values()))

    fig, ax = plt.subplots(figsize=(40,30))

    # Spring layout - nodes that have less to do with each other are further apart
    pos_new = nx.spring_layout(g_high_weight, 
                                #k=0.7, 
                                #scale=0.05,
                                center = [0.0,  0.0],                            
                                seed = 2021)


    nx.draw_networkx_nodes(g_high_weight,
                            pos = pos_new,#pos_nodes_edges,
                            nodelist = key_list_nodes,
                            node_size = size_node_array,
                            node_color ='grey',
                            ax=ax,
                            alpha = 0.8)

                    
        # Edges
    nx.draw_networkx_edges(g_high_weight,
                            pos = pos_new,#pos_nodes_edges,
                            edgelist = widths_list,
                            width = widths_node_array,
                            edge_color='lightblue',
                            ax=ax,
                            alpha=0.6)

    nx.draw_networkx(g_high_weight, 
                    nodelist=key_list_nodes, 
                    with_labels = False, 
                    node_size=size_node_array, 
                    node_color = 'grey', 
                    edge_color = 'grey',
                    pos=pos_new)

    # Labels
    # Make offset of labels so that they are not in the middle of the node
    pos_attrs = {}
    for node, coords in pos_new.items():
        pos_attrs[node] = (coords[0], coords[1] + 0.02)

    # Add labels to network    
    nx.draw_networkx_labels(g_high_weight, 
                            pos_attrs, 
                            font_size = 25,
                            labels=dict(zip(nodelist, nodelist)), 
                            font_color='black', 
                            ax=ax)

    plt.tight_layout()
    plt.box(False)
    plt.savefig(f"network_individual_weights_{name}.png",dpi=300, bbox_inches="tight")
 #   plt.show()


# %%
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os
import collections
import numpy as np
import re
import os
import glob
edgelist = pd.DataFrame(pd.read_csv("./networks/edgelists/ANJA_B_5g_combined_edgelist.csv"))

# Here there will not always be a connection to the main hashtag
high_weight = edgelist.sort_values("count", ascending = False).head(200) 

high_weight=high_weight[~high_weight['0'].isin(['laptop', 'gb', 'gaming', 'smartphone', 'android'])]
high_weight=high_weight[~high_weight['1'].isin(['laptop', 'gb', 'gaming', 'smartphone', 'android'])]

# high_weight = edgelist.loc[(edgelist.iloc[:,0]==name) | (edgelist.iloc[:,1]==name)].sort_values("count", ascending = False).head(300)

high_weight["count_divided"] = high_weight["count"]/200 # was 200

# high_weight = high_weight.iloc[:,1:]

text = pd.read_csv("./networks/hashtag_list/ANJA_B_5g_combined_hashtag_list.csv")
g_high_weight = nx.from_pandas_edgelist(high_weight, 
                                        source='0', 
                                        target='1', 
                                        edge_attr='count_divided')

# Count number of tweets hashtag occurs in
import ast
h_list = [ast.literal_eval(h) for h in text.hashtags]
flat_list = [item for sublist in h_list for item in sublist]
count = collections.Counter(flat_list)

# Getting all highweight nodes
hmm = high_weight.iloc[:,1]
hmm = hmm.append(high_weight.iloc[:,2])

# Taking only high weight nodes
# nodelist_w_size = {key: count[key] for key in count.keys() & set(high_weight.iloc[:,0])}
nodelist_w_size = {key: count[key] for key in count.keys() & set(hmm)}

# Adding attributes
nx.set_node_attributes(g_high_weight, values = nodelist_w_size, name='size')
# Edge weights
widths = nx.get_edge_attributes(g_high_weight, 'count_divided')
# Node size
node_size = nx.get_node_attributes(g_high_weight, 'size')
# Node list
nodelist = g_high_weight.nodes()

# Dictionary changes
# The network needs a list with nodes / edges and an array with values
# Dividing values with 5 to minimize node size
size_node_dict = {k:(float(v)/50) for k, v in nodelist_w_size.items()}

key_list_nodes = list(size_node_dict.keys()) 
size_node_array = np.array(list(size_node_dict.values()))

widths_dict = {k:(float(v)) for k, v in widths.items()}
widths_list = list(widths_dict.keys()) 
widths_node_array = np.array(list(widths_dict.values()))

fig, ax = plt.subplots(figsize=(40,30))

# Spring layout - nodes that have less to do with each other are further apart
pos_new = nx.spring_layout(g_high_weight, 
                            #k=0.7, 
                            #scale=0.05,
                            center = [0.0,  0.0],                            
                            seed = 2021)


nx.draw_networkx_nodes(g_high_weight,
                        pos = pos_new,#pos_nodes_edges,
                        nodelist = key_list_nodes,
                        node_size = size_node_array,
                        node_color ='grey',
                        ax=ax,
                        alpha = 0.8)

                
    # Edges
nx.draw_networkx_edges(g_high_weight,
                        pos = pos_new,#pos_nodes_edges,
                        edgelist = widths_list,
                        width = widths_node_array,
                        edge_color='lightblue',
                        ax=ax,
                        alpha=0.6)

nx.draw_networkx(g_high_weight, 
                nodelist=key_list_nodes, 
                with_labels = False, 
                node_size=size_node_array, 
                node_color = 'grey', 
                edge_color = 'grey',
                pos=pos_new)

# Labels
# Make offset of labels so that they are not in the middle of the node
pos_attrs = {}
for node, coords in pos_new.items():
    pos_attrs[node] = (coords[0], coords[1] + 0.02)

# Add labels to network    
nx.draw_networkx_labels(g_high_weight, 
                        pos_attrs, 
                        font_size = 25,
                        labels=dict(zip(nodelist, nodelist)), 
                        font_color='black', 
                        ax=ax)

plt.tight_layout()
plt.box(False)
plt.savefig(f"network_individual_weights_5g_nosubnetwork.png",dpi=300, bbox_inches="tight")

# %%
