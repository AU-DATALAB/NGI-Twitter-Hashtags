''' 
This script attempts to create a network. This command-line tool will take a given dataset and perform simple network analysis. In particular, it will build networks based on entities appearing together in the same tweets.
'''
import argparse
import pandas as pd
import os
# Basic network without edge weights
import matplotlib as plt
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import re
import collections
import warnings
warnings.filterwarnings("ignore")


# Setting frame so network isn't too small or big
# plt.rcParams["figure.figsize"] = (20,20)


# Function for making a network from the edgelist dataframe
def make_edgelist(hashtag_ds, name, args):
    ''' Create an edgelitst with weights
    Input: filename of dataframe with hashtags, args passed from the terminal
    Outout: preprocessed text and highly weighted edgelist 
    '''
    print("[INFO] Creating edgelist...")
    # Filter english
    hashtag_ds_en=hashtag_ds[hashtag_ds["lang"]=="en"]
    # Filter text
    text = hashtag_ds_en[['text']]
    # Drop duplicates
    text.drop_duplicates(subset=None, keep="first", inplace=True)
    # Lower text
    text['text'] = [row.lower() for row in text['text']]
    # Adding as column in dataframe
    text['hashtags'] = [re.findall(r"#(\w+)", row) for row in text['text']]
    
    # Creating dataframe with 0 and 1 containing a hashtag and its co-mention using itertools
    import itertools
    col1 = []
    col2 = []

    for index, row in text.iterrows():
        hashtags = row['hashtags']
        for n in list(itertools.combinations(hashtags, 2)):
            col1.append(n[0])
            col2.append(n[1])
    df = pd.DataFrame(list(zip(col1, col2)))

    # Creating weighted edgelist when two hashtags are mentioned more frequently together
    weighted = pd.DataFrame({'count' : df.groupby([0, 1]).size()}).reset_index()
   # if len(weighted[weighted["count"]>args["weight_treshold"]])==0:
   #     high_weight = weighted
   #     high_weight["count_divided"] = high_weight["count"]

    # else:
     #   high_weight = weighted[weighted["count"]>args["weight_treshold"]]
    high_weight = weighted.sort_values("count", ascending = False).head(100)
    high_weight["count_divided"] = high_weight["count"]/200
    # Save edgelist
    weighted.to_csv(os.path.join("edgelists",f"{name}_edgelist.csv"))
    high_weight.to_csv(os.path.join("edgelists",f"{name}_weighted_edgelist.csv"))
    text.to_csv(f"{name}_hashtag_list.csv")
    return text, high_weight

def make_network(text, high_weight, args, name):
    '''
    Function that creates a plot of the network and saves it in the specified output folder
    Input: preprocessed text, weighted edgelist, name of hashtag
    '''
    print("[INFO] Creating network...")
    
    g_high_weight = nx.from_pandas_edgelist(high_weight, 
                                            source='0', 
                                            target='1', 
                                            edge_attr='count_divided')

    #pos_new = nx.circular_layout(g_high_weight)


    # Count number of tweets
    t=list(text.hashtags)
    flat_list = [item for sublist in t for item in sublist]
    count = collections.Counter(flat_list)
    
    # Taking only high weight nodes
    nodelist_w_size = {key: count[key] for key in count.keys() & set(high_weight[0])}

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
    #plt.figure(figsize=(40,30))
    # plt.axis("scale")

    # Spring layout - nodes that have less to do with each other are further apart
    pos_new = nx.spring_layout(g_high_weight, 
                                #k=0.7, 
                                #scale=0.05,
                                center = [0.0,  0.0],                            
                                seed = 2021)

    # pos_new = {}
    # for key,value in pos.items():
    #    pos_new[key] = [x/2 for x in value]

    # Nodes
    #pos_nodes_edges = {}
    #for node, coords in pos_new.items():
    #    pos_nodes_edges[node] = np.array([coords[0]/2, coords[1]/2])

    #print(pos_nodes_edges)
    # nx.set_node_attributes(g_high_weight,'coord', pos_nodes)

    nx.draw_networkx_nodes(g_high_weight,
                            pos = pos_new,#pos_nodes_edges,
                            nodelist = key_list_nodes,
                            node_size = size_node_array,
                            node_color ='grey',
                            ax=ax,
                            alpha = 0.8)


    #    nx.set_edge_attributes(g_high_weight,'coord', pos_edges)

        # Edges
    nx.draw_networkx_edges(g_high_weight,
                            pos = pos_new,#pos_nodes_edges,
                            edgelist = widths_list,
                            width = widths_node_array,
                            edge_color='lightblue',
                            ax=ax,
                            alpha=0.6)

    nx.draw_networkx(g_high_weight, with_labels = False, pos=pos_new)

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

    plt.savefig(os.path.join("figs",f"{name}_network.png"),dpi=300, bbox_inches="tight")
    print(f'[INFO] network visualization saved in figs as {name}_network.png')
        
    plt.box(False)
    plt.show()
    
    
    return g_high_weight
    
    
def calc_measures(g_high_weight, args, name):
    '''
    Function providing calculations of eigenvector centrality and betweenness centrality and saves this information in the folder specified with the argument outpath
    Input: network, arguments, name of hashtag
    '''
    print("[INFO] Calcularing centrality measures...")
    # Centrality measures
    # Using nx to calculate eigenvector and betweenness centrality
    ev = nx.eigenvector_centrality(g_high_weight)
    bc = nx.betweenness_centrality(g_high_weight)

    # Converting to dataframe
    d = pd.DataFrame({'eigenvector':ev, 'betweenness':bc})
    d.reset_index(level=0, inplace=True)
    
    # Save as csv
    d.to_csv(os.path.join("centrality", f"{name}_network_info.csv"), index=False)
    print(f'[INFO] Eigenvector and betweeness centrality saved in centrality as network_info.csv')

def process(args):
    
    path = args['path']
    for ds in os.listdir(path):

        if ds.endswith(".csv"):
            basename = os.path.splitext(os.path.basename(ds))[0]

            print(f"Currently processing dataset {basename}")

            tmp_path = os.path.join(path, ds)

            hashtag_ds = pd.read_csv(tmp_path)

            # Run edgelist function
            text, weighted_edgelist = make_edgelist(hashtag_ds, basename, args)

            # Run network function
            network_high_weight = make_network(text, weighted_edgelist, args, basename)

            # Extract centrality measure
            calc_measures(network_high_weight, args, basename)
        
        
def main(): 
    # Add description
    ap = argparse.ArgumentParser(description = "[INFO] creating network pipeline") # Defining an argument parse

    ap.add_argument("-p","--path", 
                    required=False, # As I have provided a default name it is not required
                    type = str, # Str type
                    default = os.path.join("..", "..", "data-twitter", "combined_datasets"), # Setting default to the name of my own edgelist
                    help = "str of filename location")
    
    ap.add_argument("-w","--weight_treshold", 
                    required=False, 
                    type = int, 
                    default = 500, 
                    help = "int of threshold weights")
    
    args = vars(ap.parse_args())
    
    process(args)
    
        
if __name__=="__main__":
    main()
    
         