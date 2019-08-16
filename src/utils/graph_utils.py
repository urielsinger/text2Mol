import pandas as pd
import scipy.sparse as sp

from pyvis.network import Network

import warnings
from tqdm import tqdm

from utils.general_utils import *

def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten())
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten())
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def random_edges(graph_nx, num):
    E, N = len(graph_nx.edges()), len(graph_nx.nodes())
    max_num_false_edges = N*(N-1)-E if graph_nx.is_directed() else N*(N-1)/2-E
    if max_num_false_edges <= 0:
        warnings.warn('not enough false edges for sampling, switching to half of possible samples')
        num = max_num_false_edges/2
    try:
        all_false_edges = np.array(list(nx.non_edges(graph_nx)))
        false_edges = all_false_edges[np.random.choice(len(all_false_edges), num, replace=False)]
    except:

        # false_edges = []
        # non_edges = nx.non_edges(graph_nx)
        # for i, non_edge in enumerate(non_edges):
        #     if i < num:
        #         false_edges.append(non_edge)
        #         pbar.update(1)
        #     else:
        #         j = np.random.randint(0, i)
        #         if j < num:
        #             false_edges[j] = non_edge
        # false_edges = np.array(false_edges)

        false_edges = []
        nodes = graph_nx.nodes()
        with tqdm(total=num) as pbar:
            while len(false_edges) < num:
                random_edge = sorted(np.random.choice(nodes, 2, replace=False))
                if random_edge[1] not in graph_nx[random_edge[0]] and random_edge not in false_edges:
                    false_edges.append(random_edge)
                    pbar.update(1)

    return false_edges


def get_distance(source,target,graph_nx):
    if source not in graph_nx or target not in graph_nx:
        return np.inf
    try:
        distance = nx.shortest_path(graph_nx,source,target)
    except:
        return np.inf
    return len(distance)-1


def get_graph_T(graph_nx, min_time=-np.inf, max_time=np.inf, return_df=False):
    '''
    Given a graph with a time attribute for each edge, return the subgraph with only edges between an interval.
    Args:
        graph_nx: networkx - the given graph
        min_time: int - the minimum time step that is wanted. Default value -np.inf
        max_time: int - the maximum time step that is wanted. Default value np.inf
        return_df: bool - if True, return a DataFrame of the edges and attributes,
                          else, a networkx object

    Returns:
        sub_graph_nx: networkx - subgraph with only edges between min_time and max_time
    '''
    relevant_edges = []
    attr_keys = []

    if len(graph_nx.nodes()) == 0:
        return graph_nx

    for u,v,attr in graph_nx.edges(data=True):
        if min_time < attr['time'] and attr['time'] <= max_time:
            relevant_edges.append((u,v,*attr.values()))

            if attr_keys != [] and attr_keys != attr.keys():
                raise Exception('attribute keys in \'get_graph_T\' are different')
            attr_keys = attr.keys()

    graph_df = pd.DataFrame(relevant_edges, columns=['from', 'to', *attr_keys])

    if return_df:
        node2label = nx.get_node_attributes(graph_nx, 'label')
        if len(node2label) > 0:
            graph_df['from_class'] = graph_df['from'].map(lambda node: node2label[node])
            graph_df['to_class'] = graph_df['to'].map(lambda node: node2label[node])
        return graph_df
    else:
        sub_graph_nx = nx.from_pandas_edgelist(graph_df,'from','to',list(attr_keys),create_using=type(graph_nx)())

        # add node attributes
        all_node_attributes = graph_nx.nodes(data=True)[list(graph_nx.nodes())[0]]
        for node_attribute in all_node_attributes:
            node2attribue = nx.get_node_attributes(graph_nx, node_attribute)
            nx.set_node_attributes(sub_graph_nx, node2attribue, node_attribute)

        return sub_graph_nx

def get_graph_times(graph_nx):
    return np.sort(np.unique(list(nx.get_edge_attributes(graph_nx, 'time').values())))

def get_node_attribute_matix(graph_nx, attribute, nbunch=None):
    if nbunch is None:
        nbunch = graph_nx.nodes()
    return np.array([graph_nx.nodes[node][attribute] for node in nbunch])

def get_pivot_time(graph_nx, wanted_ratio=0.2):
    '''
    Given a graph with 'time' attribute for each edge, calculate the pivot time that gives
    a wanted ratio to the train and test edges
    Args:
        graph_nx: networkx - Graph
        wanted_ratio: float - number between 0 and 1 representing |test|/(|train|+|test|)

    Returns:
        pivot_time: int - the time step that creates such deviation
    '''
    times = get_graph_times(graph_nx)

    if wanted_ratio == 0:
        return times[-1]

    time2dist_from_ratio = {}
    for time in times[int(len(times)/3):]:
        train_graph_nx = multigraph2graph(get_graph_T(graph_nx, max_time=time))
        num_edges_train = len(train_graph_nx.edges())

        test_graph_nx = get_graph_T(graph_nx, min_time=time)
        test_graph_nx.remove_nodes_from([node for node in test_graph_nx if node not in train_graph_nx])
        test_graph_nx = multigraph2graph(test_graph_nx)
        num_edges_test = len(test_graph_nx.edges())

        time2dist_from_ratio[time] = np.abs(wanted_ratio - num_edges_test/(num_edges_train+num_edges_test))

    pivot_time = min(time2dist_from_ratio, key=time2dist_from_ratio.get)

    print(f'pivot time {pivot_time}, is close to the wanted ratio by {round(time2dist_from_ratio[pivot_time],3)}')

    return pivot_time

def multigraph2graph(multi_graph_nx):
    if type(multi_graph_nx) == nx.Graph or type(multi_graph_nx) == nx.DiGraph:
        return multi_graph_nx
    graph_nx = nx.DiGraph() if multi_graph_nx.is_directed() else nx.Graph()

    if len(multi_graph_nx.nodes()) == 0:
        return graph_nx

    # add edges + attributes
    for u, v, data in multi_graph_nx.edges(data=True):
        data['weight'] = data['weight'] if 'weight' in data else 1.0

        if graph_nx.has_edge(u, v):
            graph_nx[u][v]['weight'] += data['weight']
        else:
            graph_nx.add_edge(u, v, **data)

    # add node attributes
    all_node_attributes = multi_graph_nx.nodes(data=True)[list(multi_graph_nx.nodes())[0]]
    for node_attribute in all_node_attributes:
        node2attribue = nx.get_node_attributes(multi_graph_nx, node_attribute)
        nx.set_node_attributes(graph_nx, node2attribue, node_attribute)

    return graph_nx


def visualize(graph_nx, name='graph'):
    got_net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    # set the physics layout of the network
    got_net.barnes_hut()

    got_net.from_nx(graph_nx)

    # add neighbor data to node hover data
    neighbor_map = got_net.get_adj_list()
    for node in got_net.nodes:
        node["title"] += " Neighbors:<br>" + "<br>".join(neighbor_map[node["id"]])
        node["value"] = len(neighbor_map[node["id"]])

    got_net.show_buttons()
    got_net.show(f'{name}.html')

def text2graph(text):
    from nltk import StanfordNERTagger, word_tokenize
    import os
    os.environ['JAVAHOME'] = r"C:\Program Files (x86)\Java\jre1.8.0_181\bin\java.exe"

    st = StanfordNERTagger(r'..\..\..\stanford-ner-2018-10-16\classifiers\english.all.3class.distsim.crf.ser.gz',
                           r'..\..\..\stanford-ner-2018-10-16\stanford-ner.jar',
                           encoding='utf-8')

    # merge objects into one
    classified_text = st.tag(word_tokenize(text))
    merged_classified_text = [classified_text[0]]
    full_word = []
    for i in range(1, len(classified_text)):
        prev_word, prev_class = classified_text[i - 1]
        current_word, current_class = classified_text[i]
        if current_class != prev_class or current_class == 'O':
            merged_classified_text.append((' '.join(full_word), prev_class))
            full_word = [current_word]
        else:
            full_word.append(current_word)

    # create dataframe of all edges in graph
    edges = []
    win_size = 20
    half_win_size = int(win_size / 2)
    for i in range(half_win_size, len(merged_classified_text) - half_win_size - 1):
        word, word_type = merged_classified_text[i]
        if word_type != 'PERSON':
            continue
        for neighbor, neighbor_type in merged_classified_text[i - half_win_size:i + half_win_size + 1]:
            if neighbor_type != 'PERSON':
                continue
            edges.append([word, neighbor, i])

    graph_df = pd.DataFrame(edges, columns=['from', 'to', 'time'])

    return nx.from_pandas_edgelist(graph_df, 'from', 'to', 'time', create_using=nx.MultiGraph())
