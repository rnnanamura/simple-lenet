#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchviz import make_dot

import os
import time
import yaml
import random
import networkx as nx
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

torch.use_deterministic_algorithms(True)

#%%
class RndGraph(object):
    def __init__(self, num_nodes, graph_probability, nearest_neighbour_k=4, num_edges_attach=5):
        self.num_nodes = num_nodes
        self.graph_probability = graph_probability
        self.nearest_neighbour_k = nearest_neighbour_k
        self.num_edges_attach = num_edges_attach

    def make_graph_obj(self):
        graph_obj = nx.random_graphs.connected_watts_strogatz_graph(self.num_nodes, self.nearest_neighbour_k,
                                                                self.graph_probability)
        return graph_obj

    def get_graph_config(self, graph_obj):
        incoming_edges = {}
        incoming_edges[0] = []
        node_list = [0]
        last = []
        for n in graph_obj.nodes():
            neighbor_list = list(graph_obj.neighbors(n))
            neighbor_list.sort()

            edge_list = []
            passed_list = []
            for nbr in neighbor_list:
                if n > nbr:
                    edge_list.append(nbr + 1)
                    passed_list.append(nbr)
            if not edge_list:
                edge_list.append(0)
            incoming_edges[n + 1] = edge_list
            if passed_list == neighbor_list:
                last.append(n + 1)
            node_list.append(n + 1)
        incoming_edges[self.num_nodes + 1] = last
        node_list.append(self.num_nodes + 1)
        return node_list, incoming_edges

    def save_graph(self, graph_obj, path_to_write):
        if not os.path.isdir("cached_graph_obj"):
            os.mkdir("cached_graph_obj")
        with open(f"./cached_graph_obj/{path_to_write}", "w") as fh:
            yaml.dump(graph_obj, fh)

    def load_graph(self, path_to_read):
        with open(f"./cached_graph_obj/{path_to_read}", "r") as fh:
            return yaml.load(fh, Loader=yaml.Loader)

#%%
class SepConv2d(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_length=3, dilation_size=1, padding_size=1, stride_length=1, bias_flag=True):
        super(SepConv2d, self).__init__()
        self.conv_layer = nn.Conv2d(input_ch, input_ch, kernel_length, stride_length, padding_size, dilation_size,
                              bias=bias_flag, groups=input_ch)
        self.pointwise_layer = nn.Conv2d(input_ch, output_ch, kernel_size=1, stride=1, padding=0, dilation=1,
                                         groups=1, bias=bias_flag)

    def forward(self, x):
        return self.pointwise_layer(self.conv_layer(x))
#%%
class UnitLayer(nn.Module):
    def __init__(self, input_ch, output_ch, stride_length=1):
        super(UnitLayer, self).__init__()

        self.dropout = 0.3

        self.unit_layer = nn.Sequential(
            nn.ReLU(),
            SepConv2d(input_ch, output_ch, stride_length=stride_length),
            nn.BatchNorm2d(output_ch),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        return self.unit_layer(x)
#%%
class GraphNode(nn.Module):
    def __init__(self, input_degree, input_ch, output_ch, stride_length=1):
        super(GraphNode, self).__init__()
        self.input_degree = input_degree
        if len(self.input_degree) > 1:
            self.params = nn.Parameter(torch.ones(len(self.input_degree), requires_grad=True))
        self.unit_layer = UnitLayer(input_ch, output_ch, stride_length=stride_length)

    def forward(self, *ip):
        if len(self.input_degree) > 1:
            op = (ip[0] * torch.sigmoid(self.params[0]))
            for idx in range(1, len(ip)):
                op += (ip[idx] * torch.sigmoid(self.params[idx]))
            return self.unit_layer(op)
        else:
            return self.unit_layer(ip[0])
# %%
class RandWireGraph(nn.Module):
    def __init__(self, num_nodes, graph_prob, input_ch, output_ch, train_mode, graph_name):
        super(RandWireGraph, self).__init__()
        self.num_nodes = num_nodes
        self.graph_prob = graph_prob
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.train_mode = train_mode
        self.graph_name = graph_name

        # get graph nodes and in edges
        rnd_graph_node = RndGraph(self.num_nodes, self.graph_prob)
        if self.train_mode is True:
            print("train_mode: ON")
            rnd_graph = rnd_graph_node.make_graph_obj()
            self.node_list, self.incoming_edge_list = rnd_graph_node.get_graph_config(rnd_graph)
            rnd_graph_node.save_graph(rnd_graph, graph_name)
        else:
            rnd_graph = rnd_graph_node.load_graph(graph_name)
            self.node_list, self.incoming_edge_list = rnd_graph_node.get_graph_config(rnd_graph)

        # define input Node
        self.list_of_modules = nn.ModuleList([GraphNode(self.incoming_edge_list[0], self.input_ch, self.output_ch,
                                                        stride_length=2)])
        # define the rest Node
        self.list_of_modules.extend([GraphNode(self.incoming_edge_list[n], self.output_ch, self.output_ch)
                                     for n in self.node_list if n > 0])

    def forward(self, x):
        mem_dict = {}
        # start vertex
        op = self.list_of_modules[0].forward(x)
        mem_dict[0] = op

        # the rest vertex
        for n in range(1, len(self.node_list) - 1):
            # print(node, self.in_edges[node][0], self.in_edges[node])
            if len(self.incoming_edge_list[n]) > 1:
                op = self.list_of_modules[n].forward(*[mem_dict[incoming_vtx]
                                                       for incoming_vtx in self.incoming_edge_list[n]])
            else:
                op = self.list_of_modules[n].forward(mem_dict[self.incoming_edge_list[n][0]])
            mem_dict[n] = op

        op = mem_dict[self.incoming_edge_list[self.num_nodes + 1][0]]
        for incoming_vtx in range(1, len(self.incoming_edge_list[self.num_nodes + 1])):
            op += mem_dict[self.incoming_edge_list[self.num_nodes + 1][incoming_vtx]]
        return op / len(self.incoming_edge_list[self.num_nodes + 1])
# %%
class RandWireNNModel(nn.Module):
    def __init__(self, num_nodes, graph_prob, input_ch, output_ch, train_mode):
        super(RandWireNNModel, self).__init__()
        self.num_nodes = num_nodes
        self.graph_prob = graph_prob
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.train_mode = train_mode
        self.dropout = 0.3
        self.class_num = 10

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.output_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.output_ch),
        )

        self.conv_layer_2 = nn.Sequential(
            RandWireGraph(self.num_nodes, self.graph_prob, self.input_ch, self.output_ch * 2, self.train_mode,
                          graph_name="conv_layer_2")
        )
        self.conv_layer_3 = nn.Sequential(
            RandWireGraph(self.num_nodes, self.graph_prob, self.input_ch * 2, self.output_ch * 4, self.train_mode,
                          graph_name="conv_layer_3")
        )
        self.conv_layer_4 = nn.Sequential(
            RandWireGraph(self.num_nodes, self.graph_prob, self.input_ch * 4, self.output_ch * 8, self.train_mode,
                          graph_name="conv_layer_4")
        )

        self.classifier_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.input_ch * 8, out_channels=1280, kernel_size=1),
            nn.BatchNorm2d(1280)
        )

        self.output_layer = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(1280, self.class_num)
        )

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)
        x = self.classifier_layer(x)

        # global average pooling
        _, _, h, w = x.size()
        x = F.avg_pool2d(x, kernel_size=[h, w])
        x = torch.squeeze(x)
        x = self.output_layer(x)

        return x