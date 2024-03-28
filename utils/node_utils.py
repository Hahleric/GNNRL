import numpy as np

def get_edge_index(vehicle_num):
    """
    Get the edge index
    :param vehicle_num: number of nodes
    :return: edge index
    """
    edges = [(0, i + 1) for i in range(vehicle_num)] + [(i + 1, 0) for i in range(vehicle_num)]
    edges = np.array(edges, dtype=np.int64)
    return edges

def get_edge_attr(edge_index, feature):
    """
    Get the edge attribute
    :param edge_index: edge index
    :param feature: feature, currently recommend list
    :return: edge attribute
    """
    edge_attr = []
    for i in range(edge_index.shape[0] // 2):
        edge_attr.append(feature[i])
    for i in range(edge_index.shape[0] // 2):
        edge_attr.append(feature[i])
    return edge_attr

if __name__ == "__main__":
    vehicle_num = 10
    edge_index = get_edge_index(vehicle_num)
    print(edge_index)
    feature = np.random.normal(100, 50, vehicle_num)
    edge_attr = get_edge_attr(edge_index, feature)
    print(edge_attr)
    print(len(edge_attr))
    print(len(edge_attr[0]))