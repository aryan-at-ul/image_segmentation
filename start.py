from skimage import data, segmentation
from skimage import io, color

from skimage.future import graph
import networkx as nx
from skimage.measure import regionprops
from skimage.io import imread
import matplotlib.pyplot as plt
import os
import numpy as np

def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])



def read_image():
    """
    read all images from data folder and return a list of images
    :return: list of images
    """
    path = 'data/images/'
    images = os.listdir(path)
    images_anno = [x+'_anno.png' for x in images]
    #images = [path+x for x in images]
    images_seg = ['data/images-gt/'+x.replace('jpg','png') for x in images ]
    images = [path+x for x in images]
    images_anno = ['data/images-labels/'+x for x in images_anno]
    return images,images_seg,images_anno


def slic_seg(images,seg_data,compactness,threshold,mode):
    rand_idx = [70]#np.random.choice(range(len(images)), 1)
    print(rand_idx,img_data[rand_idx[0]])
    img = imread(img_data[rand_idx[0]])
    seg = imread(seg_data[rand_idx[0]])
    labels_i = segmentation.slic(img, n_segments = 300, compactness=compactness)
    print(labels_i.shape)
    rag = graph.rag_mean_color(img, labels_i,mode=mode) #mode='distance')

    labels = graph.merge_hierarchical(labels_i, rag, thresh= threshold, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)

    #labels = labels_i
    fig, (ax_img, ax_slic, ax_seg) = plt.subplots(1, 3, figsize = (36, 12))
    ax_img.imshow(img)
    ax_slic.imshow(labels)#, cmap = plt.cm.BrBG)
    out = color.label2rgb(labels, img, kind='avg', bg_label=0)
    out = segmentation.mark_boundaries(out, labels, (0, 0, 0))
    #ax_seg.imshow(seg[:,:])
    ax_seg.imshow(out)

    pos_info = {c.label-1: np.array([c.centroid[1],c.centroid[0]]) for c in regionprops(labels_i+1)}
    nx.draw(rag, pos = pos_info, ax = ax_slic,node_size=0.5)#, node_color='r', edge_color='b', alpha=0.5, width=0.1)
    #plt.show()



def run_experiments():
    img_data,seg_data,anno_data = read_image()
    for modei,mode in enumerate(['similarity','distance']):
        for comi,comp in enumerate([0.01,0.1,0.5,1,2,5,10]):
            for thresi,thres in enumerate([0.01,0.1,0.5,1,2,5,10]):
                print("for compactness = ",comp," and threshold = ",thres," and mode = ",mode)
                slic_seg(img_data,seg_data,comp,thres,mode)
                plt.savefig('results/{}_{}_{}.png'.format(comp,thres,mode))
                plt.close()

if __name__ == '__main__':
    img_data,seg_data,anno_data = read_image()
    #slic_seg(img_data,seg_data)
    run_experiments()
