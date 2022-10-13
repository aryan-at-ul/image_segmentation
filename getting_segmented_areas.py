from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import data, segmentation
from skimage import io, color
from skimage.io import imread
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
from skimage.future import graph
import networkx as nx
from skimage.measure import regionprops
import os
import numpy as np
from numpy import sqrt


path = "data/images/"
image = imread(path + 'person3.jpg')
# segments = slic(img_as_float(image), n_segments = 100, sigma = 5)
segments = slic(image, n_segments = 100, sigma = 5)
blank_image = 255 * np.ones_like(image , dtype = np.uint8)
rag = graph.rag_mean_color(image, segments,mode='distance')
# rag = graph_cut(rag, segments, 0.1)



# this part is for visualization from image to graph
# fig, (ax_img, ax_slic, ax_seg,ax_graph) = plt.subplots(1, 4, figsize = (36, 12))
# ax_img.imshow(image)
# ax_slic.imshow(segments)#, cmap = plt.cm.BrBG)
# ax_slic.imshow(image)
# out = color.label2rgb(segments, image, kind='avg', bg_label=0)
# #out = segmentation.mark_boundaries(out, labels, (0, 0, 0))
# #ax_seg.imshow(seg[:,:]) #remove it later
# ax_seg.imshow(segments)
# ax_seg.imshow(out)
# ax_graph.imshow(blank_image)

pos_info = {c.label-1: np.array([c.centroid[1],c.centroid[0]]) for c in regionprops(segments+1)}
# nx.draw(rag, pos = pos_info, ax = ax_slic,node_size=0.5)#, node_color='r', edge_color='b', alpha=0.5, width=1)
# nx.draw(rag, pos = pos_info, ax = ax_seg,node_size=0.5, node_color='r', edge_color='b', alpha=0.5, width=0.2)
# nx.draw(rag, pos = pos_info, ax = ax_graph, node_size = 0.5, edge_color='b', alpha=0.5, width=0.2)
# plt.show()
# end of visualization part


all_nodes_and_images = {}
rag2 = rag.copy()
G2 = nx.Graph()
for (i, segVal) in enumerate(np.unique(segments)):
	# construct a mask for the segment
    print(f"[x] inspecting segment {i}")
    mask = np.zeros(image.shape[:2], dtype = "uint8")
    mask[segments == segVal] = 255
    #print([[np.array([c.centroid[1],c.centroid[0]])]  for c in regionprops(mask)])
    #print(i,segVal,pos_info[segVal])
    # show the masked region
    segimg = cv2.cvtColor(cv2.bitwise_and(image, image, mask = mask), cv2.COLOR_BGR2RGB)

    # this code is to remove background
    # gray = cv2.cvtColor(segimg, cv2.COLOR_BGR2GRAY)
    # # threshold input image using otsu thresholding as mask and refine with morphology
    # ret, mask2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    # kernel = np.ones((9,9), np.uint8)
    # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
    # # put mask into alpha channel of result
    # result = segimg.copy()
    # result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    # result[:, :, 2] = mask2
    # print(result.shape)
    # end of removing background

    gray = cv2.cvtColor(segimg, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    # Find contour and sort by contour area
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Find bounding box and extract ROI
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        seg = image[y:y+h, x:x+w]
        break
    seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
    # cv2.imshow('ROI',ROI)
    #cv2.imwrite('ROI.png',ROI)
    # cv2.waitKey()

    # save resulting masked image
    cv2.imwrite(f"output/person3/{segVal}.png", seg)

    all_nodes_and_images[segVal] = seg
    rag2.nodes[segVal]['image'] = seg
    G2.add_node(segVal,image= seg)

    # cv2.imwrite(f"output/person3/{segVal}.png")
    # cv2.imwrite(f"output/person3/{segVal}.png", segimg)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Applied", cv2.bitwise_and(image, image, mask = mask))
    # cv2.waitKey(0)

# nx.draw(rag2, pos = pos_info, ax = ax_graph, node_size = 0.5, edge_color='b', alpha=0.5, width=0.2)
# plt.show()

# plt.close('all')

G = G2
edges = rag.edges
for e in edges:
    G.add_edge(e[0],e[1])

N = 100
pos=nx.spring_layout(G,k=3/sqrt(N))
pos = pos_info

# draw with images on nodes
nx.draw_networkx(G,pos,width=1,edge_color="b",alpha=0.6)
ax=plt.gca()
fig=plt.gcf()
fig.set_size_inches(20, 20)
# fig = plt.figure(figsize=(20,20))
trans = ax.transData.transform
trans2 = fig.transFigure.inverted().transform
# trans2 = fig.transFigure.transform
imsize = 0.05 # this is the image size
for n in G.nodes():
    (x,y) = pos[n]
    xx,yy = trans((x,y)) # figure coordinates
    xa,ya = trans2((xx,yy)) # axes coordinates
    a = plt.axes([xa-imsize/2.0,ya-imsize/2.0, imsize, imsize ])
    a.imshow(G.nodes[n]['image'])
    a.set_aspect('equal')
    a.axis('off')
plt.show()
nx.write_gpickle(G, "person3.gpickle")




# pos = pos_info
# pos=nx.spring_layout(G)

# fig=plt.figure(figsize=(20,20))
# ax=plt.subplot(111)
# ax.set_aspect('equal')
# nx.draw_networkx_edges(G,pos,ax=ax)

# plt.xlim(-1.5,1.5)
# plt.ylim(-1.5,1.5)


# trans=ax.transData.transform
# trans2=fig.transFigure.inverted().transform

# piesize=10 # this is the image size
# p2=piesize/2.0

# for n in G:
#     print(n)
#     print(pos[n])
#     print(G.nodes[n]['image'])
#     xx,yy=trans(pos[n]) # figure coordinates
#     xa,ya=trans2((xx,yy)) # axes coordinates
#     a = plt.axes([xa-p2,ya-p2, piesize, piesize])
#     a.set_aspect('equal')
#     a.imshow(G.nodes[n]['image'])
#     a.axis('off')
# plt.show()









