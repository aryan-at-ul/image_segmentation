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
from cnn_feature_extractor import fet_from_img
import  errno
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import time

current_file_path = os.path.dirname(os.path.abspath(__file__))

data_dir = f"{current_file_path}/chest_xray"

def make_dirs(dir):
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise




def read_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    val_dir = os.path.join(data_dir, 'val')
    train_normal_dir = os.path.join(train_dir, 'NORMAL')
    train_pneumonia_dir = os.path.join(train_dir, 'PNEUMONIA')
    test_normal_dir = os.path.join(test_dir, 'NORMAL')
    test_pneumonia_dir = os.path.join(test_dir, 'PNEUMONIA')
    val_normal_dir = os.path.join(val_dir, 'NORMAL')
    val_pneumonia_dir = os.path.join(val_dir, 'PNEUMONIA')
    return train_dir, test_dir, val_dir, train_normal_dir, train_pneumonia_dir, test_normal_dir, test_pneumonia_dir, val_normal_dir, val_pneumonia_dir


def read_image(path):
    images = []
    for one in os.listdir(path):
        image = cv2.imread(os.path.join(path, one))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (224, 224))
        image = img_as_float(image)
        images.append(image)
    return images

def read_one_image(path):
    image = imread(path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (224, 224))
    # image = img_as_float(image)
    return image


def get_image_paths(data_dir):
    train_dir, test_dir, val_dir, train_normal_dir, train_pneumonia_dir, test_normal_dir, test_pneumonia_dir, val_normal_dir, val_pneumonia_dir = read_data(data_dir)
    train_normal_image_paths = [os.path.join(train_normal_dir, f) for f in os.listdir(train_normal_dir)]
    train_pneumonia_image_paths = [os.path.join(train_pneumonia_dir, f) for f in os.listdir(train_pneumonia_dir)]
    test_normal_image_paths = [os.path.join(test_normal_dir, f) for f in os.listdir(test_normal_dir)]
    test_pneumonia_image_paths = [os.path.join(test_pneumonia_dir, f) for f in os.listdir(test_pneumonia_dir)]
    val_normal_image_paths = [os.path.join(val_normal_dir, f) for f in os.listdir(val_normal_dir)]
    val_pneumonia_image_paths = [os.path.join(val_pneumonia_dir, f) for f in os.listdir(val_pneumonia_dir)]
    return [train_normal_image_paths, train_pneumonia_image_paths, test_normal_image_paths, test_pneumonia_image_paths, val_normal_image_paths, val_pneumonia_image_paths]


def load_image(tain_normal_path,train_pneumonia_path,test_normal_path,test_pneumonia_path,val_normal_path,val_pneumonia_path):
    train_normal_image = read_image(tain_normal_path)
    train_pneumonia_image = read_image(train_pneumonia_path)
    test_normal_image = read_image(test_normal_path)
    test_pneumonia_image = read_image(test_pneumonia_path)
    val_normal_image = read_image(val_normal_path)
    val_pneumonia_image = read_image(val_pneumonia_path)
    return train_normal_image, train_pneumonia_image, test_normal_image, test_pneumonia_image, val_normal_image, val_pneumonia_image


# def slic_segment():
#     train_normal_image_paths, train_pneumonia_image_paths, test_normal_image_paths, test_pneumonia_image_paths, val_normal_image_paths, val_pneumonia_image_paths = get_image_paths(data_dir)
#     train_normal_image, train_pneumonia_image, test_normal_image, test_pneumonia_image, val_normal_image, val_pneumonia_image = load_image(train_normal_image_paths[0],train_pneumonia_image_paths[0],test_normal_image_paths[0],test_pneumonia_image_paths[0],val_normal_image_paths[0],val_pneumonia_image_paths[0])
#     segments = slic(train_normal_image, n_segments=100, sigma=5)
#     plt.imshow(mark_boundaries(train_normal_image, segments))
#     plt.show()





def make_graph_from_image(image,dirs_names,class_name,data_typ,image_name):
    G2 = nx.Graph()
    # img2 = np.zeros_like(image)
    # image[:,:,0] = gray
    # image[:,:,1] = gray
    # image[:,:,2] = gray
    if len(image.shape) < 3:
        image = np.stack((image,)*3, axis=-1)
    # image = np.stack((image,)*3, axis=-1)
    start = time.time()
    segments = slic(image, n_segments=100, sigma=5)
    rag = graph.rag_mean_color(image, segments,mode='distance')
    seg_imgs = []
    end = time.time()
    print(f"time taken for slic segmentation was : {end-start}")

    start_time = time.time()
    for (i, segVal) in enumerate(np.unique(segments)):
        # print(f"inspecting segment {i} {dirs_names} {class_name} {data_typ} {image_name}")
        mask = np.zeros(image.shape[:2], dtype = "uint8")
        mask[segments == segVal] = 255
        segimg = cv2.cvtColor(cv2.bitwise_and(image, image, mask = mask), cv2.COLOR_BGR2RGB)
        segimg = cv2.bitwise_and(image, image, mask = mask)
        # cv2.imshow("segimg", segimg)
        # print(segimg.shape,"::::shape of input image")
        gray = cv2.cvtColor(segimg, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        # Find contour and sort by contour area
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        seg = segimg.copy()
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            seg = image[y:y+h, x:x+w]
            break
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
        rag.nodes[segVal]['image'] = seg
        seg_imgs.append([seg,segVal])
        # G2.add_node(segVal,x = fet_from_img(seg))
        # G2.add_node(segVal,image= seg)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Start the load operations and mark each future with its URL
        futures = [executor.submit(fet_from_img, seg_img[0], seg_img[1])  for  i,seg_img in enumerate(seg_imgs)]
        for future in concurrent.futures.as_completed(futures):
            # fut = futures[future]
            # print("this result here",future)
            try:
                img_fet,i = future.result()
                # print("this result here",img_fet[0],i)
                G2.add_node(i,x = img_fet)
            except Exception as exc:
                print(f'generated an exception: {exc} for seg {i}')
            # else:
            #     print('all done for seg %s' % (i))

    end_time = time.time()
    print(f"total time take per image is {end_time - start_time}")
    edges = rag.edges
    for e in edges:
        G2.add_edge(e[0],e[1])

    return G2,segments



def draw_graph_as_image(G,segments):
    pos = {c.label-1: np.array([c.centroid[1],c.centroid[0]]) for c in regionprops(segments+1)}
    nx.draw_networkx(G,pos,width=1,edge_color="b",alpha=0.6)
    ax=plt.gca()
    fig=plt.gcf()
    fig.set_size_inches(20, 20)
    # fig = plt.figure(figsize=(20,20))
    trans = ax.transData.transform
    trans2 = fig.transFigure.inverted().transform
    # trans2 = fig.transFigure.transform
    imsize = 0.05 # this is the image size
    nodes = list(G.nodes())
    nodes = nodes[::-1]
    for n in nodes:
        (x,y) = pos[n]
        xx,yy = trans((x,y)) # figure coordinates
        xa,ya = trans2((xx,yy)) # axes coordinates
        # xa,ya = xx,yy#trans2((xx,yy))
        # xa,ya = x,y # axes coordinates
        a = plt.axes([xa-imsize/2.0,ya-imsize/2.0, imsize, imsize ])
        a.imshow(cv2.cvtColor(G.nodes[n]['image'], cv2.COLOR_BGR2RGB))
        a.set_aspect('equal')
        a.axis('off')
    plt.show()


def run_for_one_folder(folder_path):
    for one in folder_path:
        dirs_names = one.split("/")
        class_name = dirs_names[-2]
        data_typ = dirs_names[-3]
        image_name = dirs_names[-1]
        start = time.time()
        print(f"processing {class_name} {data_typ} {image_name}")
        make_dirs(f"{current_file_path}/chest_xray_graphs/{data_typ}/{class_name}")
        if image_name.split('.')[-2] + ".gpickle" in os.listdir(f"{current_file_path}/chest_xray_graphs/{data_typ}/{class_name}"):
            continue
        image = read_one_image(one)
        print("="*50)
        G,segments = make_graph_from_image(image,dirs_names,class_name,data_typ,image_name)
        end = time.time()
        print(f"this is the true time taken {end - start}")
        print("="*50)
        nx.write_gpickle(G, f"{current_file_path}/chest_xray_graphs/{data_typ}/{class_name}" + "/" + image_name.split('.')[-2] + ".gpickle")
        # break




def test_run():
    print("starting")
    make_dirs(current_file_path+'/'+'chest_xray_graphs')
    all_images_path = get_image_paths(data_dir)
    tasks = []
    for one_class in all_images_path:
        tasks.append(str(run_for_one_folder(one_class)))
        # break
    print("done")
    with ThreadPoolExecutor(max_workers=3) as executor:
        future = executor.submit([run_for_one_folder(x) for x in all_images_path])
    for result in future:
        print(result.result())



if __name__ == '__main__':
    test_run()
