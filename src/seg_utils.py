import math
import numpy.linalg as LN
import numpy as np

ep = lambda p,b: p*b
is_close = lambda old, new, e: LN.norm(old-new) <= e

def meanshift_segmentation(im, features, bandwidth):
 
    cluster_means = []

    points = range(features.shape[0])
    np.random.shuffle(points)
    points = set(points)

    while len(points) > 0:
        points, cluster = meanshift(features, points, bandwidth)
        meanshift_merge(cluster_means, cluster, bandwidth)

    im_cluster = np.zeros(features.shape[0], dtype=np.int)

    fvecs = np.asarray(cluster_means)
    
    for f_idx in xrange(features.shape[0]):
        
        differ = fvecs-features[f_idx]
        diff_norm = LN.norm(differ, axis=1)
        closest_cluster = diff_norm.argmin()
        im_cluster[f_idx] = closest_cluster

    # print "No. of clusters:", len(cluster_means), "at bandwidth", bandwidth

    im_cluster_assign = np.reshape(im_cluster, im.shape[0:-1])
    return im_cluster_assign


def meanshift(features, pts, bandwidth):

    pt = pts.pop()
    mean_vec = features[pt]
    do_move = True
    cluster = set([pt])

    while do_move and len(pts)>0:

        in_pts = np.asarray(list(pts))
        fvecs = features[in_pts]
        differ = fvecs-mean_vec
        diff_norm = LN.norm(differ, axis=1)
        diff_is_close = np.where(diff_norm<=bandwidth)
        
        is_close_pts = in_pts[diff_is_close]

        if is_close_pts.shape[0] > 0:
            
            cluster = set(is_close_pts)
            cl_in_pts = np.asarray(list(cluster))
            is_close_feat = features[cl_in_pts]
            is_close_feat_mean = np.mean(is_close_feat, axis=0)

            do_move = not is_close(mean_vec, is_close_feat_mean, ep(.01, bandwidth))
            mean_vec = is_close_feat_mean

            pts -= set(is_close_pts)
        else:
            break


    return pts, mean_vec


def meanshift_merge(cluster_means, new_cluster, bandwidth):

    found = False    
    for c in cluster_means:
        if is_close(new_cluster, c, ep(.5, bandwidth)):
            found = True
            break
        
    if not found:
        cluster_means.append(new_cluster)

def kmeans_segmentation(im, features, num_clusters):

    f_indx = lambda h,w: (h*im.shape[1])+w

    mean = features[np.random.choice(features.shape[0], num_clusters)]
 
    ep = 0.05
    mean_change = np.inf
    im_cluster = None
    while mean_change > ep:
        new_mean, im_cluster = centroid_mean(mean, features, im.shape)
        mean_change = np.linalg.norm(mean-new_mean)
        mean = new_mean

    im_assign = np.reshape(im_cluster, im.shape[0:-1])
    return im_assign



def centroid_mean(mean, features, im_shape):

    rsums = np.zeros(mean.shape)
    rcounts = np.zeros(mean.shape[0], dtype=np.int)
    im_cluster = np.zeros(features.shape[0], dtype=np.int)

    for f_idx in range(features.shape[0]):
        closest_cluster = None
        dist = np.inf
        for i in range(mean.shape[0]):
            d = np.linalg.norm(features[f_idx]-mean[i,:])
            if d < dist:
                closest_cluster = i
                dist = d
        
        rsums[closest_cluster] = rsums[closest_cluster] + features[f_idx]
        rcounts[closest_cluster] = rcounts[closest_cluster] + 1
        im_cluster[f_idx] = closest_cluster

    new_mean = rsums.transpose()/rcounts
    new_mean = new_mean.transpose()

    return new_mean, im_cluster

