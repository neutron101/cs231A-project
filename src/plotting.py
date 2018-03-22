import numpy as np
from numpy import sin, cos, pi
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import math

#Debugging Tools
def show_images(kargs, titles):

    if len(kargs) > 0:
        index = 1
        plt.figure()
        for im in range(0, len(kargs)):

            plt.subplot(int(math.ceil(len(kargs)/3.0)), 3, index)
            plt.imshow(kargs[im], cmap='Greys')
            if len(titles) > im:
                plt.title(titles[im])
            index+=1

        plt.show()


def plot_camera_loc(image_data):
        
        pt = np.linalg.inv(image_data.frames[0].K).dot(np.array([500,500, 1]))
        pt = np.concatenate((pt, [1]))
        pt = pt * [1, 1, -.05, 1]
        print pt
        #pt = np.array([0,0,-2,1])
        points = np.zeros([len(image_data.frames), 3])
        points1 = np.zeros([len(image_data.frames), 3])


        line = np.zeros([10,4])
        for k in range(10):
            line[k] = pt + [0, 0, -.01*k, 0]   

        fig = plt.figure(figsize=(12,9))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_ylabel('y')
        ax.set_xlabel('x')
        ax.set_zlabel('z')

        rot = np.array([[math.cos(45), 0, -math.sin(45)],
            [0, 1, 0],
            [math.sin(45), 0, math.cos(45)]])
        #print rot

        i = 0
        for fr in image_data.frames[0:]:
            #print fr.RT
            RT = np.vstack((fr.RT, [0,0,0,1]))
            pt1 = RT.dot(pt)
            pt1 = (pt1/pt1[3])[0:3]
            points[i, :] = pt1
            print pt1

            draw_line = np.zeros([10,4])
            for k in range(line.shape[0]):
                rt = np.copy(RT)

                # r = rot.dot(RT[0:3,0:3])
                r = RT[0:3,0:3].dot(rot)
                #line[k][0:3] = r.dot(line[k][0:3])

                rt[0:3,0:3] = r 
                #draw_line[k] = rt[:,0:3].dot(line[k][0:3])
                #draw_line[k] = RT[:,0:3].dot(line[k][0:3])

                draw_line[k] = RT.dot(line[k])
#                draw_line[k,0:3] = RT[0:3,0:3].transpose().dot(line[k][0:3]) + RT[0:3,3].dot(line[k][0:3])
            ax.plot(draw_line[:,0], draw_line[:,1], draw_line[:,2],'k-', lw=2) 

            pt1 = np.hstack((pt1, [1]))
            R = RT[0:3,0:3]
            T = RT[0:3,3]        
            P = np.hstack((R.transpose(), -1*np.reshape(R.transpose().dot(T), [3,1])))
            pt1 = P.dot(pt1)            
            points1[i, :] = pt1

            i+=1

        points = np.asarray(points)
        ax.scatter(points[:,0], points[:,1], points[:,2], c='r', marker='o')
        ax.scatter(points1[:,0], points1[:,1], points1[:,2], c='b', marker='^')
        plt.show()


def plot_voxels_bounds(sil, K, RT, v_in):

    r = np.where(sil == 1)
    r = np.vstack((r[0], r[1]))
    r = np.vstack((r, np.ones([r.shape[1]])))
    new_voxels = np.linalg.inv(K).dot(r)
    new_voxels = new_voxels.transpose()

    R = RT[0:3,0:3]
    T = RT[0:3,3]
    P = np.hstack((R.transpose(), -1*np.reshape(R.transpose().dot(T), [3,1])))
    old = np.hstack((v_in, np.ones([v_in.shape[0],1])))
    old = P.dot(old.transpose())
    old = old.transpose()
    
    print "In--", "Min", np.min(old, axis=0), "Max", np.max(old, axis=0), np.max(old, axis=0)-np.min(old, axis=0)
    print "Out--", "Max", np.min(new_voxels, axis=0), "Max", np.max(new_voxels, axis=0), np.max(new_voxels, axis=0)-np.min(new_voxels, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_zlabel('z')
    ax.scatter(old[:,0], old[:,1], old[:,2], c='b', marker='o')
    ax.scatter(new_voxels[:,0], new_voxels[:,1], new_voxels[:,2], c='r', marker='^')
    plt.show()


def update_debugging_images(c, sil, img_list, title_list, img_frame):

    test = np.copy(sil)
    for j in range(c.shape[1]):
        if test[c[1,j]][c[0,j]] == 1.0:
            test[c[1,j],c[0,j]] = .05
        else:
            test[c[1,j],c[0,j]] = .6
    img_list.append(img_frame.image)
    img_list.append(img_frame.silhouette)
    img_list.append(test)

    title_list.append(img_frame.image_path)
    title_list.append(img_frame.sil_path)
    title_list.append("Test")


def axis_equal(ax, X, Y, Z):
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def plot_surface(voxels, voxel_size = 0.1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # First grid the data
    res = np.amax(voxels[1,:] - voxels[0,:])
    ux = np.unique(voxels[:,0])
    uy = np.unique(voxels[:,1])
    uz = np.unique(voxels[:,2])

    # Expand the model by one step in each direction
    ux = np.hstack((ux[0] - res, ux, ux[-1] + res))
    uy = np.hstack((uy[0] - res, uy, uy[-1] + res))
    uz = np.hstack((uz[0] - res, uz, uz[-1] + res))

    # Convert to a grid
    X, Y, Z = np.meshgrid(ux, uy, uz)

    # Create an empty voxel grid, then fill in the elements in voxels
    V = np.zeros(X.shape)
    N = voxels.shape[0]
    for ii in xrange(N):
            ix = ux == voxels[ii,0]
            iy = uy == voxels[ii,1]
            iz = uz == voxels[ii,2]
            V[iy, ix, iz] = 1

    marching_cubes = measure.marching_cubes_lewiner(V, 0, spacing=(voxel_size, voxel_size, voxel_size))
    verts = marching_cubes[0]
    faces = marching_cubes[1]
    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], lw=0, color='red')
    axis_equal(ax, verts[:, 0], verts[:,1], verts[:,2])
    plt.show()
