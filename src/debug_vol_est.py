import numpy as np
from plotting import *
import math
import utils
from utils import diff
import matplotlib.pyplot as plt

class DebugVolumeEstimator:

	default_num_voxels = 6e7

	def __init__(self, num_voxels=default_num_voxels):
		self.num_voxels = num_voxels
		self.disp_images = []
		self.im_titles = []

	def estimate(self, image_data):

		xlim, ylim, zlim = self.get_voxel_bounds(image_data)
		voxels, voxel_size = self.form_initial_voxels(xlim, ylim, zlim, self.num_voxels)

		for frame in image_data.frames:
			print "\t", "Processing frame: image=", frame.image_path, "Sil=", frame.sil_path
			voxels = self.carve(voxels, frame)
			if voxels.shape[0] > 0:
				pass
				#plot_surface(voxels, voxel_size)
			else:
				print "No voxels remaining ... skipping rest of the frames"
				break

		show_images(self.disp_images, self.im_titles)
		self.disp_images = []

		return self.calculate_volume(voxels, voxel_size)


	def calculate_volume(self, voxels, voxel_size, scale=1):

		volume = math.pow(scale, 3) * voxels.shape[0] * math.pow(voxel_size, 3)
		return volume


 
	def form_initial_voxels(self, xlim, ylim, zlim, num_voxels):

	    voxels = None
	    voxel_size = None

	    pattern_volume = diff(xlim)*diff(ylim)*diff(zlim)
	    voxel_volume = pattern_volume/num_voxels
	    voxel_size = abs(voxel_volume) ** (1./3)

	    xboxes = int(math.ceil(diff(xlim)/voxel_size))
	    yboxes = int(math.ceil(diff(ylim)/voxel_size))
	    zboxes = int(math.ceil(diff(zlim)/voxel_size))  
	    
	    initx = xlim[0]+voxel_size/2
	    inity = ylim[0]+voxel_size/2
	    initz = zlim[0]+voxel_size/2

	    z = np.tile(np.arange(zboxes)*voxel_size + np.ones(zboxes)*initz, xboxes*yboxes)
	    y = np.tile(np.repeat(np.arange(yboxes), zboxes)*voxel_size+inity, xboxes)
	    x = np.repeat(np.arange(xboxes), yboxes*zboxes)*voxel_size+initx

	    voxels = np.vstack((x, y, z)).T

	    return voxels, voxel_size



	def get_voxel_bounds(self, image_data, num_voxels = 4000):

		camera_positions = np.vstack([c.translation() for c in image_data.frames])

		xlim = [camera_positions[:,0].min(), camera_positions[:,0].max()]
		ylim = [camera_positions[:,1].min(), camera_positions[:,1].max()]
		zlim = [camera_positions[:,2].min(), camera_positions[:,2].max()]

		camera_range = .9 * np.sqrt(diff( xlim )**2 + diff( ylim )**2)
		for f in image_data.frames:
		    viewpoint = f.translation() - camera_range * f.get_camera_direction()
		    zlim[0] = min( zlim[0], viewpoint[2] )
		    zlim[1] = max( zlim[1], viewpoint[2] )


		####################### Debugging
		shift_x = (diff(xlim) / 4) * np.array([1, -1])

		#shift_y = (diff(ylim) / 4) * np.array([1, -1])
		shift_y = (diff(ylim) / 4) * np.array([1, -1]) + [-.05, -0.05]

		#print xlim, diff(xlim), shift_x, ylim, diff(ylim), shift_y
		#######################
		
		
		xlim = xlim + shift_x
		ylim = ylim + shift_y 


		# frame_idx = 0
		# voxels, voxel_size = self.form_initial_voxels(xlim, ylim, zlim, 4000)
		# voxels = self.carve(voxels, image_data.frames[frame_idx])

		# if voxels.shape[0] > 1:
		# 	xlimp = [voxels[:,0].min(), voxels[:,0].max()]
		# 	ylimp = [voxels[:,1].min(), voxels[:,1].max()]
		# 	zlimp = [voxels[:,2].min(), voxels[:,2].max()]

		# 	xlimp = xlimp + voxel_size * 1 * np.array([-1, 1])
		# 	ylimp = ylimp + voxel_size * 1 * np.array([-1, 1])
		# 	zlimp = zlimp + voxel_size * 1 * np.array([-1, 1])

		# 	xlim = [np.max([xlim[0], xlimp[0]]), np.min([xlim[1], xlimp[1]])]
		# 	ylim = [np.max([ylim[0], ylimp[0]]), np.min([ylim[1], ylimp[1]])]
		# 	zlim = [np.max([zlim[0], zlimp[0]]), np.min([zlim[1], zlimp[1]])]

		xlim = [-0.5, .2]
		ylim = [-.01, .8]
		zlim = [-.05, 1]

		return xlim, ylim, zlim
	

	def carve(self, voxels, img_frame):

		print "\t", "Voxels = ", voxels.shape[0]

		a = np.hstack((voxels, np.ones([voxels.shape[0],1])))
		c = img_frame.projection.dot(a.T)
		c = (c[0:2,:]/c[2,:])[0:2,:].astype(int)

		h = np.arange(voxels.shape[0], dtype=int)    
		fall = np.logical_and.reduce((c[0,:]>=0, c[0,:]<img_frame.silhouette.shape[1], c[1,:]<img_frame.silhouette.shape[0], c[1,:]>=0))
		c = c[:, fall]
		h = h[fall]

		indices = img_frame.silhouette == 1
		#######	 Debugging   #######
		print img_frame.projection
		update_debugging_images(c, img_frame.silhouette, self.disp_images, self.im_titles, img_frame)
		plot_voxels_bounds(img_frame.silhouette, img_frame.K, img_frame.RT, voxels)
		#######

		g_pts = h[indices[c[1], c[0]]]
		return voxels[g_pts]	
