"""
@Author: Chaoyu Guan
Realization of Ray tracing and 3D constructing
"""
import numpy as np
from utils import *

# mini serves as the epsilon, which is used to compare with 0
mini = 0.0001

arr = np.array

class Camera():
	"""
	Camera used in the Spacial domain
	Parameters:
	@ position 		: Camera's position
	@ orientation 	: Camera's orientation
	@ upvec			: Camera's up vector
	"""

	def __init__(self,position = arr([50.,0.,0.]),orientation = arr([-1.,0.,0.]),upvec = arr([0,0,1])):
		self.position = position
		self.orientation = normal(orientation)
		self.upvec = normal(upvec)

	def setPosition(self,position):
		self.position = position

	def setOrientation(self,orientation):
		if lenth(orientation) < mini:
			raise ValueError("Camera orientation lenth wrong!")
		self.orientation = normal(orientation)

	def setUpvec(self,upvec):
		if lenth(upvec) < mini:
			raise ValueError("Camera upvec lenth wrong!")
		self.upvec = normal(upvec)

class Spacial():
	"""
	Spacial domain realization
	Parameters:
	@ Camera 		: The camera used in this space, one space must and only have one camera
	@ objectlists	: The object list in this space, object must be the <class 'MyObject'> type
	@ lightlist		: The light in this space, object must be the <class 'light'> type
	"""

	def __init__(self):
		self.Camera = Camera()
		self.objectlists = []
		self.lightlist = []
		self.pointlight = PointLight(np.array([20,0,0]),np.array([1.,1.,1.]),5)

	def setCameraParameter(self,position,orientation,upvec):
		"""
		Set this space's camera parameter
		@ position 		: np.array with 3 elements
		@ orientation 	: np.array with 3 elements
		"""
		self.Camera.setPosition(position)
		self.Camera.setOrientation(orientation)
		self.Camera.setUpvec(upvec)

	def setCameraPosition(self,position):
		self.Camera.setPosition(position)

	def AddObject(self,obj):
		"""
		Add object to this space
		@ obj 	: MyObject type
		"""
		self.objectlists.append(obj)

	def AddObjects(self,objects):
		"""
		Add a number of objects in this space
		@ objects 	: list with elements of MyObject type
		"""
		for obj in objects:
			self.objectlists.append(obj)

	def AddLight(self,lit):
		"""
		Add a light source to space
		@ lit 	: light type
		"""
		self.lightlist.append(lit)

	def AddLights(self,lits):
		"""
		Add a list of lights to space
		@ lits 	: list with elements of light type
		"""
		for lit in lits:
			self.lightlist.append(lit)

	def Render(self, orth=True, resolution_width = 200, resolution_height = 200, width = 10, height = 10):
		"""
		Render the space and output a camera picture
		@ orth 				: Whether to use orthognal or perspective
		@ resolution_width	: resolution on width
		@ resolution_height	: resolution on height
		@ width 			: lenth of width
		@ height 			: lenth of height
		return value:
		@ pic_matrix 		: picture matrix with shape [resolution_height,resolution_width,3] with float RGB coding
		"""
		pic_matrix = np.zeros([resolution_height,resolution_width,3])
		for i in range(resolution_width):
			for j in range(resolution_height):
				x_idx = 30
				y_idx = (i - resolution_width/2) / resolution_width * width
				z_idx = (-j + resolution_height/2) / resolution_height * height
				orth_idx = np.array([x_idx,y_idx,z_idx])
				if not orth:
					pic_matrix[j][i][:] = self.ray_tracing(self.Camera.position,np.array(orth_idx-self.Camera.position),0,5,[1,1])
				else:
					pic_matrix[j][i][:] = self.ray_tracing(orth_idx,np.array([-1,0,0]),0,5,[1,1])
		return pic_matrix

	def render_test(self,orth = True, y = 10, z = 10):
		start = np.array([30,y,z])
		vec = np.array([-1,0,0])
		self.ray_tracing(start,vec,0,5,[1,1])

	def ray_tracing(self,start,vec,iters,max_iter,number):
		"""
		Ray tracing realization
		@ start 			: the start of the ray
		@ vec 				: the orientation of the vec
		@ iters 			: the current iteration number
		@ max_iter 			: max iteration number
		@ number 			: np.array with two parameters, number[0] is for diffusion theta, and number[1] is for diffusion alpha
		return value:
		An np.array with three parameters standing for R G B seperately
		"""
		vec = normal(vec)
		if iters >= max_iter:
			return np.array([0.,0.,0.])
		di_color = am_color = sp_color = np.array([0.,0.,0.])
		current_depth = 10000
		for i in range(len(self.objectlists)):
			obj = self.objectlists[i]
			# Calculate intersect with obj
			tmp_depth,point,After_ray_list,norm,inter = obj.intersect(start,vec,number)
			if not inter:
				continue
			if tmp_depth < current_depth:
				# Refresh the depth
				current_depth = tmp_depth
				# Calculate specular light
				sp_color = arr([0.,0.,0.])
				for lit in self.lightlist:
					sp_color += lit.calculate(point,vec,norm,self.objectlists,number)
				sp_color = get_min(sp_color,arr([1.,1.,1.])) * obj.ts
				# Calculate diffusion light
				di_color = arr([0.,0.,0.])
				for ray in After_ray_list:
					di_color += self.ray_tracing(point,ray[0],iters+1,max_iter,number) * ray[1] * obj.td
				# Calculate ambient light
				am_color = obj.ta

		return get_min(di_color * 0.4  + am_color * 0.1 + sp_color * 0.5, np.array([1.,1.,1.]))


class light(object):
	"""
	light class in space
	Parameters:
	@ color 		: np.array with 3 elements representing R G B
	@ n 			: the power of cos
	"""
	def __init__(self, color, n):
		super(light, self).__init__()
		self.color = color
		self.n = n

	def setcolor(self,color):
		"""
		set light color
		@ color 		: np.array with 3 elements representing R G B
		"""
		self.color = color

	def setn(self,n):
		"""
		set n
		@ n 			: the power of cos
		"""
		self.n = n

	def calculate(self,point,vec,norm,objects,number):
		"""
		calculate the specular light on point
		@ point 		: np.array, size = 3, the point we want to calculate on
		@ vec 			: np.array, size = 3, the ray tracing into point
		@ norm 			: np.array, size = 3, the normal vector of current plane
		@ objects 		: the object list from space
		@ number 		: np.array with two parameters, number[0] is for diffusion theta, and number[1] is for diffusion alpha
		"""
		return arr([0.,0.,0.])
		
class PointLight(light):
	"""
	Point light class
	@ position  		: The point light position, np.array type
	"""
	def __init__(self, position, color = arr([1.,1.,1.]), n = 5):
		super(PointLight,self).__init__(color,n)
		self.position = position

	def setPosition(self,position):
		"""
		set the position
		@ position 		: The point light position, np.array type
		"""
		self.position = position

	def calculate(self,point,vec,norm,objects,number):
		vec = normal(vec)
		norm = normal(norm)
		ray_to_light = self.position - point
		ray_to_light = normal(ray_to_light)
		current_depth = 10000
		for i in range(len(objects)):
			obj = objects[i]
			tmp_depth, _, _, _, inter = obj.intersect(point,ray_to_light,number)
			if inter:
				return np.array([0,0,0])

		norm = np.inner(norm,ray_to_light)/np.inner(norm,norm)*norm
		target_ray = - 2 * norm + ray_to_light
		coss = np.inner(target_ray,vec)/(lenth(target_ray)*lenth(vec))
		coss = coss ** self.n
		if (coss < 0):
			return np.array([0.,0.,0.])
		return self.color * coss

class Parallel_light(light):
	"""
	Parallel_light class
	@ orientation 	: the light orientation
	"""
	def __init__(self, orientation = arr([0,0,-1]), color = arr([1.,1.,1.]), n = 5):
		super(Parallel_light, self).__init__(color,n)
		self.orientation = normal(-orientation)

	def setorientation(self,orientation):
		self.orientation = normal(-orientation)

	def calculate(self,point,vec,norm,objects,number):
		vec = normal(vec)
		norm = normal(norm)
		ray_to_light = self.orientation
		current_depth = 10000
		for i in range(len(objects)):
			obj = objects[i]
			tmp_depth, _, _, _, inter = obj.intersect(point,ray_to_light,number)
			if inter:
				return np.array([0,0,0])

		norm = np.inner(norm,ray_to_light)/np.inner(norm,norm)*norm
		target_ray = - 2 * norm + ray_to_light
		coss = np.inner(target_ray,vec)/(lenth(target_ray)*lenth(vec))
		coss = coss ** self.n
		if (coss < 0):
			return np.array([0.,0.,0.])
		return self.color * coss

		

class MyObject(object):
	"""
	MyObject class
	father class on all of space object
	Parameters:
	@ texture_s 		: np.array, size = 3, specular texture
	@ texture_d 		: np.array, size = 3, diffusion texture
	@ texture_a 		: np.array, size = 3, ambitious texture
	"""
	def __init__(self,texture_s,texture_d,texture_a):
		super(MyObject,self).__init__()
		self.ts = texture_s
		self.td = texture_d
		self.ta = texture_a

	def setTexture_s(self,texture_s):
		self.ts = texture_s

	def setTexture_d(self,texture_d):
		self.td = texture_d

	def setTexture_a(self,texture_a):
		self.ta = texture_a

	def intersect(self,start,vec,number):
		"""
		Use given start and vec to get the intersect point of this vector and this object
		@ start 		: np.array, size = 3, the vector start point
		@ vec 			: np.array, size = 3, the orientation of vector
		@ number 		: np.array with two parameters, number[0] is for diffusion theta, and number[1] is for diffusion alpha
		return value
		@ t 			: the depth of intersect point from start
		@ interpoint	: the intersect point
		@ ray_list 		: rays from this interpoint
		@ norm 			: objects norm on this interpoint
		@ Inter 		: whether the vector has intersected with this object
		"""
		t = 0
		interpoint = arr([0,0,0])
		ray_list = []
		norm = arr([0,0,0])
		Inter = False
		return t,interpoint,ray_list,norm,Inter


class Circle(MyObject):
	"""
	Sphere object
	Parameters:
	@ position 			: np.array, size = 3, center point of this shpere
	@ r 				: double, the radius of sphere
	"""
	def __init__(self,position,radius,texture_s = arr([1.,1.,1.]),texture_d = arr([1.,1.,0.]),texture_a = arr([1.,1.,1.])):
		MyObject.__init__(self,texture_s,texture_d,texture_a)
		self.r = radius
		self.position = position

	def setRadius(self,radius):
		self.r = radius

	def setPosition(self,position):
		self.position = position

	def intersect(self,start,vec,number):
		v = start - self.position
		dv = np.inner(vec,v)
		delta = dv*dv - np.inner(v,v) + self.r*self.r
		if delta < 0:
			#print("No inter 1")
			return 0,arr([0.,0.,0.]),[],arr([0.,0.,0.]),False
		t1 = - dv - np.sqrt(delta)
		t2 = - dv + np.sqrt(delta)
		if t2 < mini:
			#print("No inter 2")
			return 0,arr([0.,0.,0.]),[],arr([0.,0.,0.]),False
		elif t1 < mini:
			t = t2
		else:
			t = t1
		circle_point = start + vec * t
		norm = self.get_norm(circle_point)
		norm = normal(norm)
		ray_list = get_ray_list(norm,number)
		return t,circle_point,ray_list,norm,True
		"""
		circle_point_1 = start + vec * t1
		circle_point_2 = start + vec * t2
		if lenth(circle_point_2 - start) < mini or t2 < 0:
			return 0,0,[],np.array([0,0,0]),False
		if lenth(circle_point_1 - start) < mini or t1 < 0:
			norm = self.get_norm(circle_point_2)
			norm = normal(norm)
			ray_list = get_ray_list(norm,1,1)
			if t2 < 0.1:
				print(t2)
			return t2,circle_point_2,ray_list,norm,True
		norm = self.get_norm(circle_point_1)
		norm = normal(norm)
		ray_list = get_ray_list(norm,1,1)
		if(t1<0.1):
			print(t1)
		return t1,circle_point_1,ray_list,norm,True
		"""

	def get_norm(self,point):
		"""
		get the norm at the point given
		@ point 	: np.array, size = 3, point on the sphere
		"""
		return normal(point - self.position)

class Triangle(MyObject):
	"""
	@ v1 v2 v3 		: Three vertices representing this triangle
	@ norm 			: Plane norm
	"""
	def __init__(self,v1,v2,v3,texture_s = arr([1.,1.,1.]),texture_d = arr([1.,0.,1.]),texture_a = arr([1.,0.,1.])):
		MyObject.__init__(self,texture_s,texture_d,texture_a)
		self.v1 = v1
		self.v2 = v2
		self.v3 = v3
		self.norm = np.cross(v2-v1,v3-v2)
		self.norm = normal(self.norm)

	def intersect(self,start,vec,number):
		if np.abs(np.inner(vec,self.norm)/lenth(vec)) < 0.001:
			return 0,arr([0.,0.,0.]),[],arr([0.,0.,0.]),False
		if self.ContainPoint(start):
			return 0,arr([0.,0.,0.]),[],arr([0.,0.,0.]),False
		else:
			t = np.inner((self.v1 - start),self.norm)/np.inner(self.norm,vec)
			if t < 0:
				return 0,arr([0.,0.,0.]),[],arr([0.,0.,0.]),False
			point = start + t * vec
			#print(point)
			if self.ContainPoint(point):
				ray_list = get_ray_list(self.norm,number)
				return t,point,ray_list,self.norm,True
			return 0,arr([0.,0.,0.]),[],arr([0.,0.,0.]),False


	def ContainPoint(self,point):
		"""
		Used to judge whether this object contains given point
		"""
		if same_point(point,self.v1) or same_point(point,self.v2) or same_point(point,self.v3):
			#print("get 1")
			return True
		vv1 = point - self.v1
		vv2 = point - self.v2
		vv3 = point - self.v3
		if np.abs(np.inner(self.norm,vv1)/lenth(vv1)) > 0.001:
			return False
		v21 = self.v1 - self.v2
		v32 = self.v2 - self.v3
		v13 = self.v3 - self.v1
		v31 = -v13
		v23 = -v32
		v12 = -v21
		if different_side(vv1,v12,v13) and different_side(vv2,v21,v23) and different_side(vv3,v31,v32):
			return True
		return False

"""
class Complex_Triangle(MyObject):
	def __init__()
"""

def Polynominal(v_list,texture_s = arr([1.,1.,1.]),texture_d = arr([0.4,0.3,0.9]),texture_a = arr([0.4,0.3,0.9])):
	"""
	Used to quickly construct a polynominal
	@ v_list 		: the vertices of polynominal
	@ texture 		: the texture of this object
	ATTENTION!
	This is just for convex polynominal
	return value
	@ objectlist 	: an objectlist full of triangles
	"""
	objectlist = []
	for i,_ in enumerate(v_list):
		if i < 2:
			continue
		objectlist.append(Triangle(v_list[0],v_list[i-1],v_list[i],texture_s,texture_d,texture_a))
	return objectlist

def get_ray_list(norm,number):
	"""
	calculate the ray list from norm
	input:
	@norm 	: norm
	@x_num	: how many rays in one theta
	@z_num	: how many theta
	output:
	@ray_list : ray list from norm
	"""
	x_num = number[0]
	z_num = number[1]
	ray_list = []
	theta_list = []
	alpha_list = []
	for i in range(z_num):
		theta_list.append(1.57/z_num*i)

	for i in range(x_num):
		alpha_list.append(6.28/x_num*i)

	for theta in theta_list:
		costheta = np.cos(theta)
		ray_x,ray_y = get_xy_ray(norm,costheta)
		if lenth(ray_x) and lenth(ray_y):
			continue
		for alpha in alpha_list:
			ray_list.append([norm+ray_x*np.cos(alpha)+ray_y*np.sin(alpha),costheta])

	return ray_list

def get_xy_ray(norm,costheta):
	"""
	Get the basis ray of given norm and costheta
	Input:
	@norm 	   : norm
	@costheta  : the cosine of theta
	Output:
	@new_vec_x : one basis ray
	@new_vec_y : another basis ray
	"""
	if check_value_zero(costheta):
		return np.array([0,0,0]),np.array([0,0,0])
	size2 = np.inner(norm,norm)
	if check_value_zero(size2):
		return np.array([0,0,0]),np.array([0,0,0])
	target_size = np.sqrt(size2)/costheta

	if not (check_value_zero(norm[0]) and check_value_zero(norm[1])):
		a2b2 = size2 - norm[2]*norm[2]
		t = np.sqrt((1-costheta*costheta)*size2/a2b2)
		new_vec_x = np.array([-norm[1],norm[0],0]) * t
		new_vec_y = np.array([norm[2]*norm[0],norm[2]*norm[1],-a2b2])
		new_vec_y = target_size * new_vec_y / np.inner(new_vec_y,new_vec_y)
	
	elif not (check_value_zero(norm[0]) and check_value_zero(norm[2])):
		a2c2 = size2 - norm[1]*norm[1]
		t = np.sqrt((1-costheta*costheta)*size2/a2c2)
		new_vec_x = np.array([-norm[2],0,norm[0]]) * t
		new_vec_y = np.array([norm[1]*norm[0],-a2c2,norm[1]*norm[2]])
		new_vec_y = target_size * new_vec_y / np.inner(new_vec_y,new_vec_y)
	
	else:
		b2c2 = size2 - norm[0]*norm[0]
		t = np.sqrt((1-costheta*costheta)*size2/b2c2)
		new_vec_x = np.array([0,-norm[2],norm[1]]) * t
		new_vec_y = np.array([-b2c2,norm[1]*norm[0],norm[2]*norm[0]])
		new_vec_y = target_size * new_vec_y / np.inner(new_vec_y,new_vec_y)
	
	return new_vec_x,new_vec_y

