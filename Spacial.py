"""
@Author: Chaoyu Guan
Realization of Ray tracing and 3D constructing
"""
import numpy as np
from utils import *
from tqdm import tqdm

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

	def __init__(self,position = arr([50.,0.,0.]), rotation = arr([[0.,0.,np.pi]]), PosDepth = 20, PostDepth = -1):
		self.position = position
		self.rotation = rotation
		self.pos = PosDepth
		self.post = PostDepth

	def setPosition(self,position):
		self.position = position

	def setRotation(self,rotation):
		self.rotation = rotation

	def setPosDepth(self,PosDepth):
		self.pos = PosDepth

	def setPostDepth(self,PostDepth):
		self.post = PostDepth

class Spacial():
	"""
	Spacial domain realization
	Parameters:
	@ camera 		: The camera used in this space, one space must and only have one camera
	@ objectlists	: The object list in this space, object must be the <class 'MyObject'> type
	@ lightlist		: The light in this space, object must be the <class 'light'> type
	"""

	def __init__(self):
		self.camera = Camera()
		self.objectlists = []
		self.lightlist = []

	def setCamera(self, camera):
		self.camera = camera

	def AddObject(self, obj):
		"""
		Add object to this space
		@ obj 	: MyObject type
		"""
		self.objectlists.append(obj)

	def AddObjects(self, objects):
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

	def Render(self, orth=True, resolution_width = 200, resolution_height = 200, width = 10, height = 10, number = [1,1], max_iter = 5):
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
		print("Start Rendering")
		pic_matrix = np.zeros([resolution_height,resolution_width,3])
		print("Get the matrix size:\nresolution height\t%d\tresolution width%d\nwindows height\t%dwindows width\t%d" 
			% (resolution_height,resolution_width,height,width))
		for item in tqdm(range(resolution_height * resolution_width)):
			i = int(item / resolution_height)
			j = item - resolution_height * i
			x_idx = self.camera.pos
			y_idx = (i - resolution_width/2) / resolution_width * width
			z_idx = (-j + resolution_height/2) / resolution_height * height
			orth_idx = arr([x_idx,y_idx,z_idx])
			orientation = arr([1,0,0])
			for ori in self.camera.rotation:
				orth_idx = Rotate(orth_idx, ori)
				orientation = Rotate(orientation, ori)
			orth_idx += self.camera.position
			if not orth:
				pic_matrix[j][i][:] = self.ray_tracing(self.camera.position,np.array(orth_idx-self.camera.position),0,max_iter,number)
			else:
				pic_matrix[j][i][:] = self.ray_tracing(orth_idx,orientation,0,max_iter,number)
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
		current_depth = self.camera.post if self.camera.post > mini else 100000
		for i in range(len(self.objectlists)):
			obj = self.objectlists[i]
			# Calculate intersect with obj
			tmp_depth,point,After_ray_list,norm,inter,textures = obj.intersect(start,vec,number)
			if not inter:
				continue
			if tmp_depth < current_depth:
				# Refresh the depth
				current_depth = tmp_depth
				# Calculate specular light
				sp_color = arr([0.,0.,0.])
				for lit in self.lightlist:
					sp_color += lit.calculate(point,vec,norm,self.objectlists,number)
				sp_color = get_min(sp_color,arr([1.,1.,1.])) * textures[0] * textures[0]
				# Calculate diffusion light
				di_color = arr([0.,0.,0.])
				for ray in After_ray_list:
					di_color += self.ray_tracing(point,ray[0],iters+1,max_iter,number) * ray[1] * textures[1]
				di_color = di_color * obj.rd
				# Calculate ambient light
				am_color = textures[2] * obj.ra

		return get_min(di_color * 1.  + am_color * 1. + sp_color * 1., np.array([1.,1.,1.]))


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
			tmp_depth, _, _, _, inter,_ = obj.intersect(point,ray_to_light,number)
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
			tmp_depth, _, _, _, inter,_ = obj.intersect(point,ray_to_light,number)
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
	def __init__(self,texture_s,texture_d,texture_a,ratio_s, ratio_d, ratio_a):
		super(MyObject,self).__init__()
		self.ts = texture_s
		self.td = texture_d
		self.ta = texture_a
		self.rs = ratio_s
		self.rd = ratio_d
		self.ra = ratio_a

	def setTexture_s(self,texture_s):
		self.ts = texture_s

	def setTexture_d(self,texture_d):
		self.td = texture_d

	def setTexture_a(self,texture_a):
		self.ta = texture_a

	def setRatio_s(self, ratio_s):
		self.rs = ratio_s

	def setRatio_d(self, ratio_d):
		self.rd = ratio_d

	def setRatio_a(self, ratio_a):
		self.ra = ratio_a

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
		texture = arr([self.ts,self.td,self.ta])
		return t,interpoint,ray_list,norm,Inter,texture


class Circle(MyObject):
	"""
	Sphere object
	Parameters:
	@ position 			: np.array, size = 3, center point of this shpere
	@ r 				: double, the radius of sphere
	"""
	def __init__(self,position,radius,texture_s = arr([1.,1.,1.]),texture_d = arr([1.,1.,0.]),texture_a = arr([1.,1.,1.]), ratio_s = 0.5, ratio_d = 0.4, ratio_a = 0.1):
		MyObject.__init__(self,texture_s,texture_d,texture_a, ratio_s, ratio_d, ratio_a)
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
			return 0,arr([0.,0.,0.]),[],arr([0.,0.,0.]),False,[]
		t1 = - dv - np.sqrt(delta)
		t2 = - dv + np.sqrt(delta)
		if t2 < mini:
			#print("No inter 2")
			return 0,arr([0.,0.,0.]),[],arr([0.,0.,0.]),False,[]
		elif t1 < mini:
			t = t2
		else:
			t = t1
		circle_point = start + vec * t
		norm = self.get_norm(circle_point)
		norm = normal(norm)
		ray_list = get_ray_list(norm,number)
		return t,circle_point,ray_list,norm,True,arr([self.ts,self.td,self.ta])
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
	def __init__(self,v1,v2,v3,texture_s = arr([1.,1.,1.]),texture_d = arr([1.,0.,1.]),texture_a = arr([1.,0.,1.]), ratio_s = 0.5, ratio_d = 0.4, ratio_a = 0.1):
		MyObject.__init__(self,texture_s,texture_d,texture_a, ratio_s, ratio_d, ratio_a)
		self.v1 = v1
		self.v2 = v2
		self.v3 = v3
		self.norm = np.cross((v2-v1)/100,(v3-v2)/100)
		self.norm = normal(self.norm)

	def intersect(self,start,vec,number):
		if np.abs(np.inner(vec,self.norm)/lenth(vec)) < 0.001:
			return 0,arr([0.,0.,0.]),[],arr([0.,0.,0.]),False,[]
		if self.ContainPoint(start):
			return 0,arr([0.,0.,0.]),[],arr([0.,0.,0.]),False,[]
		else:
			t = np.inner((self.v1 - start),self.norm)/np.inner(self.norm,vec)
			if t < 0:
				return 0,arr([0.,0.,0.]),[],arr([0.,0.,0.]),False,[]
			point = start + t * vec
			if self.ContainPoint(point):
				ray_list = get_ray_list(self.norm,number)
				return t,point,ray_list,self.norm,True,arr([self.ts,self.td,self.ta])
			return 0,arr([0.,0.,0.]),[],arr([0.,0.,0.]),False,[]


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

class CompleteTriangle(MyObject):
	def __init__(self, v1, v2, v3, norm1, norm2, norm3, texture_s = arr([[0.,1.,0.]]*3), texture_d = arr([[0.,1.,0.]]*3), texture_a = arr([[0.,1.,0.]]*3), ratio_s = 0.5, ratio_d = 0.4, ratio_a = 0.1):
		if texture_s.shape != (3,3):
			raise ValueError("texture_s size is wrong! The input texture is: "+str(texture_s))
		if texture_d.shape != (3,3):
			raise ValueError("texture_d size is wrong! The input texture is: "+str(texture_d))
		if texture_a.shape != (3,3):
			raise ValueError("texture_a size is wrong! The input texture is: "+str(texture_a))
		super(CompleteTriangle,self).__init__(texture_s,texture_d,texture_a,ratio_s,ratio_d,ratio_a)
		self.v1 = v1
		self.v2 = v2
		self.v3 = v3
		self.norm = np.cross((v2-v1)/100,(v3-v2)/100)
		self.norm = normal(self.norm)
		self.norm1 = norm1
		self.norm2 = norm2
		self.norm3 = norm3

	def intersect(self,start,vec,number):
		if np.abs(np.inner(vec,self.norm)/lenth(vec)) < 0.001:
			return 0,arr([0.,0.,0.]),[],arr([0.,0.,0.]),False,[]
		if self.ContainPoint(start):
			return 0,arr([0.,0.,0.]),[],arr([0.,0.,0.]),False,[]
		else:
			t = np.inner((self.v1 - start),self.norm)/np.inner(self.norm,vec)
			if t < 0:
				return 0,arr([0.,0.,0.]),[],arr([0.,0.,0.]),False,[]
			point = start + t * vec
			if self.ContainPoint(point):
				if same_point(point,self.v1):
					pnorm = self.norm1
					texture = np.array([self.ts[0],self.td[0],self.ta[0]])
				elif same_point(point,self.v2):
					pnorm = self.norm2
					texture = np.array([self.ts[1],self.td[1],self.ta[1]])
				elif same_point(point,self.v2):
					pnorm = self.norm3
					texture = np.array([self.ts[2],self.td[2],self.ta[2]])
				else:
					pnorm, texture = self.get_norm_and_color(point)
				ray_list = get_ray_list(pnorm,number)
				return t,point,ray_list,pnorm,True,texture
			return 0,arr([0.,0.,0.]),[],arr([0.,0.,0.]),False,[]

	def get_norm_and_color(self, point):
		veca = point - self.v1
		vecb = self.v3 - self.v2
		f,s = [0,1]
		if check_value_zero(vecb[f]*veca[s] - vecb[s]*veca[f]):
			f,s = [0,2]
			if check_value_zero(vecb[f]*veca[s] - vecb[s]*veca[f]):
				f,s = [1,2]
				if check_value_zero(vecb[f]*veca[s] - vecb[s]*veca[f]):
					raise ValueError("Wrong point found with " +str(vecb)+str(veca) +str(point) + str(self.v1) + str(self.v2) + str(self.v3))
		alpha = ((self.v2[s] - self.v1[s]) * vecb[f] + (self.v1[f] - self.v2[f]) * vecb[s]) / (vecb[f]*veca[s] - vecb[s]*veca[f])
		if check_value_zero(alpha):
			raise ValueError("Wrong alpha found with " + str([f,s]) + str(point) + str(self.v1) + str(self.v2) + str(self.v3))
		cal = f
		if check_value_zero(vecb[cal]):
			cal = s
			if check_value_zero(vecb[cal]):
				raise ValueError("Wrong beta found with " + str(point) + str(self.v1) + str(self.v2) + str(self.v3))
		beta = (self.v1[cal] + alpha * veca[cal] - self.v2[cal]) / vecb[cal]
		alpha = 1 / alpha
		norm = (1 - alpha) * self.norm1 + alpha * (beta * self.norm3 + (1-beta) * self.norm2)
		texture_s = (1 - alpha) * self.ts[0] + alpha * (beta * self.ts[2] + (1-beta) * self.ts[1])
		texture_d = (1 - alpha) * self.td[0] + alpha * (beta * self.td[2] + (1-beta) * self.td[1])
		texture_a = (1 - alpha) * self.ta[0] + alpha * (beta * self.ta[2] + (1-beta) * self.ta[1])
		texture = arr([texture_s,texture_d,texture_a])
		return norm, texture


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


class Plane(MyObject):
	def __init__(self, point, norm, texture_s = arr([1.,1.,1.]), texture_d = arr([0.5,0.,0.9]),texture_a = arr([0.5,0.,0.9]), ratio_s = 0.5, ratio_d = 0.4, ratio_a = 0.1):
		super(Plane,self).__init__(texture_s, texture_d, texture_a, ratio_s, ratio_d, ratio_a)
		self.v = point
		self.norm = normal(norm)

	def intersect(self,start,vec,number):
		if np.abs(np.inner(vec,self.norm)/lenth(vec)) < 0.001:
			return 0,arr([0.,0.,0.]),[],arr([0.,0.,0.]),False,[]
		if same_point(start,self.v):
			return 0,arr([0.,0.,0.]),[],arr([0.,0.,0.]),False,[]
		if check_value_zero(np.inner(start - self.v, self.norm)):
			return 0,arr([0.,0.,0.]),[],arr([0.,0.,0.]),False,[]
		else:
			t = np.inner((self.v - start),self.norm)/np.inner(self.norm,vec)
			if t < mini:
				return 0,arr([0.,0.,0.]),[],arr([0.,0.,0.]),False,[]
			point = start + t * vec
			ray_list = get_ray_list(self.norm,number)
			return t,point,ray_list,self.norm,True,arr([self.ts,self.td,self.ta])

def Complete_Polynominal(v_list,norm_list,texture_s = arr([[1.,1.,1.]]*3),texture_d = arr([[0.4,0.3,0.9]]*3),texture_a = arr([[0.4,0.3,0.9]]*3),ratio_s = 0.5, ratio_d = 0.4, ratio_a = 0.1):
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
		objectlist.append(CompleteTriangle(v_list[0],v_list[i-1],v_list[i],norm_list[0],norm_list[i-1],norm_list[i],texture_s = texture_s,
			texture_d = texture_d, texture_a = texture_a,ratio_s = ratio_s,ratio_d = ratio_d,ratio_a = ratio_a))
	return objectlist

def Polynominal(v_list,texture_s = arr([1.,1.,1.]),texture_d = arr([0.4,0.3,0.9]),texture_a = arr([0.4,0.3,0.9]),ratio_s = 0.5, ratio_d = 0.4, ratio_a = 0.1):
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
		objectlist.append(Triangle(v_list[0],v_list[i-1],v_list[i],texture_s,texture_d,texture_a,ratio_s,ratio_d,ratio_a))
	return objectlist

def Cube(position, lenths, width, height, rotation = arr([0.,0.,0.]), texture_s = arr([1.,1.,1.]), texture_d = arr([0.5,0.5,1.]), texture_a = arr([0.5,0.5,1.]),ratio_s = 0.5, ratio_d = 0.4, ratio_a = 0.1):
	mid_lenth = lenths/2
	mid_width = width/2
	mid_height = height/2
	v1 = arr([mid_lenth,-mid_width,mid_height])
	v2 = arr([mid_lenth,mid_width,mid_height])
	v3 = arr([mid_lenth,mid_width,-mid_height])
	v4 = arr([mid_lenth,-mid_width,-mid_height])
	v5 = -v3
	v6 = -v4
	v7 = -v1
	v8 = -v2
	v1,v2,v3,v4,v5,v6,v7,v8 = Rotate(arr([v1,v2,v3,v4,v5,v6,v7,v8]),rotation) + position
	objlist = np.append(Polynominal([v1,v4,v3,v2,v6,v5,v8,v4],texture_s,texture_d,texture_a,ratio_s,ratio_d,ratio_a),
		Polynominal([v7,v6,v2,v3,v4,v8,v5,v6],texture_s,texture_d,texture_a,ratio_s,ratio_d,ratio_a))
	return objlist



def Rotate(vertex, rotation):
	vec_x = arr([0,np.sin(rotation[0]),np.cos(rotation[0])])
	vec_y = arr([np.sin(rotation[1]),0,np.cos(rotation[1])])
	vec_z = arr([np.cos(rotation[2]),np.sin(rotation[2]),0])
	vertex = arr([[1.,0.,0.],[0.,vec_x[2],vec_x[1]],[0.,-vec_x[1],vec_x[2]]]).dot(vertex.T)
	vertex = arr([[vec_y[2],0.,vec_y[1]],[0.,1.,0.],[-vec_y[0],0.,vec_y[2]]]).dot(vertex)
	vertex = arr([[vec_z[0],vec_z[1],0.],[-vec_z[1],vec_z[0],0.],[0.,0.,1.]]).dot(vertex)
	return vertex.T


def get_ray_list(norm, number):
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
		theta_list.append(np.pi / (2 * z_num)*i)

	for i in range(x_num):
		alpha_list.append(2 * np.pi /x_num*i)

	for theta in theta_list:
		costheta = np.cos(theta)
		ray_x,ray_y = get_xy_ray(norm,costheta)
		if check_vec_zero(ray_x) and check_vec_zero(ray_y):
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

