from PIL import Image
import numpy as np
from utils import *
from Spacial import *
import time
from load_model import *
import sys
"""
arr = np.array

my_space = Spacial()
model_list,max_index,min_index = get_model('test.obj',detail=True,specular = True)
my_space.AddObjects(model_list)
Light = Parallel_light(orientation = arr([-1.,0.,-1.]))
my_space.AddLight(Light)
camera = Camera(position = arr([1000,500,0]),rotation = arr([[- np.pi / 2.,0.,np.pi]]), resolution_width = 20, resolution_height = 20, width = 1000, height = 1000,orth = True)
my_space.setCamera(camera)
render = my_space.Render()
im = Matrix2Image(render)
im.show()
path_to_save = 'my_space_final.png'
im.save(path_to_save)
"""

orientation = np.array([0,0,np.pi/2])
a = np.array([1.,0.,0.])
print(Rotate(a,orientation))

"""
my_circle = Circle(np.array([0,0,0]),2,np.array([1.,1.,1.]),np.array([1.,0.,0.]),np.array([1.,0.,0.]))

my_circle2 = Circle(np.array([0,0,6]),3,np.array([1.,1.,1.]),np.array([0.,1.,0.]),np.array([0.,1.,0.]))

my_Triangle = Polynominal([arr([-10,10,-10]),arr([-10,10,10]),arr([-10,-10,10]),arr([-10,-10,-10])],np.array([1.,1.,1.]),np.array([1.,1.,1.]),np.array([1.,1.,1.]))

my_cube = Cube(arr([0,0,0]), 10,10,10, rotation = arr([0.,0.90,0.]))

#my_space.AddObject(my_circle)

Light_1 = PointLight(arr([0.,0.,40.]))
Light_2 = Parallel_light(orientation=arr([-1.,0.,-1.]))

my_space.AddLight(Light_2)

#my_space.AddObject(my_circle2)

#my_space.AddObjects(my_Triangle)

my_space.AddObjects(my_cube)

before = time.time()
render = my_space.Render(orth = True, resolution_width = 200, resolution_height = 200, width = 20, height = 20)
after = time.time()

print("Total time used to render: {}".format(after-before))

im = Matrix2Image(render)

im.show()

im.save(path_to_save)

#my_space.render_test(orth = True, y = 0, z = 9.5)
"""

"""
my_space = Spacial()

plane1 = Cube(arr([0,0,30]),10,80,10)
plane2 = Cube(arr([0,0,50]),10,60,10)
cube1 = Cube(arr([0,-15,40]),10,10,10)
cube2 = Cube(arr([0,15,40]),10,10,10)
cube3 = Cube(arr([0,-20,12.5]),10,10,25)
cube4 = Cube(arr([0,20,12.5]),10,10,25)
surface = Plane(arr([0.,0.,-5.]),arr([0.,0.,1.]),texture_s = arr([1.,1.,1.]), texture_d = arr([1.,1.,1.]), texture_a = arr([1.,1.,1.]), ratio_s = 0.2, ratio_d = 0.8, ratio_a = 0.0, specular = True, decay = 0.5)

#new_triangle = CompleteTriangle(arr([0.,-20.,-20.]),arr([0.,20.,-20.]),arr([0.,0.,20.]),arr([1.,-1.,0.]),arr([1.,1.,0.]),arr([1.,0.,1.]),
#	texture_s = arr([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]),texture_d = arr([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]),texture_a = arr([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]))

ci = Circle(position = arr([0.,0.,5.]),radius = 5,texture_a = arr([1.,0.,0.]), texture_d = arr([1.,0.,0.]))

my_camera = Camera(position = arr([50,0,10]))

Light = PointLight(arr([70,0,70]))

Light_2 = Parallel_light(orientation = arr([-1,0,-1]))

#my_space.AddObject(ci)



my_space.AddObjects(plane1)
my_space.AddObject(surface)
my_space.AddObjects(plane2)
my_space.AddObjects(cube1)
my_space.AddObjects(cube2)
my_space.AddObjects(cube3)
my_space.AddObjects(cube4)
my_space.AddObject(ci)

#my_space.AddObject(new_triangle)
my_space.AddLight(Light_2)

my_space.setCamera(my_camera)

#my_space.render_test(orth = True, y = 0, z = 0)

before = time.time()
render = my_space.Render(orth = False, resolution_width = 1920, resolution_height = 1080, width = 160, height = 90, number = [1,1], max_iter = 2)
after = time.time()

print("Total time used to render: {}".format(after-before))

im = Matrix2Image(render)

im.show()

path_to_save = 'my_space_spe.png'

im.save(path_to_save)

"""