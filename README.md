# Ray-Tracing 光线追踪算法实现 [README_english](English version)
_2017 fall SJTU CG course assignment2_
## 版权信息
作者：关超宇

E-mail : frozenmad2015@outlook.com

License : [MIT License](LICENSE)

> 注：引用或转载请注明出处

## 相关文件
##### `main.py`
运行时的主程序，请将修改代码增添至该文件中
##### `Spacial.py`
空间、几何体、光源的实现代码
##### `utils.py`
一些功能函数的封装
##### `load_model.py`
从obj文件中构建模型
## 环境依赖项
* *python 2.7 or 3.x*
* *numpy*
* *PIL*
* *tqdm*
## 使用简易教程
### Step 0 环境准备
最开始当然是一些引用项：
```
from PIL import image
import numpy as np
from Spacial import *
from utils import *
# 如果需要从.obj文件导入几何模型，那么还需要包含如下环境
# from load_model.py import *
```
通过重命名`np.array()`函数来得到更简洁的语法：
```
arr = np.array
```
接下来步入正题
### Step 1 新建空间并设置参数
首先，你需要新建一个绘制物体的空间
```
my_space = Spacial()
```
### Step 2 新建并设置相机
每一个空间都有一个默认的相机，该相机位于空间的`(50,0,0)`的位置上，方向向量为：`(-1,0,0)`，up向量为`(0,0,1)`，前景面为20，后景面无穷远。

如果需要自己设置相机的参数，可以通过新建相机并设置为空间的相机来达到目的
```
my_camera = Camera(position = arr([50,0,10]), orientation = arr([[np.pi/2,0,np.pi]]))
my_space.setCamera(my_camera)
```
`Camera`类的相关参数详见`Spacial.py`
### Step 2 新建并添加几何体
接下来，你可以新建一些几何体，举个例子：
```
my_sphere = Circle(arr([0,0,0]),2))
my_quad = Polynominal([arr([-10,10,-10]),arr([-10,10,10]),arr([-10,-10,10]),arr([-10,-10,-10])])
```
目前软件可以支持的几何体有：
###### 原生几何体
* 球体  `Circle()`
* 简易三角面片  `Triangle()`
* 三角面片 `CompleteTriangle()`
* 平面 `Plane()`
###### 组合几何体
* 简单多边形  `Polynominal()`
* 多边形 `Complete_Polynominal()`
* 立方体 `Cube()`
* 从obj文件导入 `get_model(path)`  # 此项需要import load_model.py

相关参数请参照`Spacial.py`的注释，原生几何体需要通过`my_space.AddObject()`添加入场景中，而组合几何体需要通过`my_space.AddObjects()`来添加。

之后要将这些几何体加入到先前定义的场景中：
```
my_space.AddObject(my_shpere)
my_space.AddObjects(my_quad)
```
### Step 4 新建并添加光源 (Optional)
这个可选步骤是添加光源，没有光源的话，也可以进行渲染，但这样渲染出来的只有物体的泛在光，效果并不好，所以加入光源是一种不错的选择。

首先定义一个光源：
```
Light = PointLight(arr([0.,0.,40.]))
```
目前程序支持的光源类型有：
* 点光源 `PointLight()`
* 平行光源 `Parallel_light()`

相关参数请参照`Spacial.py`的注释

之后同样要将光源加入场景中：
```
my_space.AddLight(Light)
```
你也可以调用`my_space.AddLights()`来一次性加入多个光源（一个light_list）

### Step 5 渲染
调用`my_space.Render()`即可渲染，改变参数可以调整是正投影还是透视投影，分辨率和窗口大小等等，该函数返回的是一个float类型数编码的RGB矩阵
```
im_matrix = my_space.Render(orth = True, resolution_width = 200, resolution_height = 200, width = 20, height = 20)
```

### Step 6 显示与保存
可以利用`utils.py`中提供的函数`Matrix2Image()`来实现RGB矩阵与image的图片类型的转换，之后可以利用PIL库来执行图片的显示于保存操作
```
im = Matrix2Image(im_matrix)
im.show()
im.save('example.png')
```
## 相关说明
由于程序采用python写成，仅提供光线追踪算法思想的学习，并没有使用GPU，所以导致渲染速度较慢。

示例程序的环境中一共有两个球，两个三角面片，一个光源，分辨率为200\*200，经计算渲染所需时间为50s左右。

**再次声明，转载与引用请注明出处~**
