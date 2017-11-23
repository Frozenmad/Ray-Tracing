# Ray-Tracing 光线追踪算法实现
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
## 环境依赖项
* *python 2.7 or 3.x*
* *numpy*
* *PIL*
## 使用简易教程
### Step 0 环境准备
最开始当然是一些引用项：
```
from PIL import image
import numpy as np
from Spacial import *
from utils import *
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
之后你可以通过调用`my_space.setCameraParameter()`函数来修改默认相机的参数

或者直接使用`my_space.setCameraPosition()`函数来修改相机的位置，该相机默认是在空间的`\(50,0,0\)`的位置上

### Step 2 新建并添加几何体
接下来，你可以新建一些几何体，举个例子：
```
my_sphere = Circle(arr([0,0,0]),2))
my_quad = Polynominal([arr([-10,10,-10]),arr([-10,10,10]),arr([-10,-10,10]),arr([-10,-10,-10])])
```
目前软件可以支持的几何体有：
* 球体  `Circle()`
* 三角面片  `Triangle()`
* 凸多边形  `Polynominal()`

相关参数请参照`Spacial.py`的注释

之后要将这些几何体加入到先前定义的场景中：
```
my_space.AddObject(my_shpere)
my_space.AddObjects(my_quad)
```
请注意这里加入Polynominal的物体时，需要用AddObjects函数，具体原因请参考`Spacial.py`的实现细节

### Step 3 新建并添加光源 (Optional)
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

### Step 4 渲染
调用`my_space.Render()`即可渲染，改变参数可以调整是正投影还是透视投影，分辨率和窗口大小等等，该函数返回的是一个float类型数编码的RGB矩阵
```
im_matrix = my_space.Render(orth = True, resolution_width = 200, resolution_height = 200, width = 20, height = 20)
```

### Step 5 显示与保存
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
