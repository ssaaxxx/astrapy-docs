# 参考系模块

Astra 的参考系模块在 Python 中的绑定为 `AstraPy.frames`。

## 模块属性

```python
ecef: ITRFFrame
eci: ICRFFrame
icrf: ICRFFrame
itrf: ITRFFrame
```

`frames` 模块中的 `icrf`、`eci` 和 `ICRFFrame.instance` 相同，`itrf`、`ecef` 和 `ITRFFrame.instance` 相同。

## 基础参考系 ReferenceFrameBase

基础参考系是 Astra 中所有参考系的父类。

### 属性

#### 参考系名称

```python
@property
def name(self) -> str: ...
```

该参考系的名称。

#### 父参考系

```python
@property
def parent(self) -> ReferenceFrameBase: ...
```

该参考系的父参考系。

### 方法

#### 坐标变换

```python
def rotation_to_parent(self, epoch: AstraPy.time.Epoch) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 3]']: ...
def translation_to_parent(self, epoch: AstraPy.time.Epoch) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]']: ...
def angular_velocity_to_parent(self, epoch: AstraPy.time.Epoch) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]']: ...
def translation_velocity_to_parent(self, epoch: AstraPy.time.Epoch) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]']: ...
```

方法 `rotation_to_parent` 返回该参考系转换到父参考系的坐标转换矩阵（旋转矩阵）。

方法 `translation_to_parent` 返回该参考系转换到父参考系的平移向量，即该参考系原点在父参考系中的表示。

方法 `angular_velocity_to_parent` 返回该参考系相对于父参考系的旋转角速度。该角速度是在父参考系下的表示。

方法 `translation_velocity_to_parent` 返回该参考系相对于父参考系的平移速度。该速度是在父参考系下的表示。

#### 创建轨道状态

```python
def create_orbit_state(
    self, 
    position: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, '[3, 1]'], 
    velocity: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, '[3, 1]'] = ..., 
    epoch: AstraPy.time.Epoch = ...
    ) -> AstraPy.OrbitState: ...
```

创建在该参考系下的轨道状态。

参数 `position` 为位置向量；参数 `velocity` 为速度向量，默认值为零向量；参数 `epoch` 为状态的时刻，默认值为 J2000 时刻。

#### 得到从当前参考系到根的参考系列表

```python
def path_to_root(self) -> list[ReferenceFrameBase]: ...
```

方法返回从当前参考系到根参考系的列表。

!!! example
    以地面站局部参考系为例
    ```python
    print(AstraPy.frames.GroundStationFrame(0, 0, 0).path_to_root())
    ```
    将会打印
    ```
    [GroundStationFrame(lat=0, lon=0, alt=0), ITRFFrame(), ICRFFrame()]
    ```

## 国际天球参考架 ICRFFrame

`ICRFFrame` 继承自 `ReferenceFrameBase`，其静态属性 `instance` 为该参考系的唯一实例。

AstraPy 提供的 ICRF 以地球球心为原点。该参考系与 J2000 仅有一个小的常值偏差矩阵 $B$。

??? note "国际天球参考架（International Celestial Reference Frame，ICRF）"
    国际天球参考架是天体测量学使用电波观测到的参考源实现的国际天球参考系（ICRS）。在国际天球参考系的上下文中，参考架是参考系的物理实现，即参考架是参考源报告的坐标集，使用的是国际天球参考系所规定的程式。

    国际天球参考架是以太阳系质心为中心的准惯性参考系，它的轴心是星系天文学使用特长基线干涉测量法进行观测，测量所得到的外星系（主要是类星体）位置来定义的。虽然广义相对论意味着在万有引力天体的周围没有真正的惯性架，但是国际天球参考架是很重要的，因为用于定义参考架的天体距离都非常遥远，因此它没有表现出任何可测量的角运动。现在，国际天球参考架是用于定义行星（包括地球）和其它天体位置的标准参考架。

## 国际地球参考架 ITRFFrame

`ITRFFrame` 继承自 `ReferenceFrameBase`，其静态属性 `instance` 为该参考系的唯一实例。

ITRF 原点在地球质心（包含大气海洋等质量），坐标系 $xOy$ 平面为地球赤道面，$z$ 轴指向北极 CIO 处，$x$ 轴指向格林威治子午线与赤道面交点处。此坐标系固定在地球上，地面站测控，以及地球引力场系数等都在此坐标系下定义。

??? note "国际地球参考框架（International Terrestrial Reference Frame，ITRF）"
    国际地球参考框架由国际地球自转与参考系统服务提供相关参数定义，由国际大地测量学和地球物理学联合会、国际大地测量协会以及国际天文学会共同建立，是目前精度最高、应用范围最广的地球参考框架，为其他地球参考框架提供高精度基准。

## 地面站局部参考系 GroundStationFrame

该坐标系即东-北-天坐标系，以站心为坐标系原点 $O$，$Z$ 轴与椭球的法线重合，向上为正，$X$ 轴为站心点的正东方向，$Y$ 轴为站心点的正北方向。

### 构造函数

```python
def __init__(self, lat_rad: typing.SupportsFloat, lon_ard: typing.SupportsFloat, alt_km: typing.SupportsFloat) -> None: ...
```

参数 `lat_rad` 为站心的地理纬度，参数 `lon_rad` 为站心的地理经度，单位均为弧度。

参数 `alt_km` 为站心的高程，单位为千米。

### 属性

```python
@property
def altitude(self) -> float: ...
@property
def latitude(self) -> float: ...
@property
def longitude(self) -> float: ...
```

`altitude`、`latitude` 和 `longitude` 分别为地面站站心的高程、地理纬度和地理经度。

地理纬度是定义在参考椭圆上的，即参考椭球上一点的法线与赤道平面的夹角。