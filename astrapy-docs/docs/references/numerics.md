# 数值模块

Astra 的数值模块在 Python 中的绑定为 `AstraPy.numerics`。

## 地球球谐引力 EarthHarmonicGravity

地球球谐引力模型类，用于根据地球固定坐标系（ECEF）位置计算引力加速度。

### 构造函数

```python
EarthHarmonicGravity()
EarthHarmonicGravity(degree: int, order: int)
```

参数：

- `degree`：球谐展开的阶数（Degree）
- `order`：球谐展开的次数（Order）

### 方法

#### 得到引力加速度 get_acceleration

```python
get_acceleration(
    r_ecef: ArrayLike[float64, (3,1)]
) -> NDArray[float64, (3,1)]
```

计算指定位置处的地球引力加速度。

参数：

- `r_ecef`：ECEF 坐标系下的位置向量（单位：km）

返回：

- 引力加速度向量（单位：km/s²）

#### 设置球谐展开的阶数和次数 set_degree_order

```python
set_degree_order(degree: int, order: int) -> None
```

设置球谐模型的阶数与次数。

参数：

- `degree`：球谐展开的阶数（Degree）
- `order`：球谐展开的次数（Order）

## OrbitIntegrator

轨道数值积分器抽象基类，用于定义统一的积分接口。

### 构造函数

```python
OrbitIntegrator(*args, **kwargs)
```

用于初始化积分器对象，具体参数由派生类决定。

### 方法

#### integrate

```python
def integrate(self, 
    s0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, '[6, 1]'], 
    epoch: AstraPy.time.Epoch, 
    dt: typing.SupportsFloat, 
    f: collections.abc.Callable
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], '[6, 1]']: ...
```

参数：

- `s0`: 初始的轨道状态
- `epoch`: 初始的纪元
- `dt`: 要积分的时间（单位：s）
- `f`: 积分函数
  - 函数签名应为
    ```python
    f(x: ArrayLike[float64, (6,1)], epoch: Epoch) -> ArrayLike[float64, (6,1)]
    ```

返回：

- `dt` 时刻后的轨道状态

## RK4OrbitIntegrator

四阶 Runge-Kutta 定步长积分器。

```python
class RK4OrbitIntegrator(OrbitIntegrator):
    ...
```

### 属性

| 属性       | 类型    | 说明        |
| -------- | ----- | --------- |
| max_step | float | 最大积分步长（秒） |

### 构造函数

```python
RK4OrbitIntegrator()
```

## RKF78OrbitIntegrator

Runge-Kutta-Fehlberg 7/8 阶自适应步长积分器。

```python
class RKF78OrbitIntegrator(OrbitIntegrator):
    ...
```

### 属性

| 属性               | 类型            | 说明        |
| ---------------- | ------------- | --------- |
| abs_tol          | ndarray (6×1) | 状态量绝对误差容限 |
| rel_tol          | float         | 相对误差容限    |
| min_step         | float         | 最小步长      |
| max_step         | float         | 最大步长      |
| min_scale_factor | float         | 最小步长缩放因子  |
| max_scale_factor | float         | 最大步长缩放因子  |

### 构造函数

```python
RKF78OrbitIntegrator()
```

## VerletOrbitIntegrator

Verlet 定步长积分器（适用于保守系统）。

```python
class VerletOrbitIntegrator(OrbitIntegrator):
    ...
```

### 属性

| 属性       | 类型    | 说明     |
| -------- | ----- | ------ |
| max_step | float | 最大积分步长 |


### 构造函数

```python
VerletOrbitIntegrator()
```