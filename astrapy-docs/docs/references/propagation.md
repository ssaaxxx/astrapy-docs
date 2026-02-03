# 轨道递推模块

## 轨道递推器 Propagator

抽象基类。

### 方法

#### 轨道递推 Propagate

```python
def Propagate(self, init_state: AstraPy.OrbitState, dt: typing.SupportsFloat) -> AstraPy.OrbitState: ...
```

参数：

- `init_state`: 初始状态
- `dt`: 轨道递推的时间

## 二体轨道递推器 TwoBodyPropagator

继承自 `Propagator`，用于二体轨道递推。

### 属性

| 属性       | 类型    | 说明        |
| -------- | ----- | --------- |
| `max_iterations` | int | 最大迭代步数 |
| `tolerance` | float | 容差 |
| `mu` | float | 引力常数 |

### 构造函数

```python
@overload
def __init__(self, mu: typing.SupportsFloat) -> None: ...
@overload
def __init__(self) -> None: ...
```

参数：

- `mu`: 引力常数

## 引力轨道递推器 GravityPropagator

继承自 `Propagator`，用于球谐摄动的引力递推。

### 属性

| 属性       | 类型    | 说明        |
| -------- | ----- | --------- |
| `degree` | int | 球谐展开的阶数（Degree） |
| `order` | int | 球谐展开的次数（Order） |

### 构造函数

```python
def __init__(self) -> None: ...
```

### 方法

#### 设置球谐展开的阶数和次数 set_degree_order

```python
def set_degree_order(self, degree: typing.SupportsInt, order: typing.SupportsInt) -> None: ...
```

设置计算引力摄动使用的球谐展开的阶数和次数。

参数：

- `degree`：球谐展开的阶数（Degree）
- `order`：球谐展开的次数（Order）

#### 设置积分器 set_integrator

```python
@overload
def set_integrator(self, integrator: str) -> None: ...
@overload
def set_integrator(self, integrator: AstraPy.numerics.OrbitIntegrator) -> None: ...
```

设置轨道递推使用的积分器。

参数：

- `integrator`（`str` 类型，不区分大小写），支持的积分器有：
    - `rk4`
    - `verlet`
    - `rkf78`
- `integrator`（`OrbitIntegrator` 类型）：使用的积分器