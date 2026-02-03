# 基础模块

基础模块指 `AstraPy`。

## 轨道状态 OrbitState

### 构造函数

```python
@overload
def __init__(
    self, 
    position: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, '[3, 1]'] = ..., 
    velocity: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, '[3, 1]'] = ..., 
    epoch: time.Epoch = ..., 
    frame: frames.ReferenceFrameBase = ...
    ) -> None: ...
@overload
def __init__(
    self, 
    state: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, '[m, 1]'], 
    epoch: time.Epoch = ..., 
    frame: frames.ReferenceFrameBase = ...
    ) -> None: ...
```

`OrbitState` 可以使用位置、速度两个向量进行初始化，也可以使用位置速度长度为 6 的向量进行初始化。

参数 `epoch`，默认值为 `Astra.time.Epoch.J2000()`。

参数 `frame`，默认值为 `Astra.frames.ICRFFrame.instance`。

!!! example
    ```python
    r = [7000, 0, 0]
    v = [0, 6, 0]

    AstraPy.OrbitState(r, v)     // OK
    AstraPy.OrbitState([*r, *v]) // OK
    ```

### 属性

#### 位置

```python
position: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]']
```

轨道的位置，单位千米。

#### 速度

```python
velocity: typing.Annotated[numpy.typing.NDArray[numpy.float64], '[3, 1]']
```

轨道的速度，单位千米每秒。

#### 纪元

```python
epoch: time.Epoch
```

轨道状态对应的时刻。

#### 参考系

```python
frame: frames.ReferenceFrameBase
```

轨道状态表示的参考系。

### 方法

#### 转换到指定参考系

```python
def transform_to(self, target: frames.ReferenceFrameBase) -> OrbitState: ...
```

将当前轨道状态转换到指定参考系下，即得到指定参考系下的状态表示。

参数 `target` 为要转换到的参考系。