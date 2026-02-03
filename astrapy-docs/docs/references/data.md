# 数据模块

## 地球定向参数 EarthOrientationParameters

抽象类。

### 静态方法

#### 得到默认的参数

```python
@staticmethod
def get() -> EarthOrientationParameters: ...
```

得到当前的地球定向参数。

#### 设置参数为默认

```python
@staticmethod
def set(eop: EarthOrientationParameters) -> None: ...
```

设置 `eop` 为默认的地球定向参数。

### 方法

#### 得到 dut1

```python
def get_dut1(self, mjd_utc: typing.SupportsFloat) -> float: ...
```

得到对应 UTC 时刻（修正儒略日表示）的 UT1 与 UTC 的差值的，单位为秒。

#### 得到极移值

```python
def get_Xp(self, mjd_utc: typing.SupportsFloat) -> float: ...
def get_Yp(self, mjd_utc: typing.SupportsFloat) -> float: ...
```

得到对应 UTC 时刻（修正儒略日表示）的 $x$ 向和 $y$ 向的极移值，单位为弧度。

#### 得到章动值

```python
def get_dX(self, mjd_utc: typing.SupportsFloat) -> float: ...
def get_dY(self, mjd_utc: typing.SupportsFloat) -> float: ...
```

得到对应 UTC 时刻（修正儒略日表示）的 $x$ 向和 $y$ 向的章动值，单位为弧度。

#### 得到日长变化

```python
def get_lod(self, mjd_utc: typing.SupportsFloat) -> float: ...
```

得到对应 UTC 时刻（修正儒略日表示）的日长变化（Length Of Day, LOD），单位为秒。

## 默认地球定向参数 DefaultEarthOrientationParameters

继承自 `EarthOrientationParameters`。

默认的地球定向参数。

```python
def get_Xp(self, mjd_utc: typing.SupportsFloat) -> float: ...
def get_Yp(self, mjd_utc: typing.SupportsFloat) -> float: ...
def get_dX(self, mjd_utc: typing.SupportsFloat) -> float: ...
def get_dY(self, mjd_utc: typing.SupportsFloat) -> float: ...
def get_dut1(self, mjd_utc: typing.SupportsFloat) -> float: ...
def get_lod(self, mjd_utc: typing.SupportsFloat) -> float: ...
```

上述函数都返回 0。

## 来自文件的地球定向参数 RapidBulletinEarthOrientationParameters

继承自 `EarthOrientationParameters`。

读取 `eop/rapid` 文件的地球定向参数。

### 构造函数

```python
def __init__(self, filepath: str) -> None: ...
```

从 `filepath` 文件读取数据作为地球定向参数。

!!! important
    可以从 [https://datacenter.iers.org/products/eop/rapid/](https://datacenter.iers.org/products/eop/rapid/) 获取指定的 rapid 文件。

    应该使用 finals2000A.all 文件，即 [https://datacenter.iers.org/products/eop/rapid/standard/finals2000A.all](https://datacenter.iers.org/products/eop/rapid/standard/finals2000A.all) ，因为 AstraPy 中的模型是 IAU 2000A。

## 地球重力场模型 EGM

抽象类。

### 静态方法

#### 得到默认的模型

```python
@staticmethod
def get() -> EGM: ...
```

得到当前的地球重力场模型。

#### 设置参数为默认

```python
@staticmethod
def set(egm: EGM) -> None: ...
```

设置 `egm` 为默认的地球重力场模型。

### 方法

#### 得到地球重力场参数

```python
def get_C(self, n: typing.SupportsInt, m: typing.SupportsInt) -> float: ...
def get_S(self, n: typing.SupportsInt, m: typing.SupportsInt) -> float: ...
def get_CS(self, n: typing.SupportsInt, m: typing.SupportsInt) -> tuple[float, float]: ...
```

参数 `n` 为阶（degree），参数 `m` 为次（order）。

`get_C` 返回归一化的重力场球谐系数 $\bar{C}_{nm}$，`get_S` 返回归一化的重力场球谐系数 $\bar{S}_{nm}$。

`get_CS` 返回 $\bar{C}_{nm}$ 和 $\bar{S}_{nm}$ 构成的元组。

## 默认的地球重力场模型 EGM2008First180

继承自 `EGM`。

提供 EGM2008 地球重力场模型的前 180 阶。

!!! tip
    可以从 [https://icgem.gfz.de/getmodel/zip/c50128797a9cb62e936337c890e4425f03f0461d7329b09a8cc8561504465340/EGM2008.zip](https://icgem.gfz.de/getmodel/zip/c50128797a9cb62e936337c890e4425f03f0461d7329b09a8cc8561504465340/EGM2008.zip) 下载 EGM 2008 模型参数。