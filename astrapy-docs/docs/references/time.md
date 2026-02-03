# 时间模块

Astra 的时间模块在 Python 中的绑定为 `AstraPy.time`。

## 纪元 Epoch

AstraPy 的时间模块主要由 `Astra::Time::Epoch` 类实现（Python 绑定为 `Astra.time.Epoch`），该类有两个私有属性，`tai1_` 和 `tai2_`。

```cpp
class Epoch {
private:
    double tai1_;
    double tai2_;
}
```

正如属性的名称所示，`Epoch` 基于国际原子时。`tai1_ + tai2_` 是 TAI 下的儒略日，儒略日的值可以以任意方式拆分到 `tai1_` 和 `tai2_` 中。但在 `Epoch` 初始化时，`tai1_` 一般为 `2400000.5`，即修正儒略日的零点。

??? note "国际原子时（Temps Atomique International, TAI）"
    TAI 是基于原子共振频率的高精度时间标准。它提供了一个连续、稳定的时间尺度，不受地球自转不规律性的影响。

    TAI 由全球原子钟网络确定，这些原子钟基于铯-133 原子发出的电磁辐射测量时间。TAI 中的秒被定义为这种辐射 9,192,631,770 个周期的持续时间。

    与 UTC 不同，TAI 不包括闰秒，因此连续运行而无需调整。截至 2023 年，由于自 1972 年以来添加到 UTC 的闰秒，TAI 比 UTC 快 37 秒。

    原子时对于需要极其精确计时的应用至关重要，如 GPS、科学研究和电信。

??? note "儒略日（Julian Day, JD）"
    儒略日（JD）是自公元前 4713 年 1 月 1 日世界时中午以来的连续日计数和日的小数部分。它由天文学家引入，为处理不同历法提供单一日期系统。

    儒略日系统由约瑟夫·斯卡利格在1583年提出。选择起始点是因为它早于有记录的历史，代表古代年代学中使用的三个主要周期重合的点。

    儒略日表示为十进制数，其中整数部分代表日数，小数部分代表一天中的时间。

    一个称为修正儒略日（Modified Julian Day, MJD）的变体定义为 $MJD = JD - 2400000.5$，这减少了位数并将整数边界置于午夜。

    儒略日主要用于天文学和空间科学，以简化时间间隔计算。

### 静态方法

#### 从协调世界时（UTC）构造

``` python
@staticmethod
def from_utc(year: typing.SupportsInt, month: typing.SupportsInt, day: typing.SupportsInt, hour: typing.SupportsInt, minute: typing.SupportsInt, second: typing.SupportsFloat) -> Epoch: ...
```

输入参数依次为年、月、日、小时、分钟和秒。

#### 从国际原子时（TAI）构造

```python
@staticmethod
def from_tai_mjd(tai_mjd: typing.SupportsFloat) -> Epoch: ...
```

输入参数为在 TAI 下的修正儒略日。

#### 从地球时（TT）构造

```python
@staticmethod
def from_tt_mjd(tt_mjd: typing.SupportsFloat) -> Epoch: ...
```

输入参数为在 TT 下的修正儒略日。

??? note "地球时（Terrestrial Time, TT）"
    地球时是国际天文学联合会（IAU）定义的一种现代天文时间标准，主要用于从地球表面进行的天文观测的时间测量。

    在这一角色上，TT 延续了地球动力学时（Terrestrial Dynamical Time，TDT 或 TD），而 TDT 又是历书时（Ephemeris Time，ET）的继承者。TT 与 ET 最初的设计目标相同，即摆脱地球自转不规则性对时间测量的影响。

    TT 的时间单位是 SI 秒，其当前定义基于铯原子钟。但 TT 本身并不是由原子钟直接定义的，而是一种理论上的理想时间尺度，现实中的时钟只能对其进行近似实现。

    TT 不同于通常用于民用目的的时间尺度——协调世界时（UTC）。TT 通过国际原子时（TAI）间接成为 UTC 的基础。由于在引入 TT 时，TAI 与 ET 之间存在历史上的差异，TT 比 TAI 超前 32.184 秒。

#### 从当前时刻构造

```python
@staticmethod
def now() -> Epoch: ...
```

该方法使用 `std::chrono::system_clock::now` 得到当前时刻，并通过 `std::gmtime` 获得对应的 UTC 时间，然后调用 `FromUTC` 方法构造 `Epoch`。

#### 从 J2000 构造

```python
@staticmethod
def J2000() -> Epoch: ...
```

该方法返回 J2000 时刻的 `Epoch`。

### 方法

#### 得到修正儒略日

```python
def to_tai_mjd(self) -> float: ...
def to_utc_mjd(self) -> float: ...
def to_tt_mjd(self) -> float: ...
@overload
def to_ut1_mjd(self) -> float: ...
@overload
def to_ut1_mjd(self, dut1: typing.SupportsFloat) -> float: ...
```

`to_tai_mjd` 从 `Epoch` 得到 TAI 下的修正儒略日，`to_utc_mjd` 得到 UTC 下的修正儒略日，`to_tt_mjd` 得到 TT 下的修正儒略日。

`to_ut1_mjd` 得到 UT1 下的修正儒略日，其中无参的版本使用默认的地球定向参数 `Astra::Data::EarthOrientationParameters` 获得对应 UTC 时刻的 UT1 与 UTC 的差值，有参的版本直接使用输入参数作为差值。

??? note "世界时（Universal Time, UT）"
    世界时是一种以格林尼治子夜起算的平太阳时。世界时是以地球自转为基准得到的时间尺度，其精度受到地球自转不均匀变化和极移的影响，为了解决这种影响，1955 年国际天文联合会定义了 UT0、UT1 和 UT2 三个系统：

    - UT0 系统是由一个天文台的天文观测直接测定的世界时，没有考虑极移造成的天文台地理坐标变化。该系统曾长期被认为是稳定均匀的时间计量系统，得到过广泛应用。

    - UT1 系统是在 UT0 的基础上加入了极移改正 $Δλ$，修正地轴摆动的影响。UT1 是目前使用的世界时标准。被作为目前世界民用时间标准 UTC 在增减闰秒时的参照标准。

    - UT2 系统是 UT1 的平滑处理版本，在 UT1 基础上加入了地球自转速率的季节性改正 $ΔT$。

#### 加上一段时间

```python
def add_seconds(self, seconds: typing.SupportsFloat) -> Epoch: ...
```

得到当前的 `Epoch` 时刻度过 `seconds` 秒后的时刻。

#### 得到经过的秒数

```python
def seconds_since(self, other: Epoch) -> float: ...
```

得到当前时刻自 `other` 时刻经过的秒数。

#### 得到格林尼治平恒星时

```python
@overload
def to_gmst06(self) -> float: ...
@overload
def to_gmst06(self, dut1: typing.SupportsFloat) -> float: ...
```

该方法计算并返回根据 IAU 2006 标准的格林尼治平恒星时（GMST）。其中无参的版本使用默认的地球定向参数 `Astra::Data::EarthOrientationParameters` 获得对应 UTC 时刻的 UT1 与 UTC 的差值，有参的版本直接使用输入参数作为差值。

#### 得到协调世界时

```python
def to_utc(self, precision: typing.SupportsInt = ...) -> datetime.datetime: ...
def to_utc_string(self, precision: typing.SupportsInt = ...) -> str: ...
```

参数 `precision` 表示秒字段中的小数位数，可以取正值也可以取负值，但安全的范围为 `-5` 到 `+9`，默认值为 `3`。

### 重载的运算符

`Astra.time.Epoch` 重载了等于（`__eq__`）、大于（`__gt__`）、大于等于（`__ge__`）、小于（`__lt__`）、小于等于（`__le__`）和不等于（`__ne__`）共六个运算符。

## 添加闰秒

```python
def add_leap_second(year: typing.SupportsInt, month: typing.SupportsInt, delat: typing.SupportsFloat) -> None: ...
```

使用 `Astra.time.add_leap_second` 可以动态添加闰秒。

该函数的 `year` 为生效年份（UTC），`month` 为生效月份，`delat` 表示从该月起适用的 $TAI − UTC$ 基础值（秒）。