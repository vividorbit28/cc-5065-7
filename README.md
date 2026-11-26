# 偏振光表面渲染原型 (Stokes / Mueller polarization renderer)

用 Stokes 向量表示入射光，通过显式的偏振片 Mueller 矩阵与介质界面
Fresnel Mueller 矩阵沿光路传播，最终在相机前用可旋转的虚拟检偏片积分
感光强度。所有像素在进入颜色/灰度转换前都必须通过物理可行锥检查。

## 目录结构

```
polrender/
  stokes.py     Stokes 向量、物理可行锥 (I>=0, Q^2+U^2+V^2<=I^2)、零光态
  mueller.py    旋转矩阵、理想偏振片、Fresnel 反射/透射 Mueller 矩阵
  frames.py     3D 路径局部基 (e1,e2,k)、反射传输、近法线退化策略
  path.py       PolarizedPath / PathEngine：完整光路状态传播与事件审计
  scene.py      倾斜/波纹平面、V 形槽双反射场景与逐像素渲染器
  gui.py        matplotlib 控件实验窗口（由内核驱动，失败回显诊断区）
tests/
  acceptance.py 134 项解析与数值边界验收测试
  smoke_gui.py  无头 GUI 冒烟测试（控件、错误反馈路径）
scripts/
  render_headless.py  无显示环境下生成全部图像、诊断与 npz 数据
outputs/             渲染产物与 report.txt
```

## 如何启动界面

```bash
# 1) 创建虚拟环境并安装依赖（本机无系统 pip/tkinter，使用 venv）
python3 -m venv --without-pip .venv
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python /tmp/get-pip.py
.venv/bin/python -m pip install -r requirements.txt

# 2) 启动实验窗口（需要任一交互式 matplotlib 后端：TkAgg / Qt5Agg 等）
.venv/bin/python -m polrender.gui
```

当前容器没有安装 Tk/Qt 且无 root 权限，交互窗口无法在此环境弹窗；
窗口逻辑由 `tests/smoke_gui.py` 在 Agg 后端下完整验证。无显示时可用
等价的无头流程复现窗口里的全部结果：

```bash
.venv/bin/python scripts/render_headless.py   # 写 outputs/
.venv/bin/python tests/acceptance.py          # 134 项验收
.venv/bin/python tests/smoke_gui.py           # GUI 控件冒烟
```

## 实验窗口控件（全部由偏振内核驱动）

- source psi [deg]：入射光偏振轴方向（线偏振方位角）
- source DoP：偏振度，合法范围 [0,1]，越界会被内核拒绝并在红色诊断栏回显
- ellipticity chi [deg]：椭圆角，0=线偏振，±45=圆偏振
- index n2：介质折射率（n1=1 空气）
- analyzer axis [deg]：相机前虚拟检偏片透光轴，可 0–180° 旋转
- analyzer extinction：非理想偏振片消光泄漏（0 为理想偏振片）
- 场景：波纹平面（单次反射）/ V 形槽（两次反射）
- 退化策略：transport（平行传输基）/ reset（世界轴重设）

结果区：四个检偏角度的反射灰度图（统一色标）、总强度 I 图、DoP 图、
Q/U 符号图、中心像素检偏角扫描曲线（含 Brewster 角标记）以及中心像素
完整 Stokes 状态与可行性结论。

## 约定（必须完整，不能只乘一个角度装饰因子）

1. Stokes 列向量 `S=(I,Q,U,V)^T`，矩阵作用为 `S' = M S`；
   物理顺序为 M1 后 M2 时 `S_out = M2 M1 S_in`，事件链按此记录。
2. 每条光线携带右手横基 `(e1,e2,k)`，满足 `e1 x e2 = k`。
   界面处 `e1=s=n×k`、`e2=k×s`（p 方向），Fresnel 矩阵在该基下对角。
3. `R(theta)` 采用标准被动 Mueller 坐标旋转（Goldstein/Collett 约定）：
   参考轴自身转 +theta 时，同一物理态的分量按 R(theta) 变换；
   `R(-theta)=R(theta)^T`，`R(a)R(b)=R(a+b)`。
4. 反射 rp 使用反射后坐标系的翻号约定，使近法线各向同性；
   透射振幅 ts/tp 直接由标准定义给出，并乘导纳因子
   `eta = n2 cos_t / (n1 cos_i)`，保证无耗介质 R+T=1。
5. 近法线 `n×k -> 0` 时 s 轴无定义：
   - transport：沿路径平行传输旧 e1；
   - reset：重锚到世界 x 轴。
   两种策略只相差一次坐标旋转，可观测强度逐位相同（验收与 report.txt）。
6. 完全无光 `S=0` 是合法态，DoP 定义为 0（不做除 I）；
   `I<=0` 但 QUV 非零是错误，直接抛异常而不是静默钳制。
7. 任何矩阵链输出先经 `sanitize`（仅修复 1e-12 级数值越界）再做
   `check_feasible`；不可行状态不会进入图像/颜色转换。

## 验收依据（解析情形 + 数值边界）

`tests/acceptance.py` 覆盖：

- Malus 定律全角度、平行/交叉检偏、非偏振光过偏振片 I/2、双正交偏振片零矩阵；
- Fresnel R/T 在 [0,1] 内、全角度 R_s+T_s=R_p+T_p=1、
  法线反射率公式、Brewster 角 Rp=0 且反射光全偏振、掠射 R→1；
- 任意坐标轴旋转下可观测强度不变（含算子恒等式）；
- 只旋转路径基（亮度不变）与只旋转检偏片（可复测消光差异）两个对照实验；
- 近法线退化两种策略观测等价、精确法线反射各向同性；
- V 形槽双反射可行性与能量范围、零光输入精确零输出；
- 渲染器拒绝不可行光源、平行+交叉检偏强度和等于反射总 I；
- 矩阵乘法次序与事件链顺序审计。

## 当前模型忽略的效应（诊断边界）

- 单一波长、标量辐照度：无色散、无光谱渲染与颜色混合；
- 无吸收/复折射率界面的完整椭圆偏振相位链（矩阵保留复数接口但场景只用实数 n）；
- 无粗糙面 BSDF、体散射、荧光、退偏器与偏振相关多次散射统计；
- 正交相机、解析几何求交，无通用网格光线追踪与遮挡；
- 忽略镜头双折射、传感器微偏振片阵列、量子噪声与有限数值孔径；
- 无相干干涉/衍射，Stokes 描述的是时间平均的非相干强度量。
