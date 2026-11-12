# 服务器 D=4 → 10 长时间链式计算

入口是 **`run_D4_D10.py`**，所有可编辑选项都在这个文件顶部。
这是刚测试的六边形＋三自旋块、27-site `3×3` iPEPS ansatz，
不是旧的 U(1) `sqrt3×sqrt3` iPESS。C3通过旋转副本施加，实数dense张量；总磁化不固定。

## 上传与启动

上传整个 **`kagome_hexagon_triangle` 目录**，保持这个名字及内部结构。
无需上传其 `results/`、任意 `__pycache__/`、测试日志或本机以前算好的态。
不要只上传 Python 入口和 sh：运行需要上层已有的 `__init__.py`、`geometry.py`、
`simple_update.py`、`honeycomb_ctm.py`。`observables.py`供小测试使用，也一并保留。
本服务器入口不导入`analyze.py`，不读取`references/`，两者均可不上传。
不需要其他 `iPEPS_Z2` 模块，也不需要 vendor、Julia 或任何 zip。

sh沿用旧`kagome_su_server`的逻辑，自动加载
`/home/sniu/miniconda3/etc/profile.d/conda.sh`并激活`pytorch`环境。
默认无需手动从`base`切换。所需包：`torch`、`numpy`、`opt_einsum`；
默认只保存数值；把`MAKE_PLOTS`设为True才额外生成本次结果的图表，需要`matplotlib`。
sh文件不自动安装或修改环境。

可通过`CONDA_SH`、`CONDA_ENV`、`PYTHON_BIN`覆盖初始化文件、环境名和解释器。
已手动激活环境时可设置`KAGOME_SKIP_CONDA=1`跳过激活。
启动前会打印实际Python路径，并检查PyTorch、NumPy、opt_einsum能否导入。

在 `kagome_hexagon_triangle` 的上一级执行一次：

```sh
sh kagome_hexagon_triangle/server/run_D4_D10.sh
```

这条命令会自动使用`nohup ... &`后台启动，打印PID和日志路径，然后返回终端。
可以退出SSH，计算继续。stdout和stderr合并写入
`server/logs/D4_D10_时间戳_PID.log`，同名`.pid`文件保存任务PID。
**不要再在外面加一层nohup或&。**

查看进度和停止（用启动时打印的真实路径/PID）：

```sh
tail -f /实际路径/server/logs/D4_D10_时间戳_PID.log
kill -TERM 实际PID
```

需要前台调试时，使用`sh run_D4_D10.sh --foreground`。
不使用Slurm。sh是POSIX语法，可以直接用`sh`，没有bash数组。
默认只有一个长时间后台Python进程，没有自动重启、watchdog或额外计算worker。
nohup让它忽略退出终端产生的SIGHUP；没有屏蔽SIGTERM。
启动前会检查依赖、GPU编号、已有设置和输出锁；运行中的同目录任务会阻止重复提交。

## 默认计算设置

- h=0.38；默认CPU；10线程；D=4,5,6,7,8,9,10。
- GPU选项也在`run_D4_D10.py`顶部：把`MEASUREMENT_DEVICE='cpu'`改为`'cuda:1'`。
  这会把CTMRG、双层张量和全部观测量收缩放到GPU上；SU始终在CPU上。
  环境张量、态张量和插入算符会在测量时使用同一设备；保持float64精度。
  没有可用CUDA或编号不存在时明确报错，不会悄悄退回CPU。
  CUDA编号是PyTorch可见设备的编号，脚本不修改`CUDA_VISIBLE_DEVICES`。
- 检查点始终保存CPU态，可以在CPU与GPU测量之间切换。只改变测量设备允许继续原任务，
  不需要新建结果目录；已完成的D仍然跳过，未完成的测量使用新设备。
  结果中记录实际设备和GPU峰值已分配显存。
- D4从随机seed1701开始。D5从D4的最终态扩充，D6从D5，依次到D10。
- 扩充严格补零，不添加随机扰动；旧的张量和Schmidt权重不变。
  后续门和保留D个奇异值的SVD可以激活新维度，小D测试已验证。
- 可将`INITIAL_D4_CHECKPOINT`设为**这个分块ansatz**的D4 `.pt` 文件，
  以替代D4的随机初始化。旧Julia/U1/iPESS态不能直接作为这个选项。
- 二阶完整Trotter步。每个阶段至少运行下表的最小tau；若仍漂移则继续到最大tau：

| dt | 最小tau | 最大tau |
|---|---:|---:|
| 0.03 | 80 | 320 |
| 0.01 | 40 | 160 |
| 0.003 | 10 | 40 |
| 0.001 | 4 | 16 |
| 0.0005 | 2 | 8 |

最小tau约为之前每D时间表的两倍，最大约八倍。阶段结束需要完整步监测连续3次稳定；
因此实际步数还受监测间隔取整影响，每D约2万到7.2万个完整步。
监测比较**完整Trotter步结束时**的Schmidt谱和SU乘积环境的物理密度矩阵变化率，
不是旧日志的子更新`lambda_delta`。默认阈值均为每单位tau的1e-7。

这些仅是便宜的SU收敛代理，不是CTMRG能量，也不证明全局最优或精确物理收敛。
达到最大tau仍不稳定时会保留 `proxy_converged=false`，继续后续阶段和D，
不会悄悄把它标为已收敛。

## 测量：每D结束后一次

SU阶段不做CTMRG。每D演化完成后：χ=40，CTMRG至少20次、最多50次、角谱容差1e-6，
然后在同一环境上计算完整观测量。没有按stage反复测量能量。

1. 27个site：`Sz,Sx,Sy`、归一化磁化、局域自旋长度、单site纯度。
2. 54条最近邻bond：`Sxx,Syy,Szz,S·S`及其连通关联。
3. 18个基本三角形（9个Tup、9个Tdn）：总/平均磁化、交换能量、
   `SzSzSz`、连通三体关联、总Sz的方差和三个方向的二体关联之和。
4. 总能量、交换能量、Zeeman能量、平均磁化、偏离1/9的量、最大横向磁化。
5. 九条粗虚拟键的Schmidt谱、SU键熵、CTMRG日志、不同面位置的观测量差异。
6. 可选：本次计算自身的逐D能量与磁化曲线，以及每个D的27-site磁化/bond图。

保留 `x,y,sublattice,site_id`，沿用原Julia库的物理site坐标；x,y从1开始。
Tup/Tdn以原张量连接命名：Tup(R)连接B1(R),B2(R),B3(R)，
Tdn(R)连接B1(R+ex),B2(R),B3(R+ey)。每个三角形的x,y由其B2顶点确定。
所有54条物理键恰好分别属于18个三角形之一，另做能量求和一致性检查。

Sy与标量手性在**实数ansatz**下恒为0，记录中明确说明这一约束；
YY二体关联并不为0，通过实数Sp/Sm组合计算，避免错误丢弃虚部。
键熵是SU的Schmidt权重熵，不声称是无限二维态的精确纠缠熵。

只限制了重复使用的观测量角矩阵缓存（默认256MB），没有声称总内存只有256MB。
χ=40、D较大时双层收缩与环境仍需要较多内存，适合服务器运行。

## 结果和续算

默认输出：`kagome_hexagon_triangle/results/server_D4_D10_h038_C3_chain/`。

- `summary.txt`：逐D进度、能量、磁化及SU/CTM收敛标记。
- `summary.csv`：本次计算的数值汇总。
- `site_magnetization.csv`、`bond_energies.csv`、`triangle_observables.csv`：所有D合并表。
- `schmidt_spectra.csv`：虚拟键谱。
- 可选绘图开启后：`report.html`、`energy_magnetization.png`和每D的 `pattern.png`。
- `D4/`…`D10/`：各自的`initial.pt`、`checkpoint.pt`、`final.pt`、
  全部`observables.json`、逐项CSV、`su_history.csv`和`su_stages.json`。
- `settings.json`、`provenance.json`：设置、源码与状态指纹。父态SHA256记录在子D的数据中。

默认每200个完整步及监测点保存一次。张量和进度写在**同一个原子检查点**中，
避免中断后两者错位。中断后用同一个sh重新启动即可：
已完成的D跳过；正在演化的D从检查点接着算；SU已完成而测量中断的D只重做尚未完成的测量。

一次只启动一个进程写同一输出目录。内核锁会随进程退出释放，
遗留的`.writer.lock`文件本身不会阻止重启；不要通过删锁文件绕过仍存活的进程。
改变物理参数或时间表时使用新的`OUT`，程序会拒绝混合不同设置的结果。

## 能量约定与方法范围

`H=ΣSi·Sj−hΣSz`，`m/ms=2 mean Sz`，
`Etot/site=Eex/site−h*(m/ms)/2`。
服务器只计算本次链，不读取论文作图数据、以前的态或以前的测量表，
不做能量对照或图案匹配。完整原始数据下载后可再在本机分析。

沿用已检查的六自旋/三自旋分块、两张量旋转副本表示、simple update及原生蜂窝CTMRG。
论文未公开原始代码和完整的门拆分、时间表等细节；本实现的五自旋门分配、二阶Trotter、
初始就施加C3、以及本文件列出的收敛监测，都是明确记录的复现选择，
不声称这些未公开参数与作者的程序逐项一致。

## 本机验证范围

只做D1/D2/D3的小检查：完全极化态、随机簇乘积态的三体关联、
完整关联分量与原能量函数的对照、精确补零和新维度激活、
中断续算与连续运行一致、父态指纹、完成项跳过、锁释放与重用。
服务器D4--D10的长任务没有在本机启动。
