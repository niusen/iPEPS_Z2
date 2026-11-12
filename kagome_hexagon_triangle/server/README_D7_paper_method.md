# 更贴近论文描述的 D7 随机初态任务

在服务器的 `kagome_hexagon_triangle/server/` 目录运行一次：

```sh
sh run_D7_paper_method.sh
```

默认自动激活 `/home/sniu/miniconda3` 的 `pytorch` 环境，后台运行一个 Python
进程，并打印日志及 PID 路径。停止用打印出的 `kill -TERM PID`，没有自动重启
循环或额外守护进程。所有物理选项放在 `run_D7_paper_method.py` 顶部。

## 修改了什么

原来的 `simple_update.py`、`honeycomb_ctm.py`、`observables.py` 和旧任务均不修改。
新增算法在 `kagome_hexagon_triangle/paper_method/`：

- `physical_nn_su.py`：逐物理最近邻键的两自旋门。每个跨块物理键作用后都做
  一次 QR/SVD，保留 D；六边形块内的键和各块 onsite field 精确作用于物理腿。
  每一步正确覆盖 54 条 NN 键和 27 个 site 的磁场。
- `c3_reduction.py`：在相等 λ 子空间内对齐正交规范，再固定虚拟腿符号。
  只有旋转相关张量、λ 和局部密度矩阵代理的差异均在阈值内才约化。
  不满足时保持六个独立张量，不强制复制。
- `workflow.py`：独立六张量随机初态、固定步数、条件 C3 转换、断点续算及最终
  同态 χ 对照；复用已有完整观测量和设备传输函数。

论文明确描述先六张量优化、后利用近似 C3，采用逐 NN 门与 SVD 的 SU，且对 χ
做收敛检查。本实现更贴近这些步骤，但作者没有公开具体 blocked-gate 排列、
时间表、规范匹配及约化算法，因此不能宣称与作者程序逐行相同。

原生六角 CTMRG 保持三个120度角、六条边的框架。其内核没有在这次重写；
新版协议增加的是同一个最终态在不同 χ 下的独立测量与逐项比较。
CTMRG 没有被用于 SU 门的环境截断。

论文及补充材料：https://arxiv.org/html/2610.34439v1

## 默认计算

- D=7，h=0.4，seed=7101…7110，十个独立的六张量随机态。
- SU 在 CPU，10 threads；CTMRG 和完整观测量在 **cuda:1**。
- 五阶段 `(dt,tau)=(.03,40),(.01,20),(.003,5),(.001,2),(.0005,1)`。
  使用 `round(tau/dt)`，共 **9000** 个完整二阶 Trotter 步，实际总 tau=67.991。
  不自动延长到几十万步；完成预算与数值收敛分别标记。
- 六张量阶段每个完整步含72次跨块键 SVD，C3约化后为24次；单个门/SVD比原来
  五自旋门小。因此“步数相同”不意味着 wall-clock 相同。
- 前两个阶段六张量自由优化。此后在第2、3、4阶段结束时尝试 C3 约化，成功
  后不再尝试；失败则继续六张量。不会在最后一次演化之后强制改变态。
- `C3_POLICY='never'` 可以作为全程六张量的对照，改此选项时使用新 OUT。
- 转换采用 tensor Frobenius 残差≤1e-3、λ轨道差≤1e-3、SU局部密度矩阵代理
  漂移≤1e-4 的条件。代理不等于无限晶格物理观测量；约化是有记录的近似投影，
  并非无限波函数保真度保证。每次转换前的态保存在 `before_C3_stage*.pt`。
- 最终态分别在 **χ=40、60** 测完整观测量；每次均从新环境开始，最多50轮，
  角谱容差1e-6。比较总能、交换能、每个 site 的 Sz、每条 NN bond 能量。
  默认跨 χ 阈值：能量1e-6/site，局部磁化和 bond 各1e-5。
- 单点 SU 判据仍为 λ 和局部密度矩阵代理每单位 tau 漂移的监测，并不是变分
  最优性证明。若 SU、CTM 或 χ 对照失败，数据仍保存，同时明确标记。

没有固定总磁化、局部 Sz、镜面对称或论文目标图案。实数 float64 是本实现的
选择；论文未交代实数/复数的具体约束。

## 数据与断点

新输出目录：`results/server_D7_random10_h040_physicalNN_six_then_C3/`。
不读取或覆盖原来的 D7 筛选结果。

每个 seed 保存：

- `initial.pt`：原始六张量随机态，SHA 与设置固定。
- `checkpoint.pt`：同一个原子文件保存态、阶段、步数和 C3 转换决定。
- `final.pt`：完成固定 SU 预算的态；并不自动等同于收敛态。
- `su_history.csv`、`su_stages.json`、`c3_reduction.json`。
- `measurements/chi40/`、`measurements/chi60/`：逐 site、逐 bond、三角形、Schmidt
  谱及完整 JSON；每一份测量都绑定同一个 `final.pt` SHA。
- `chi_convergence.json`：逐项 χ 对照和通过标记。
- `completed.json`：`protocol_complete` 表示协议运行完成；只有同时通过 SU
  代理、CTMRG 和跨χ检查才标记 `numerically_checked=true`。

根目录提供 `summary.csv/txt`、`all_chi_measurements.csv`、各类完整观测量 CSV。
根目录观测量 CSV 同时包含两个 χ，有明确的 `chi` 列；`summary.csv` 数值取最高χ。
不要将不同 χ 的行混合求和。

中断后仍运行同一个 sh：继续未完成 SU 步数，或只重试未完成的测量；已完成的
χ 不重复计算。完成9000步的任务重启不会自动追加 SU，避免意外变成长任务。
更改算法/物理设置会拒绝混用旧断点。单个 seed 失败保留错误与断点，并继续其它 seed。

## 上传哪些新文件

服务器已有原来的完整 package 时，只需增加：

1. `kagome_hexagon_triangle/paper_method/` 中的四个 `.py` 文件。
2. `kagome_hexagon_triangle/server/run_D7_paper_method.py`。
3. `kagome_hexagon_triangle/server/run_D7_paper_method.sh`。

本文档、分类脚本和测试不是运行任务的依赖。保留服务器原有的根目录和 server
公共模块；仅上传两个入口文件是不够的。运行时不需要 SciPy、Matplotlib 或论文数据。

## 本机验证

只在 D1/D2 上测试，没有在本机运行 D7 扫描：

- 所有36个跨块物理键：更新后完整张量系数与直接作用两自旋门、完整矩阵 SVD
  的结果比较；块内键和 onsite field 与完整算符比较。
- C3轨道更新与六张量显式更新比较；54键计数及全极化解析态。
- 带随机等λ子空间规范的精确C3态可正确约化，且CTMRG的能量、site磁化和每条
  bond观测量保持一致；远离C3的态拒绝转换。
- SU中断续算与不中断结果一致；第二χ测量中断只重试测量；已完成任务跳过；
  改变初态或设置时检查完整性。
- POSIX sh 语法、后台返回、conda选择、日志和PID用模拟程序检查，不启动D7。

测试代码：`server/test_paper_method.py`。
