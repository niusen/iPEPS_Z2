# 从之前本机 D=5 图案出发，逐级计算到 D=10

入口为 `run_D5_D10_from_local_D5.py`，选项全部在文件顶部。
这是新的独立分支，不覆盖旧的 D4→D10 任务或结果。
继续使用已有六边形＋三自旋块、27-site C3 iPEPS 算法；没有固定磁化、
没有添加目标局域场或镜面对称约束，也没有重新随机初始化。

## 上传和启动

服务器已具备上一轮完整的 `kagome_hexagon_triangle` 时，新增：

- `server/run_D5_D10_from_local_D5.py`
- `server/workflow_from_state.py`
- `server/run_D5_D10_from_local_D5.sh`
- 整个 `server/initial_states/`，包括 `.pt` 初态和 `.json` 来源记录。

本说明可选上传。`test_seeded_chain.py` 是本机小测试，运行不需要上传。
必须保留此前已有的 `server/workflow.py`、`server/observables.py`、
`server/device_support.py` 及包根目录的算法文件，无需修改这些文件。
无需上传全部本机历史结果、绘图模块、references、vendor 或 pycache。

进入服务器的 `kagome_hexagon_triangle/server` 后，运行一次：

```sh
sh run_D5_D10_from_local_D5.sh
```

脚本自动激活 `/home/sniu/miniconda3` 中的 `pytorch` 环境，
检查初态 SHA256、依赖、GPU和输出锁，然后用一个 `nohup ... &` 进程后台运行。
无需在外层再加 `&`。终端会打印实际 Python PID、日志和 PID 文件路径。
日志位于 `server/logs/from_local_D5_D5_D10_时间戳_PID.log`，合并 stdout/stderr。
与旧脚本一样，可用 `CONDA_SH`、`CONDA_ENV`、`PYTHON_BIN`、`KAGOME_SKIP_CONDA`
覆盖环境选择；需要前台调试时加 `--foreground`。
没有自动重启、watchdog 或额外计算进程。停止用打印出的 `kill -TERM PID`。

## 初态是哪一个

`initial_states/D5_local_paperlike.pt` 是下面文件的逐字节副本：

`results/D5_continuation_h038/D5_seed1701/final.pt`

SHA256：
`ac0e8cbc2d9141f6f2574bf830957949d310f90fcd090a98d5f750636734f6bf`

这是之前独立随机初始化并继续演化后的 D5、h=0.38、seed1701、12835步态。
对应 χ=24 的复测 Eex/site = −0.4166921402793999，m/ms = 0.11151194936473192。
JSON来源记录中保留了该次完整观测量；程序会核验初态指纹以防选错分支。
它与论文图案较接近，但不是作者提供的波函数，也没有被认证为完全收敛。
不会读入这次服务器 D4 分支中的 D5。

## 演化和收敛控制

- h=0.38；SU在CPU，10线程；CTMRG及全部观测量默认 `cuda:1`。
- χ=40；CTMRG最多50轮，至少20轮，角谱阈值1e-6。
- D5先从保存态继续细化，随后 D6←D5、D7←D6，依次到D10。
- 每个新D严格补零继承前一个D；没有随机噪声。
- D5不重跑大步长0.03/0.01，仅先做0.003、0.001，再做0.0005。
- D6–10先沿用旧任务前四阶段，再进入0.0005的收敛控制。

| warm-up dt | D5 最小/最大τ | D6–10 最小/最大τ |
|---|---|---|
| 0.03 | 跳过 | 80 / 320 |
| 0.01 | 跳过 | 40 / 160 |
| 0.003 | 10 / 40 | 10 / 40 |
| 0.001 | 4 / 16 | 4 / 16 |

最后的 dt=0.0005 每段最小τ=2、最大τ=32。比较完整Trotter步结束时的
Schmidt权重及SU乘积环境物理密度矩阵，变化率都需小于每单位τ的1e-7，
连续3个监测点通过。若段内不通过，就继续下一段0.0005，不重复warm-up。
每次提交、每个D最多16个细化段；若仍不通过，保存后停止，**不进入下一个D**。
重新执行相同sh会接着做更多细化段，已经完成的warm-up不重做。
这些是SU代理收敛判据，不证明全局最低能或物理观测量严格收敛。

SU代理通过后，每个D只做一次最终CTMRG和完整观测量。
如果CTMRG在50轮内未达标，仍保存全部观测量，但停止增加D；
重启会只重试当前D的测量，不重做SU。单个χ也不能证明χ收敛。
自动检查不会强迫态保留最初图案，演化后若图案改变，会如实保存。

## 数据和中断恢复

新结果目录：
`kagome_hexagon_triangle/results/server_from_local_D5_D5_D10_h038_C3_chain/`

每个D保存初态、最终态、原子检查点、完整SU日志和阶段报告，以及：
27个site的磁化、54条bond的各分量关联、18个三角形的能量/磁化/三体关联，
粗虚拟键Schmidt谱、全局能量、磁化、CTMRG与环境位置一致性指标。
汇总CSV与summary.txt沿用旧任务格式，方便本机分析。
`chain_status.json`区分`all_completed`、`needs_more_SU`和`needs_CTM_check`。
SU/CTM未通过时进程退出码为2，表示有检查点但尚未完成整条链。

中断后执行相同sh：已完成的D跳过，未完成的D从同一检查点继续；
父态SHA256会核验，避免混合分支。单个内核锁阻止同目录重复写入。
不要删除锁文件去绕过仍存活的任务。停止后锁随进程退出释放。
只有测量设备可以在同目录更改；更改物理设置、时间表等请使用新的OUT。

不会在服务器读取论文数据或旧结果做比较。来源JSON仅记录本机初态的历史测量。
