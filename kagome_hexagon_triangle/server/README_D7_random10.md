# D=7：十个独立随机初态的 simple update

这是原来的长期自适应细化版本。随机初态第一轮筛选请改用
`run_D7_random10_screen.sh`，固定9000步，见`README_D7_random10_screen.md`。

只编辑 `run_D7_random10.py` 顶部的选项，运行：

```sh
sh run_D7_random10.sh
```

这是新的独立任务，不读取之前的D4、D5态或Julia U1态。
默认seed=7101–7110，每个seed直接在D=7随机初始化；各seed不继承前一个结果。
使用当前六边形＋三自旋块、27-site实数dense iPEPS。
默认使用两个独立张量的C3旋转副本关系，不施加镜面，不固定磁化，不输入论文图案。
如需完全放开六个张量，可在提交前将`C3_ROTATED_COPIES=False`，计算量也会增加。

## 上传

服务器已有前一轮D5→D10的完整包时，只需新增三个运行文件：

- `server/run_D7_random10.py`
- `server/workflow_random_seeds.py`
- `server/run_D7_random10.sh`

保留服务器已有的`server/workflow_from_state.py`、`workflow.py`、`observables.py`、
`device_support.py`及包根目录的算法文件。新流程复用它们，不改动原算法。
无需上传任何初态文件。此说明和`test_random_seeds.py`运行时不需要上传。

## 默认设置

- D=7，10个种子，**h=0.4**。这是论文公开D7局域磁化数据的磁场，与之前h=0.38不同。
  若需要原磁场，只改顶部`FIELD=0.38`，结果目录会自动改名。
- χ=40；CTMRG至少20轮、最多50轮，阈值1e-6。
- SU使用CPU、10线程；CTMRG、双层张量和全部观测量使用`cuda:1`。
- 十个种子按顺序计算，只有一个后台Python进程，不同时占用十份GPU环境。
- 每个种子最终只测一次观测量，SU过程中不做CTMRG或反复算能量。
- 每200个完整步和监测点保存原子检查点。

| SU warm-up dt | 最小τ | 最大τ |
|---|---:|---:|
| 0.03 | 80 | 320 |
| 0.01 | 40 | 160 |
| 0.003 | 10 | 40 |
| 0.001 | 4 | 16 |

随后dt=0.0005，单段最小τ=2、最大τ=32；如果还未稳定，继续同dt的下一段。
完整Trotter步的Schmidt权重及SU乘积环境密度矩阵变化率均需小于每单位τ的1e-7，
连续3个监测点通过。每次提交每个seed最多16个细化段；通过则提前停止。
这是长期服务器任务，不保证每个随机初态都能在此预算内收敛。

若seed未通过SU代理或CTMRG判据，仍测量并保存其候选态，明确记录
`needs_more_SU`或`needs_CTM_check`，然后继续其他seed；不会把未通过者当成收敛结果。
某个seed出现普通数值异常时，保存`error.json`并继续其他seed。
KeyboardInterrupt和进程终止不会被自动吞掉或触发重启。

这些SU判据不是严格的物理收敛证明；χ只有一个值，也不能证明χ收敛。
搜索出与论文相似的图案后，仍需独立复核。

## 后台运行和日志

在服务器`kagome_hexagon_triangle/server`中执行上述sh。
自动激活`/home/sniu/miniconda3`的`pytorch`环境，启动前检查依赖、GPU、
设置和输出锁。支持与之前相同的`CONDA_SH`、`CONDA_ENV`、`PYTHON_BIN`、
`KAGOME_SKIP_CONDA`环境覆盖方式，不安装或修改环境。

默认`nohup ... &`，终端打印实际Python PID和日志路径。
stdout/stderr一起写入`server/logs/D7_random10_时间戳_PID.log`，
同名`.pid`记录实际任务PID。无需在外面再加`&`；退出SSH后继续计算。
停止使用打印出的`kill -TERM PID`；前台调试可加`--foreground`。
没有watchdog、自动重启或额外计算worker。

## 结果、下载和续算

默认结果目录：
`kagome_hexagon_triangle/results/server_D7_random10_h040_C3/`。

根目录汇总自动带`seed`列，便于下载后直接逐点比较：

- `summary.csv`、`summary.txt`：每个seed最新候选的能量、磁化、收敛与状态。
- `site_magnetization.csv`：每seed的27个site，含原Julia习惯的坐标和site_id。
- `bond_energies.csv`：每seed的54条最近邻bond及关联分量。
- `triangle_observables.csv`：每seed的18个基本三角形及三体观测量。
- `schmidt_spectra.csv`：每seed的虚拟键谱及SU熵。
- `all_measurements.csv`：包括手动续算前后所有已完成测量，不只保留最低能者。
- `batch_status.json`：当前seed及整批完成状态。
- `settings.json`、`provenance.json`：设置和来源指纹。

各`seed7101/`…`seed7110/`分别保存独立随机初态、初始化指纹，
`D7/`中保存initial/checkpoint/final.pt、SU历史和阶段报告。
每次最终测量另存于`D7/measurements/attempt0001/`等子目录，保留该次
**精确态快照**`state.pt`、`candidate.json`、完整观测量及CTMRG日志。
继续演化不会覆盖之前已测量的候选态。

重新执行同一sh时：

1. 已通过SU和CTMRG的seed直接跳过。
2. 未收敛SU的seed从原检查点继续细化，不重新随机化，不重复warm-up。
3. 测量中断时读取pending记录，对同一个态快照重做未完成测量，不先重新演化。
4. 已通过SU但CTMRG未通过的seed只重试测量，仍受50轮限制。

CTMRG若持续不通过，应检查日志和环境收敛，不假定重启一定能解决。
改变χ、演化日程、磁场或seed列表时请使用新OUT；只改测量设备可续算。
一次只允许一个进程写同一结果目录，停止后内核锁会释放，不要删锁绕过存活进程。
全部seed通过时退出码为0；有失败或未收敛候选时为2，结果仍已保存。

下载分析至少保留根目录汇总CSV、summary.txt、settings/provenance和batch_status。
**建议整个结果目录一起下载**，这样找到好图案后还可直接继续优化对应态。
服务器不读论文数据、不自动挑选“像论文”的seed，不会据图案强行修改优化结果。

本机验证仅使用D=2、每个测试态两步SU和很小的CTMRG环境；没有运行D7优化。
