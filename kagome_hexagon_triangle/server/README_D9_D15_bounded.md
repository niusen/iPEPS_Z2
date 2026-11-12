# 从中止的 D=9 继续到 D=15

向服务器原来的 `kagome_hexagon_triangle/server/` 上传三个文件：

- `workflow_bounded_resume.py`
- `run_D9_D15_bounded.py`
- `run_D9_D15_bounded.sh`

其余复用服务器已经运行过的算法与观测量代码，不需要重新上传初态。
保留服务器原目录 `results/server_from_local_D5_D5_D10_h038_C3_chain/`，
其中需要有 `settings.json`、`D9/checkpoint.pt`（或 `D9/final.pt`）；
已完成的 D5–8 的结果也会收集到新任务的汇总文件。

在 `server` 中提交一次：

```sh
sh run_D9_D15_bounded.sh
```

自动激活 `pytorch` conda 环境，后台运行，日志和 PID 保存在 `server/logs/`。
结束终端不影响后台任务。没有自动重启循环；停止使用输出中的 `kill -TERM PID`。
重复执行同一个 sh 会恢复新任务的断点，跳过已经测量完成的 D。
提交前会检查旧任务是否仍在写数据；若有旧进程，先停止该进程，不要删锁文件。

全部选项放在 `run_D9_D15_bounded.py` 中，无需命令行参数：

- h=0.38，保留旧任务的五自旋 SU、二阶 Trotter、C3 约束和浮点精度。
- 读取旧目录中进度最新的 D9 保存态，在新目录冻结副本；不改写旧目录。
- D9 仅补 `dt=0.0005`，tau 最多 1，即最多 **2000** 个额外完整步。
  不重复之前已经完成的大步长演化。
- D10→D15 各自从上一个 D 的最终态严格补零扩充，无随机扰动。
  dt 依次为 `[0.03,0.01,0.003,0.001,0.0005]`，
  tau 上限约 `[40,20,5,2,1]`，每个新 D 最多 **9000** 个完整步。
- SU 的两个漂移容差均为 `1e-5`；通过连续三次检查可提前结束阶段。
  达到阶段上限就进入下一阶段，不再反复追加小步长块。
- 演化过程中仅计算便宜的 SU 内部漂移指标，不做 CTMRG、能量、磁化测量。
  每个 D 的演化结束后只做一次完整测量。
- SU 使用 CPU，线程 10；CTMRG 和观测量使用 **cuda:1**，chi=40，
  最多 **50** 次，容差 `1e-6`。

新结果目录：`results/server_from_local_D5_D9_D15_h038_C3_bounded/`。
根目录的 `summary.csv`、`summary.txt`、`site_magnetization.csv`、
`bond_energies.csv`、`triangle_observables.csv`、`schmidt_spectra.csv`
随每个 D 完成更新。每个 D 还保存初态、断点、最终态、SU 与 CTMRG 历史。
`source_manifest.json` 记录旧态来源和校验值。

**达到固定预算不等于已收敛**：汇总中分别记录 `su_proxy_converged` 和
`ctm_converged`。即使未通过 SU 漂移判据，仍按本次要求测量并继续到下一个 D。
D9 的 `steps` 包含继承的旧步数；`added_steps_this_job` 是本次新增步数。
