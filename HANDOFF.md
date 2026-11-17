# 仿射区域与条件报告交接说明

本说明面向接手维护者，配合 README.md（V1 口径）与 AFFINE.md（本次契约）阅读。
全部实现仅用 Python 3.11 标准库与既有 SQLite 归档，未引入第三方依赖。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `lab/regions.py`（新增） | region 规范化、精确有理几何判据、按等式秩的 Hausdorff 测度积分，以及旧 x 行到 region 的等价转换 |
| `lab/observation_rules.py`（扩展） | 每行严格二选一 `x` 或 `region`；旧 x 规范化路径与载荷保持兼容 |
| `lab/observation_likelihood.py`（扩展） | 保留原入口签名，新增 region 分支；`enforce_truncation` 仅在发布时强制 1e-8 截断门限 |
| `lab/fit_solver.py`（扩展） | 模式键区分 x/region，region 用可行代表点构造初值；收敛后逐 region 复核截断门限 |
| `lab/conditional_reports.py`（新增） | 条件报告载荷（kind、sample、thresholds、denominator、probability、model_digest） |
| `lab/conditional_risk.py`（新增） | `conditional_risk` 服务：只认父拟合冻结任务中的样本，同模型同测度分子/分母 |
| `lab/results.py`（扩展） | 新结果类型 `conditional_report` 的请求键、发布、读取与父模型摘要复核 |
| `lab/evidence.py`（扩展） | 导出支持两类报告，额外核验条件样本确属冻结父任务且描述完全相符 |
| `lab/workflow.py`（扩展） | 挂载 `ConditionalRiskService`，新增 `conditional_risk` 方法；其余 V1 接口不变 |

## 观测测度（必须保持的不变量）

区域 `R = A ∩ 半平面交集`，等式法向量秩 `r`：

- 秩 0：二维**面积**积分。凸多边形先在固定原始坐标框 `[-50,50]^2` 内裁剪（与模型参数无关，
  保证目标函数随参数平滑），再按竖直板片用条件正态 + 既有 12/24 阶 Gauss-Legendre 积分。
- 秩 1：沿等式直线的一维**欧氏弧长**积分。直线取原始整数法向量 `(a,b)` 的本原方向
  `d=(-b/g,a/g)`，弧长参数 `p0+d·t`，因此参数化天然带欧氏长度，不依赖等式写法。
- 秩 2：单点处的二元联合密度。

关键恒等式（秩 1，`Σ⁻¹` 为精度阵，`u=p0-μ`，`h=d'Σ⁻¹u`，`v*=d'Σ⁻¹d`）：

```
f(p0+dt)·|d| = C · φ(t; m_t, sd_t)
m_t = -h/v*        sd_t = 1/sqrt(v*)
C   = |d|/sqrt(2π·v*·detΣ) · exp(-.5 u'Σ⁻¹u + .5 h²/v*)
```

因此重复、重排或对等式做合法非零缩放都不改变 L；这一点由
`tests/test_affine.py` 与 `affine_examples.py kernel_checks` 固定核账。

## 精确有理几何

`c` 先按 V1 `finite_number` 转成 binary64，再以 `Fraction(float(c))` 取得其**精确有理值**；
系数保持整数。几何判定（平行线、矛盾、窄条、正测度）全部在 Fraction 上完成，不使用浮点容差：

- 秩 0 正面积用 Fourier–Motzkin 消元的严格可行性判定；
- 秩 1 用精确 t 区间，要求 `t_lo < t_hi`（允许无界线、射线、正长度线段）；
- 秩 2 解二元一次方程组，单点零维测度为正，并逐半平面核验。

不满足时在接纳阶段抛 `ValueError`，整批回滚（见 `geometry_checks` 的 iterdump 比对）。

## 固定样本不变量

- 条件报告只从父拟合的冻结任务取样本；当前来源里同名样本不参与替代，缺失即 `KeyError`。
- 分子分母同一模型、同一观测测度；阈值严格 `>`，秩 1/2 用精确有理比较，
  整线或点落在阈值上时分子为 0（不是 `>=`，也不拒绝报告）。
- `denominator` 是正且有限的 L(R) 本身，不取 log、不取 count 次幂；count 不影响条件概率。
- 报告身份键含 fit_id、sample_key、规范阈值，因此换样本不会复用，也不会与无条件 risk 混淆。
- 读取、重放、重开、导出都不重算；源更正只产生新修订，旧证据链保持字节一致。

## 数值失败边界

- 优化迭代期间，试算点可能落在可解析支撑之外：非强制模式返回有限的平滑屏障目标，
  保证五点得分有限；这只是优化器内部行为。
- 收敛发布前对每条 region 以 `enforce_truncation=True` 复核：L(R)/L(A) 必须 ≥ 1e-8
  （秩 2 比值恒为 1），否则 `NumericalFailure`，不发布。
- 退化条件尺度、越界 binary64、未收敛、曲率与参数误差不达标均不发布，沿用 V1 事务回滚。
- 失败端口沿用 `engine.failing("fit"|"minimize"|"integrate"|"linear"|"risk")`
  与 `storage.failing(...)`；条件报告计算受 `risk` 端口控制，归档受 result.* 与 commit 控制。

## 运行与核账

```
python3.11 -B -m unittest discover -s tests -v     # 原 20 个 V1 回归 + 新增 21 个区域测试
python3.11 -B affine_examples.py reference        # 仅独立 Decimal 固定样例，初始树可跑
python3.11 -B affine_examples.py verify           # 全套件 + reference + kernel + 业务链路
python3.11 -B affine_examples.py demo             # 斜边条件风险、更正后重开旧报告
python3.11 -B -m lab.workflow demo                # V1 演示，保持不变
```

`.pointwise-runtime.json` 的 verify/demo 已切到 affine 入口；该 verify 内部仍先执行完整
`tests` 发现（含全部原回归），任一失败返回非零。

## 已知适用边界

- 不通用判定任意删失队列的全局唯一性，仍以 README/AFFINE 的良态 MLE 输入承诺为准。
- 区域几何在接纳时只校验有效性；超出分析支持（如尾部截断比 <1e-8）的记录可保存、可冻结，
  但拟合发布时按 NumericalFailure 处理。
- 裁剪框固定在原始坐标 ±50，配合参数支持（|mu|≤8、sd≤4）遗漏质量远低于误差预算，
  不引入随参数移动的不连续边界。
