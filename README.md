# 青溪水质实验室离线分析

Python 3.11 标准库，SQLite，无第三方依赖。两个指标的单位都是 mg/L，协方差单位是 (mg/L)^2。上游负责单位统一和空白校正；负校正读数合法。

化验员提交仪器离线记录，分析员冻结完整队列，质控负责人使用两个指标共同超限概率决定复测优先级。本工程保存分析依据，不执行仪器控制或复测排程。

## 一天的工作

1. put_batch 接纳完整批次，校验两项观测、次数和修订。
2. 原子保存不可变修订和批次回执，等值重复不累加次数。
3. 以更高修订更正错误检测限；低版本晚到只作历史。
4. freeze 固定所选队列全部当前成员和精确来源。
5. fit 仅读取该任务，拟合二元正态联合模型。
6. risk 消费固定 fit_id 和有序阈值，发布共同超限报告。
7. export_result 核对报告、拟合、任务和源修订，导出完整证据。
8. 关闭重开后读取旧报告；源更正不改变旧证据。使用新数据须冻结新任务。

## API 与输入

入口为 lab.workflow.Workflow(path)，支持上下文管理器。默认 :memory:；传项目数据库路径可持久化。

put_batch(batch_id, rows) 的 rows 为非空列表，每行严格包含：cohort_id、sample_id（非空字符串）、rev（正整数）、count（正整数）、x（两个按指标顺序排列的约束）。布尔值不是整数或读数。四类约束：

- {"kind":"exact","value":1.0}
- {"kind":"left","upper":0.2}
- {"kind":"right","lower":0.5}
- {"kind":"interval","lower":-0.8,"upper":0.1}

界限须有限 binary64；区间 lower < upper；拒绝多余字段、空值、NaN 和无穷。无穷范围只能用 left/right 表达。负零规范为零。样本身份是 (cohort_id,sample_id)，修订身份再含 rev。等值同版重放；同版不同内容整批 ValueError。批内等值重复折叠。最高修订按号码选择。批次身份绑定排序去重后的完整提交，同批次等值重放返回原回执，即使后来出现新修订。回执含本批 revisions 和接纳当时 current。

freeze(task_id, cohort_ids) 排序去重选择，返回含 input、input_digest 的固定任务。未知队列 KeyError，空选择 ValueError。相同任务重放须当前解析出的完整输入仍等值；源更正后使用同身份会冲突。read_frozen 可通过 workflow.tasks 读取。调用者对象与保存对象独立。

fit(task_id) 返回 kind=fit 的成果，result_id 即 fit_id，含 mu、完整 covariance、log_likelihood、numeric_status、task_id、input_digest、contract。相同任务及计算契约复用已有拟合。

risk(fit_id,[t1,t2]) 返回 kind=report，result_id 即报告身份，含 fit_id、thresholds、probability、model_digest、contract。阈值有序且有限；重复请求复用原报告。

read_result(result_id) 仅读取独立副本。未知成果 KeyError；不调用计算。export_result(report_id) 返回 report、fit、task、sources，只支持报告，父链或编码不完整抛 lab.codecs.IntegrityError。它沿精确修订读取，不读 current。

## 统计口径和数值支持

模型 X ~ N_2(mu,Sigma)，Sigma 对称正定。每行 L_i：

- 两精确值：联合密度 f(x,y)。
- 第一项精确：f_1(x) * [Phi((upper-mu_2|1)/sd_2|1)-Phi((lower-mu_2|1)/sd_2|1)]；另一轴对称。
- 两删失：在两个允许范围的笛卡尔积上积分联合密度。

总目标 sum(count_i * log(L_i))，包含通常正态密度常数。MLE 协方差以 n 作分母。初始猜测可用代表值，但最终目标始终为上述联合似然。不能将删失界限或中点当作最终读数。

条件均值为 mu_y + Sigma_yx/Sigma_xx*(x-mu_x)，条件方差为 Sigma_yy-Sigma_yx^2/Sigma_xx。联合矩形概率通过标准化第一轴的条件积分计算。risk 返回 P(X1>t1,X2>t2)，消费同一个完整协方差。连续分布下端点无额外质量。

分析支持至多 40 种不同 x 模式，不限展开样本数；count 不改变模式数。调用方承诺原目标存在唯一有限内部的良态全局 MLE。具体数值支持：MLE 的 |mu_j| <= 8、标准差在 [0.2,4]、|rho| <= 0.9；在最优点附近每条删失观测概率至少 1e-10，目标及权重运算可表示为有限 binary64。以 (mu1,mu2,log(sd1),log(sd2),atanh(rho)) 为坐标的平均负对数似然，在求解收敛邻域内正定，Hessian 逆的无穷范数 <= 100，Cholesky 各消元主元 > .002。输入的全局唯一性是模型适用承诺，程序不声称通用判定任意删失队列的全局唯一性。

均值及协方差每独立元素绝对误差 <= 1e-6，风险绝对误差 <= 1e-7。数值实现采用正定参数化、五点得分、BFGS、Newton 复核和正曲率检查；不只看优化器标志。numeric_status 中参数误差为数值估计，不是全局最优性的证明。无法解析的极小观测概率、病态曲率、未收敛、非有限结果或超支持参数抛 NumericalFailure，不发布成果。源接纳和冻结仍可保存不满足分析支持的记录。

积分以 12/24 阶自适应 Gauss-Legendre 配对估计误差，矩形计算另核对相对预算。标准化积分截到 +/-12，单侧遗漏正态质量小于 2e-33；极端风险阈值据此处理。允许的概率舍入修正至多 3e-13，明显越界拒绝。没有蒙特卡洛或随机初值。

samples.py 固定六点解析样例、四类混合样例与参照。六点 mu=(0,0)，Sigma=[[2/3,1/3],[1/3,2/3]]，零阈值风险 1/3。四角独立对照为 1/4。混合样例 12 模式、36 次：mu=(0.054220399820,0.062317514860)，Sigma=[[0.581417380321,0.354136353153],[0.354136353153,0.655918716013]]；log L=-72.274849256408；阈值 (0.15,0.25) 风险 0.278167435769。

tests/test_numerics.py 使用独立 70 位 Decimal 正态级数和固定 48/80 阶积分评价参照，并检查五点得分。该核对不调用生产似然、CDF 或积分器；全局唯一性仍按上述输入承诺。另有解析象限公式、次数展开及指标交换回归。

## 归档与失败边界

schema.py 管结构版本，storage.py 管事务；observations/revisions/batch_ledger 管不可变来源；tasks/snapshots/source_links 管完整冻结；fitting/fit_solver/observation_likelihood 管模型；risk/joint_tail/reports 管概率消费；results/codecs/evidence 管固定归档。所有 24 个生产模块接入这一闭环。

接纳、冻结、拟合发布、报告发布各自完整事务。数值计算在写事务外完成。结果载荷、请求索引和父链一起提交，失败整体回滚。请求摘要只作索引，源批次和任务等值还比较完整内容。内容编码带版本和 SHA-256；损坏、父链错配明确拒绝，不以重算修补。

公开测试端口：workflow.engine.failing("fit"|"minimize"|"integrate"|"linear"|"risk") 为作用域数值失败；workflow.storage.failing("batch"|"task.snapshot"|"task.links"|"result.payload"|"result.index"|"result.parent"|"commit") 为作用域存储失败。离开作用域可重试，异常原样传递。不启用时按真实计算和 SQLite 事务运行。

初始化只创建缺失结构、核对版本；不清库、不导样例、不覆盖用户修订。未知结构或版本拒绝。读取不重新生成时间或统计量；成果中不使用墙钟时间。

## 运行

运行清单 .pointwise-runtime.json 的 verify 为完整原套件，demo 在测试临时目录运行真实接纳、更正、冻结、拟合、报告、导出及重开。也可手工运行：

    python3.11 -B -m unittest discover -s tests -v
    python3.11 -B -m lab.workflow demo

持久调用（JSON 文件须是上述 rows 列表）：

    python3.11 -B -m lab.workflow --db lab.sqlite put day1 rows.json
    python3.11 -B -m lab.workflow --db lab.sqlite freeze task1 cohort1
    python3.11 -B -m lab.workflow --db lab.sqlite fit task1

取 fit 返回的 result_id，再调用 risk FIT_ID 0.15 0.25；read RESULT_ID、export REPORT_ID 读取及导出。CLI 成功输出 JSON、退出 0；输入、数值、数据库或文件错误输出 error JSON、退出 2。demo 输出先经断言验证的 POINTWISE_SCENARIO 实测记录。非 demo 命令需要 --db；样例不会自动写入用户数据库。

原回归包括：修订乱序和批次重放、整批拒绝、完整队列及任务绑定、四类似然与联合风险、独立高精度参照、次数与交换等价、数值失败、各发布阶段回滚及稳定重试、编码与缺失父链拒绝、更正后固定证据、真实文件数据库重开和 CLI 子进程闭环。
