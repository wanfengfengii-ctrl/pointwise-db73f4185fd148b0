# 仿射检测区域与同一样本条件报告

README.md 描述的 V1 完整可运行，原源码和原测试保留。本次增加线性组合检测通道；尚未实现 region 接纳、区域似然或条件报告。Python 标准库与 SQLite 不变。以下为新增公开契约；与本文件无关的 V1 行为仍以 README.md 为准。

## 来源、冻结及身份

Workflow.put_batch(batch_id, rows) 的行仍含 cohort_id、sample_id、rev、count，另严格包含 x 或 region 之一；其余字段规则不变。region 严格为：

    {"le": [[a,b,c], ...], "eq": [[a,b,c], ...]}

le 表示 a*X1+b*X2 <= c，eq 表示 a*X1+b*X2 = c。两项均须为列表（可空），分别至多 6、2 条；每条为三元素列表。a、b 是非布尔整数，绝对值不超过 16，不同时为零；c 按 V1 finite_number 规则转为有限 binary64，负零归零。有效性按转换后 binary64 的精确有理值判定，不以浮点容差合并平行线、矛盾或窄区域。不得将未说明的任意精度十进制语义用于 c。

等式交集记作 A，秩 r 为等式法向量的秩。不相容等式，或不等式交集在 A 内没有正的 (2-r) 维测度，均为 ValueError，整批回滚。没有等式时要求正面积；不等式隐含的线或点不能自动成为精确观测。秩 1 允许无界线、射线、正长度线段；秩 2 的可行单点具有正的零维测度。允许冗余、平行、不活动约束和整个平面。

保存规范化的原约束列表，保留其顺序和重复；无需把几何等价的两种写法视为同版重放。系数保持整数，c 规范为浮点；等式缩放不改变计算结果，但改写同一个源修订的内容仍是身份冲突。输入对象、回执、冻结载荷与存储相互独立。新行沿用最高修订、批次重放、完整队列冻结及精确 source_links。旧 x 行的规范载荷和旧任务、成果编码、ID 解释保持兼容。

## 观测似然

每条 region 行的 L 为 f_mu,Sigma 在 R=A∩所有半平面上的 (2-r) 维欧氏 Hausdorff 积分。秩 0 是二维面积积分；秩 1 是欧氏弧长积分；秩 2 是点处的二元联合密度。不是线性通道读数的未经归一化密度，不按等式数量连乘密度，不因重复、重排或合法非零缩放等式而改变 L。斜线参数化的尺度必须符合欧氏长度。

保留 observation_likelihood.observation_log_likelihood(row, model, engine) 的原入口，扩展为接受仅含 region 的行描述；含 x 的调用不变。该低层入口接收合法行描述，返回 log(L)，几何输入检查仍由来源接纳负责。低层入口用于固定模型独立核账，不要求增加模型登记 API。

fit 对冻结任务的所有 x 与 region 行最大化 sum(count*log(L))，返回原坐标 mu、完整 covariance、log_likelihood 和原数值状态。旧 x 的统计口径不变。40 种模式按规范观测描述区分，身份、rev 和 count 不计入模式；无需几何等价归并。沿用 README 的唯一有限良态内部全局 MLE 输入承诺、参数支持、曲率条件与参数误差要求，不要求程序通用判定全局唯一性。允许全量重算和有限交点枚举，复用现有积分器与优化器；不限定组织方法或新增性能目标。

对新增 region 的数值承诺是：在最优点邻域，R 在 A 上的截断概率 L(R)/L(A) 至少 1e-8，秩 2 时该比值为 1。L(A) 是同一模型在整个 A 上的同测度积分，不是额外观测。贡献、目标与权重运算须可表示为有限 binary64。无法满足既有数值支持、收敛或精度时抛 lab.numerics.NumericalFailure，不发布；源接纳与冻结仍可保存超出分析支持的几何有效记录。不能仅因存在 region 而一律拒绝支持范围内的任务。

## 同一样本的条件风险

新增 Workflow.conditional_risk(fit_id, sample_key, thresholds)，sample_key 为两个非空字符串组成的列表或元组 [cohort_id,sample_id]；thresholds 沿用两个有序有限数规则。非法参数 ValueError，未知拟合或未出现在该拟合父任务中的样本 KeyError；即使当前来源存在该样本也不得替代。支持父任务中的 x 或 region 样本，count 不改变该样本事件的条件概率。

报告的 probability 为 L(R∩{X1>t1,X2>t2})/L(R)，分子分母用同一个模型和观测测度。旧 x 按其精确坐标和允许区间解释为相应仿射观测。严格阈值在秩 1、2 下仍须严格：若整条观测线或点位于阈值上，不能将 > 偷换为 >=。分子为空或在原观测测度下为零时返回 0，不能因条件交集零测度而拒绝整份报告。分母不可解析时 NumericalFailure。概率绝对误差 <= 1e-7，不是一次独立新测量的超限率。

返回独立字典，保留通用 result_id、contract，并含：
- kind 为 "conditional_report"；
- fit_id、model_digest（沿用 canonical_digest(完整父拟合)）；
- sample（父任务中的完整规范行，含身份、rev、count 及 x 或 region）；
- thresholds、denominator（正且有限的 L(R)，不是 log 值，也不取 count 次幂）、probability。

同一 fit_id、sample_key 和规范阈值重放返回原报告；换样本不能错误复用同阈值的旧报告，也不能与无条件 risk 的请求混淆。无条件 risk 不改语义。

read_result 支持新报告且不计算。export_result 支持新报告，仍返回 report、fit、task、sources，核验新报告样本确属该固定父任务、修订与观测描述相符，并核验全部原父链与摘要；缺失或错配 IntegrityError，不读 current，不重算分母。来源更正、修改参数/返回对象及关闭重开不改变旧成果。

新计算沿用 engine 的 fit、minimize、integrate、linear、risk 失败端口（risk 作用于条件报告计算）；归档沿用 result.payload、result.index、result.parent、commit。发布各阶段失败不留成果、索引或父链半成品，已有成果不变，退出失败作用域后可重试。重放、读取和导出无需重新触发计算。

## 有限样例与交付入口

affine_examples.py 提供非答案输入和仅针对固定样例的独立核账；不包含通用区域分解、目标似然或报告实现。

独立标准正态下 R={X1>=0,X2>=0,X1+X2<=1}：
L(R)≈0.06773003070084909；条件双项 >0.4 的概率≈0.03772562049949979。
用 [0,1]^2 包围盒代替会得到约 0.2966742221666794，业务结果不同。

独立标准正态在整条 X1+X2=1 上的欧氏长度贡献≈0.31069656037692774；加 X1>=0、X2>=0 后约为 0.16171752171312473。二者均不受等式乘 2 影响。X1+X2=0 上双项严格 >0 的条件概率为 0。

四个点 (±1,±1) 各 count=4，再各加 count=1 的四个反射三角区域 {sx*X1>=0,sy*X2>=0,sx*X1+sy*X2<=1}，得到 8 模式队列。对称参照 mu=(0,0)、covariance=diag(0.8307377982665323)、log L≈-55.85222847626954。参考模块独立核对标量得分方程和积分，不代替任意队列求解。

保留原测试字节；新增独立核账应覆盖斜边、秩 0/1/2、等式冗余缩放、几何拒绝、严格阈值，以及冻结拟合—条件报告—来源更正—重开导出的共同链路。可复用示例，补充自己的有限对照。交付简短说明各职责交接的观测测度与固定样本不变量。

现有 .pointwise-runtime.json 格式沿用：version=1、kind=cli，setup 是 argv 数组列表，verify/demo 各为 argv。完成后扩展 verify，使全部原回归和新增业务核账都实际执行且失败返回非零；demo 展示斜边条件风险、来源更正后重开旧报告及固定证据。无需增加新的通用 CLI 命令或界面。

可执行命令：
    python3.11 -B affine_examples.py reference
    python3.11 -B affine_examples.py verify
    python3.11 -B affine_examples.py demo
    python3.11 -B -m unittest discover -s tests -v

reference 只执行独立固定样例核账，初始树即可运行；verify/demo 依赖本题尚缺的新能力。原运行清单在初始树保持 V1 的正常核验与演示，不能把新能力未实现当成 V1 回归故障。
