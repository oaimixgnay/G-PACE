# G-PACE

本仓库提供论文现有实验的数据处理、数据划分、模型训练、对照模型和统计分析代码。Python 包名为 `gpace`。

## 代码内容

1. **数据预处理**：欧洲和 G2F 数据读取、清洗、基因型与环境特征转换、环境序列构建、阶段暴露监督标签计算。
2. **数据划分**：基因型／环境分组验证、区域和胁迫组合完整留出，以及 G2F 历史年份训练、2024 年评价。
3. **主模型**：TCN–FiLM 阶段模型、交叉拟合、环境均值与材料偏差分解、CatBoost 产量预测及模型保存。
4. **六模型比较**：G-PACE、Direct CatBoost、EC-GBLUP、Reaction norm、CERIS-JGRA 和 GEFormer-style。GEFormer-style 为独立实现。
5. **其他实验**：不同输入天数、阶段分支消融、固定日历对照、环境适用性、预测区间和材料选择分析。
6. **生物学相关统计分析**：阶段暴露与反应规范、粒数／粒重预测、候选标记×阶段暴露互作及环境内暴露置换。

## 安装

仅运行现有预测的统计分析：

```bash
python -m pip install -e .
```

训练模型时先安装适合本机的 PyTorch，再安装训练依赖：

```bash
python -m pip install -e '.[train,dev]'
```

要求 Python 3.10 及以上；训练入口使用 Linux/Bash。

## 先核对现有预测

在仓库根目录执行：

```bash
python scripts/verify_release.py --check-metrics
python scripts/recompute_g2f_controls.py
python analyses/g2f_six_models/recompute_g2f.py
```

也可直接执行 `bash workflows/analyze_predictions.sh`。程序读取包内预测记录，输出数值结果到 `outputs/`。

## 训练入口

```bash
# 欧洲数据：按顺序运行主模型、对照及其他实验
bash workflows/europe.sh /原始欧洲数据目录
bash workflows/europe_benchmarks.sh /原始欧洲数据目录
bash workflows/windows.sh /原始欧洲数据目录
bash workflows/environment_holdouts.sh /原始欧洲数据目录
bash workflows/europe_evidence.sh /原始欧洲数据目录

# G2F：主模型，再运行配对对照
bash workflows/g2f.sh /原始G2F数据目录
bash workflows/g2f_controls.sh /原始G2F数据目录

# 按原分组和已选参数重新拟合 G2F 对照模型
bash workflows/g2f_comparators.sh /原始G2F数据目录
```

欧洲主模型使用保存的 Day-100 配置。G2F 主模型保持欧洲 TCN–FiLM 结构，在 G2F 历史数据上重新训练权重；默认比较使用包内五个对照的原有预测。对照重新拟合入口读取保存的已选参数，保留原来的两组模型和随机种子偏移，不重新搜索参数。

主比较与配对对照分别保留对应的 GPU／CPU 产量模型后端。配对无阶段和固定日历训练依赖主模型训练生成的完整 OOF 缓存目录。通过 `PYTHON`、`DEVICE`、`THREADS`、`STAGE_DEVICE` 等环境变量调整运行环境。

原始输入文件要求见 [数据说明](docs/DATA.md)，程序对应关系见 [运行说明](docs/WORKFLOWS.md)。