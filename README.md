# COPD风险预测系统

## 项目介绍

本项目是一个基于机器学习的COPD（慢性阻塞性肺疾病）风险预测系统，复刻自相关研究论文。系统使用XGBoost模型，通过QCT（定量计算机断层扫描）特征来预测患者的COPD风险。

### 主要功能

- **单样本预测**：输入QCT特征值，获得COPD风险预测结果
- **批量预测**：上传包含多个样本的CSV文件，获得批量预测结果并下载
- **自定义分析**：上传数据文件，获得详细的统计分析结果
- **可视化界面**：用户友好的Web界面，方便操作和查看结果

## 项目结构

```
├── app.py                # Flask应用主文件
├── main.py               # 主脚本，包含完整的模型训练和评估流程
├── requirements.txt      # 项目依赖
├── data/                 # 数据目录
│   ├── raw/              # 原始数据
│   ├── processed/        # 处理后的数据
│   └── features/         # 特征相关文件
├── src/                  # 源代码目录
│   ├── data_preprocess.py    # 数据预处理
│   ├── feature_selection.py  # 特征选择
│   ├── model_training.py     # 模型训练
│   ├── model_evaluation.py   # 模型评估
│   ├── subgroup_analysis.py  # 亚组分析
│   └── visualization.py      # 可视化
├── results/              # 结果目录
│   ├── models/           # 训练好的模型
│   ├── metrics/          # 模型性能指标
│   └── figures/          # 可视化图表
└── templates/            # HTML模板
    └── index.html        # 主页面
```

## 安装说明

### 环境要求

- Python 3.7+
- 主要依赖库：
  - pandas
  - numpy
  - scipy
  - scikit-learn
  - xgboost
  - shap
  - fancyimpute
  - hyperopt
  - pydelong
  - matplotlib
  - seaborn
  - flask

### 安装步骤

1. **克隆项目**

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **准备数据**
   - 将原始数据放入 `data/raw/` 目录
   - 或使用 `生成假数据.py` 生成示例数据

## 使用方法

### 1. 模型训练（首次使用时需要）

```bash
python main.py
```

此命令会执行完整的流程：
- 数据预处理
- 特征选择
- 模型训练和超参数调优
- 模型评估
- 生成结果和图表

### 2. 启动Web应用

```bash
python app.py
```

应用将在 `http://localhost:5000` 启动

### 3. 使用Web界面

1. **单样本预测**
   - 在首页输入QCT特征值
   - 点击「预测」按钮
   - 查看预测结果

2. **批量预测**
   - 准备包含QCT特征的CSV文件
   - 在「批量预测」部分上传文件
   - 点击「预测」按钮
   - 下载包含预测结果的CSV文件

3. **自定义分析**
   - 上传包含QCT特征的CSV文件
   - 点击「分析」按钮
   - 查看详细的统计分析结果

## 技术细节

### 模型信息

- **模型类型**：XGBoost分类器
- **预测阈值**：0.352（根据论文设定）
- **特征数量**：默认使用3个关键QCT特征
  - whole_lung_LAA950
  - bronchus_LD
  - whole_lung_LAA910

### 数据格式

#### 单样本预测输入格式

```json
{
  "whole_lung_LAA950": 数值,
  "bronchus_LD": 数值,
  "whole_lung_LAA910": 数值
}
```

#### 批量预测CSV文件格式

| whole_lung_LAA950 | bronchus_LD | whole_lung_LAA910 |
|-------------------|-------------|-------------------|
| 数值              | 数值        | 数值              |
| 数值              | 数值        | 数值              |

## API接口

### 1. 单样本预测

- **URL**: `/predict`
- **方法**: POST
- **请求体**: JSON格式，包含QCT特征
- **响应**: JSON格式，包含预测概率和诊断结果

### 2. 批量预测

- **URL**: `/batch_predict`
- **方法**: POST
- **请求体**: multipart/form-data，包含CSV文件
- **响应**: CSV文件，包含预测结果

### 3. 自定义分析

- **URL**: `/custom_analysis`
- **方法**: POST
- **请求体**: multipart/form-data，包含CSV文件
- **响应**: JSON格式，包含分析结果

## 结果解释

- **predicted_probability**: 预测的COPD风险概率（0-1之间）
- **diagnosis**: 诊断结果
  - COPD高风险：概率 >= 0.352
  - COPD低风险：概率 < 0.352

## 注意事项

1. 首次使用前请确保运行 `main.py` 进行模型训练
2. 批量预测和自定义分析功能需要上传格式正确的CSV文件
3. 系统默认使用论文中确定的3个关键特征，如需使用其他特征，请修改相关配置

## 项目评估

模型性能指标存储在 `results/metrics/` 目录中，包括：
- 各队列的模型性能
- 亚组分析结果

可视化图表存储在 `results/figures/` 目录中，包括：
- ROC曲线
- 校准曲线
- DCA曲线
- SHAP特征重要性图

## 参考资料

- 相关研究论文
- XGBoost官方文档
- Flask官方文档

## 许可证

本项目仅供研究和学习使用。
