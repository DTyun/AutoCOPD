import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# 固定随机种子，保证结果可重复
np.random.seed(42)

# 自动创建数据保存目录，避免文件写入报错
os.makedirs("./data/raw", exist_ok=True)

# 定义各队列参数 (队列名, 总样本数, COPD样本数)
queues = [
    ("derivation_train", 1560, 650),
    ("derivation_val", 390, 130),
    ("external1", 1186, 400),
    ("external2", 225, 75),
    ("external3", 292, 98),
    ("nlst", 453, 150)
]


def check_subgroup_validity(data, subgroup_col, subgroup_value):
    """
    检查指定亚组是否有效（满足统计分析要求）
    :param data: 数据集
    :param subgroup_col: 亚组划分列名
    :param subgroup_value: 亚组取值
    :return: (是否有效, 原因/说明)
    """
    # 筛选该亚组样本
    subgroup_data = data[data[subgroup_col] == subgroup_value]

    # 检查1：亚组是否有样本
    if len(subgroup_data) == 0:
        return False, "无样本"

    # 检查2：亚组是否包含COPD和非COPD两类样本（可计算AUC等指标）
    classes = subgroup_data["copd_diagnosis"].unique()
    if len(classes) < 2:
        return False, "仅含单一COPD诊断类别"

    return True, "有效"


def generate_fake_data(queue_name, total_samples, copd_samples):
    """
    生成单个队列的COPD假数据（整合鲁棒性修复+新增特征+临床逻辑）
    :param queue_name: 队列名称
    :param total_samples: 总样本数
    :param copd_samples: COPD阳性样本数
    :return: 生成的数据集
    """
    # 初始化数据框架
    data = pd.DataFrame()

    # ========== 1. 基础问卷特征 ==========
    # 年龄：35-80岁随机整数
    data["age"] = np.random.randint(35, 81, size=total_samples)

    # 性别：0=女性，1=男性（保证至少1男1女，避免空组）
    if total_samples >= 2:
        gender = np.zeros(total_samples, dtype=int)
        gender[0] = 1  # 强制至少1个男性
        if total_samples > 2:
            gender[1:] = np.random.binomial(1, 0.5, size=total_samples - 1)
        data["gender"] = gender
    else:
        data["gender"] = np.random.binomial(1, 0.5, size=total_samples)
    data["gender"] = data["gender"].astype(int)  # 确保整数类型

    # 饮酒：20%阳性率
    data["drinking"] = np.random.binomial(1, 0.2, size=total_samples)
    # 高血压：30%阳性率
    data["hypertension"] = np.random.binomial(1, 0.3, size=total_samples)

    # ========== 2. 吸烟包年（修复0包年空组问题） ==========
    data["smoking_pack_years"] = np.zeros(total_samples)
    # 结局变量：1=COPD，0=正常
    data["copd_diagnosis"] = 0
    copd_indices = np.random.choice(total_samples, copd_samples, replace=False)
    data.loc[copd_indices, "copd_diagnosis"] = 1
    non_copd_indices = data[data["copd_diagnosis"] == 0].index

    # 保证每个队列至少5例0包年样本（小队列也能满足亚组分析）
    min_zero_smoking = 5
    zero_smoking_count = max(min_zero_smoking, int(total_samples * 0.05))
    zero_smoking_indices = np.random.choice(total_samples, zero_smoking_count, replace=False)
    data.loc[zero_smoking_indices, "smoking_pack_years"] = 0.0

    # 剩余样本按COPD状态生成吸烟包年（指数分布，符合临床逻辑）
    remaining_indices = [i for i in range(total_samples) if i not in zero_smoking_indices]
    remaining_copd = [i for i in copd_indices if i in remaining_indices]
    remaining_non_copd = [i for i in non_copd_indices if i in remaining_indices]

    data.loc[remaining_non_copd, "smoking_pack_years"] = np.random.exponential(8, size=len(remaining_non_copd))
    data.loc[remaining_copd, "smoking_pack_years"] = np.random.exponential(20, size=len(remaining_copd))
    data["smoking_pack_years"] = data["smoking_pack_years"].clip(0, 50)  # 截断0-50包年

    # ========== 3. QCT特征（基础+新增肺叶级特征） ==========
    # 基础QCT特征
    data["whole_lung_LAA950"] = np.zeros(total_samples)
    data["whole_lung_LAA910"] = np.zeros(total_samples)
    data["bronchus_LD"] = np.zeros(total_samples)

    # 新增肺叶级LAA950特征
    data["LAA950_lung"] = np.zeros(total_samples)
    data["LAA950_left_upper_lobe"] = np.zeros(total_samples)
    data["LAA950_left_lower_lobe"] = np.zeros(total_samples)
    data["LAA950_right_upper_lobe"] = np.zeros(total_samples)
    data["LAA950_right_middle_lobe"] = np.zeros(total_samples)
    data["LAA950_right_lower_lobe"] = np.zeros(total_samples)

    # 新增肺叶级LAA910特征
    data["LAA910_left_lower_lobe"] = np.zeros(total_samples)
    data["LAA910_right_lower_lobe"] = np.zeros(total_samples)

    # 新增管腔直径特征
    data["Lumen1_max_diameter"] = np.zeros(total_samples)
    data["Lumen4_average_diameter"] = np.zeros(total_samples)

    # ========== 4. CT报告特征 ==========
    # 基础患病率：肺气肿5%、支气管扩张2%、纤维化1%
    data["emphysema"] = np.random.binomial(1, 0.05, size=total_samples)
    data["bronchiectasis"] = np.random.binomial(1, 0.02, size=total_samples)
    data["fibrosis"] = np.random.binomial(1, 0.01, size=total_samples)
    # COPD组并发症比例调整（避免特征完全区分标签）
    data.loc[copd_indices, "emphysema"] = np.random.binomial(1, 0.3, size=len(copd_indices))
    data.loc[copd_indices, "bronchiectasis"] = np.random.binomial(1, 0.08, size=len(copd_indices))
    data.loc[copd_indices, "fibrosis"] = np.random.binomial(1, 0.05, size=len(copd_indices))

    # ========== 5. QCT特征值填充（增加区间重叠+噪声，避免过拟合） ==========
    # LAA950：COPD/非COPD区间重叠，添加高斯噪声
    data.loc[non_copd_indices, "whole_lung_LAA950"] = np.random.uniform(3, 12, size=len(non_copd_indices))
    data.loc[copd_indices, "whole_lung_LAA950"] = np.random.uniform(8, 22, size=len(copd_indices))
    data["whole_lung_LAA950"] += np.random.normal(0, 0.5, size=total_samples)

    # LAA910：与LAA950正相关
    data["whole_lung_LAA910"] = data["whole_lung_LAA950"] * np.random.uniform(1.2, 1.6, size=total_samples)

    # 支气管直径：区间重叠+噪声
    data.loc[non_copd_indices, "bronchus_LD"] = np.random.uniform(1.8, 3.0, size=len(non_copd_indices))
    data.loc[copd_indices, "bronchus_LD"] = np.random.uniform(2.2, 4.0, size=len(copd_indices))
    data["bronchus_LD"] += np.random.normal(0, 0.2, size=total_samples)

    # 新增肺叶级LAA特征（基于全肺LAA，添加合理变异）
    data["LAA950_lung"] = data["whole_lung_LAA950"] * np.random.uniform(0.95, 1.05, size=total_samples)
    data["LAA950_left_upper_lobe"] = data["whole_lung_LAA950"] * np.random.uniform(0.8, 1.2, size=total_samples)
    data["LAA950_left_lower_lobe"] = data["whole_lung_LAA950"] * np.random.uniform(0.8, 1.2, size=total_samples)
    data["LAA950_right_upper_lobe"] = data["whole_lung_LAA950"] * np.random.uniform(0.8, 1.2, size=total_samples)
    data["LAA950_right_middle_lobe"] = data["whole_lung_LAA950"] * np.random.uniform(0.7, 1.1, size=total_samples)
    data["LAA950_right_lower_lobe"] = data["whole_lung_LAA950"] * np.random.uniform(0.8, 1.2, size=total_samples)
    data["LAA910_left_lower_lobe"] = data["whole_lung_LAA910"] * np.random.uniform(0.8, 1.2, size=total_samples)
    data["LAA910_right_lower_lobe"] = data["whole_lung_LAA910"] * np.random.uniform(0.8, 1.2, size=total_samples)

    # 新增管腔直径特征（COPD/非COPD区间重叠）
    data.loc[non_copd_indices, "Lumen1_max_diameter"] = np.random.uniform(1.5, 3.0, size=len(non_copd_indices))
    data.loc[copd_indices, "Lumen1_max_diameter"] = np.random.uniform(2.0, 4.0, size=len(copd_indices))
    data["Lumen1_max_diameter"] += np.random.normal(0, 0.2, size=total_samples)

    data.loc[non_copd_indices, "Lumen4_average_diameter"] = np.random.uniform(1.0, 2.5, size=len(non_copd_indices))
    data.loc[copd_indices, "Lumen4_average_diameter"] = np.random.uniform(1.5, 3.5, size=len(copd_indices))
    data["Lumen4_average_diameter"] += np.random.normal(0, 0.15, size=total_samples)

    # ========== 6. 派生特征（分组特征，用于亚组分析） ==========
    # 年龄分组：35-49(1)、50-59(2)、60-69(3)、70+(4)
    data["age_group"] = pd.cut(
        data["age"], bins=[34, 49, 59, 69, 100], labels=[1, 2, 3, 4], right=True
    ).astype(int)
    # 吸烟分组：0包年(1)、1-10(2)、11-30(3)、>30(4)
    data["smoking_group"] = pd.cut(
        data["smoking_pack_years"], bins=[-1, 0, 10, 30, 100], labels=[1, 2, 3, 4]
    ).astype(int)

    # ========== 7. 缺失值处理（添加≤1%缺失值，模拟真实医疗数据） ==========
    for col in ["drinking", "fibrosis", "bronchus_LD"]:
        missing_num = max(0, int(total_samples * 0.01))
        missing_indices = np.random.choice(total_samples, missing_num, replace=False)
        data.loc[missing_indices, col] = np.nan

    # ========== 8. 数据格式标准化 ==========
    # 数值特征保留2位小数（符合医疗数据记录规范）
    numeric_cols = [
        "age", "smoking_pack_years", "whole_lung_LAA950", "whole_lung_LAA910", "bronchus_LD",
        "LAA950_lung", "LAA950_left_upper_lobe", "LAA950_left_lower_lobe", "LAA950_right_upper_lobe",
        "LAA950_right_middle_lobe", "LAA950_right_lower_lobe", "LAA910_left_lower_lobe",
        "LAA910_right_lower_lobe", "Lumen1_max_diameter", "Lumen4_average_diameter"
    ]
    data[numeric_cols] = data[numeric_cols].round(2)

    # ========== 9. 数据保存与校验信息打印 ==========
    data.to_csv(f"./data/raw/{queue_name}.csv", index=False, encoding="utf-8")
    # 基础信息
    print(f"✅ 生成{queue_name}假数据：{total_samples}行 × {data.shape[1]}列，COPD比例：{copd_samples / total_samples:.3f}")
    # 性别分布校验
    print(f"  ├─ 性别分布：女性{sum(data['gender'] == 0)}例，男性{sum(data['gender'] == 1)}例")
    # 吸烟分组校验
    print(
        f"  ├─ 吸烟分组：组1(0包年){sum(data['smoking_group'] == 1)}例 | 组2(1-10){sum(data['smoking_group'] == 2)}例 | 组3(11-30){sum(data['smoking_group'] == 3)}例 | 组4(>30){sum(data['smoking_group'] == 4)}例")
    # 缺失值校验
    missing_summary = data[["drinking", "fibrosis", "bronchus_LD"]].isnull().sum()
    print(
        f"  ├─ 缺失值：饮酒{missing_summary['drinking']}例 | 纤维化{missing_summary['fibrosis']}例 | 支气管直径{missing_summary['bronchus_LD']}例")
    print(f"  └─ 数据已保存至 ./data/raw/{queue_name}.csv\n")

    return data


# ========== 主流程：批量生成所有队列数据 + 亚组有效性检查 ==========
if __name__ == "__main__":
    print("=" * 50)
    print("开始生成COPD假数据（含亚组有效性校验）")
    print("=" * 50)

    for queue_name, total_samples, copd_samples in queues:
        print(f"\n【处理队列：{queue_name}】")
        # 生成该队列数据
        data = generate_fake_data(queue_name, total_samples, copd_samples)

        # 亚组有效性检查
        print("📊 亚组有效性分析：")
        # 性别亚组
        for gender_val, gender_name in [(0, "女性"), (1, "男性")]:
            valid, reason = check_subgroup_validity(data, "gender", gender_val)
            count = len(data[data["gender"] == gender_val])
            print(f"  - {gender_name}：{'✅' if valid else '❌'} {reason}（样本数：{count}）")

        # 年龄组亚组
        print("  年龄组：", end="")
        for age_group in [1, 2, 3, 4]:
            valid, _ = check_subgroup_validity(data, "age_group", age_group)
            count = len(data[data["age_group"] == age_group])
            print(f" 组{age_group}({count}例){'✅' if valid else '❌'}", end="")
        print()

        # 吸烟组亚组
        print("  吸烟组：", end="")
        for smoke_group in [1, 2, 3, 4]:
            valid, _ = check_subgroup_validity(data, "smoking_group", smoke_group)
            count = len(data[data["smoking_group"] == smoke_group])
            print(f" 组{smoke_group}({count}例){'✅' if valid else '❌'}", end="")
        print("\n" + "-" * 40)

    print("\n🎉 所有队列假数据生成完成！")
    print("📁 数据文件路径：./data/raw/")
    print("⚠️  注意：所有亚组标记❌的部分无法进行有效的统计分析（无样本/单一类别）")