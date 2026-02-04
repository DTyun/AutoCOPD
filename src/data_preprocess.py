import pandas as pd
import numpy as np
from fancyimpute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, data_dir="./data/raw/", output_dir="./data/processed/"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.que_cols = None  # 问卷特征列（需根据实际数据调整）
        self.qct_cols = None  # QCT特征列
        self.ct_report_cols = None  # CT报告特征列

    def load_data(self, queue_name):
        """加载指定队列数据"""
        file_path = f"{self.data_dir}{queue_name}.csv"
        data = pd.read_csv(file_path, encoding="utf-8")
        print(f"加载{queue_name}数据：{data.shape[0]}行 × {data.shape[1]}列")
        return data

    def define_feature_cols(self, data):
        """定义3类特征列（根据实际数据列名调整）"""
        # 实际数据列名
        self.que_cols = ["age", "gender", "smoking_pack_years", "drinking", "hypertension"]  # 问卷特征
        self.qct_cols = ["whole_lung_LAA950", "whole_lung_LAA910", "bronchus_LD"]  # QCT特征
        self.ct_report_cols = ["emphysema", "bronchiectasis", "fibrosis"]  # CT报告特征
        self.all_feature_cols = self.que_cols + self.qct_cols + self.ct_report_cols
        self.target_col = "copd_diagnosis"  # 结局变量（1=COPD，0=正常）
        return self.all_feature_cols, self.target_col

    def data_cleaning(self, data):
        """数据清洗：剔除异常值+逻辑校验"""
        # 1. 剔除结局变量缺失行
        data = data.dropna(subset=[self.target_col])
        # 2. 剔除异常值（示例：年龄35-80岁，吸烟包年≤100）
        data = data[(data["age"] >= 35) & (data["age"] <= 80)]
        data = data[(data["smoking_pack_years"] >= 0) & (data["smoking_pack_years"] <= 100)]
        # 3. 逻辑一致性校验（示例：女性吸烟包年＞0时需有吸烟史标记）
        if "smoking" in data.columns:
            data.loc[(data["gender"] == 2) & (data["smoking_pack_years"] > 0) & (data["smoking"] != 1), "smoking"] = 1
        print(f"清洗后数据：{data.shape[0]}行 × {data.shape[1]}列")
        return data

    def variable_encoding(self, data):
        """变量编码：对齐论文编码规则"""
        # 1. 二分类变量编码（1=是，0=否）
        for col in ["smoking", "drinking", "hypertension", "emphysema"]:
            if col in data.columns:
                data[col] = data[col].map({1: 1, 2: 0})  # 原数据1=是，2=否 → 转为1=是，0=否
        # 2. gender列特殊处理（保持0=女性，1=男性）
        if "gender" in data.columns:
            # 确保gender值在0和1之间
            data["gender"] = data["gender"].clip(0, 1).astype(int)
        # 2. 年龄组编码（35-49=1，50-59=2，60-69=3，≥70=4）
        data["age_group"] = pd.cut(
            data["age"], bins=[34, 49, 59, 69, 100], labels=[1, 2, 3, 4], right=True
        ).astype(int)
        # 3. 多分类有序编码（示例：吸烟包年组）
        data["smoking_group"] = pd.cut(
            data["smoking_pack_years"], bins=[-1, 0, 10, 30, 100], labels=[1, 2, 3, 4]
        ).astype(int)
        return data

    def impute_missing(self, data):
        """分队列插补缺失值（IterativeImputer算法）"""
        # 仅对特征列插补，结局变量不插补
        feature_data = data[self.all_feature_cols].copy()
        # IterativeImputer插补（n_estimators=100，与原论文一致）
        imputer = IterativeImputer(max_iter=10, random_state=42)
        feature_imputed = imputer.fit_transform(feature_data)
        # 替换回原数据
        data[self.all_feature_cols] = feature_imputed
        # 检查插补后缺失率
        missing_rate = data[self.all_feature_cols].isnull().sum().sum() / (len(self.all_feature_cols) * len(data))
        print(f"缺失值插补完成，剩余缺失率：{missing_rate:.4f}（≤2%符合要求）")
        return data

    def run(self, queue_names=["derivation_train", "derivation_val", "external1", "external2", "external3", "nlst"]):
        """批量处理所有队列"""
        for queue in queue_names:
            print(f"\n===== 处理{queue}队列 =====")
            try:
                # 1. 加载数据
                data = self.load_data(queue)
                # 2. 定义特征列和目标列
                self.all_feature_cols, self.target_col = self.define_feature_cols(data)
                # 3. 数据清洗
                data_clean = self.data_cleaning(data)
                # 4. 变量编码
                data_encoded = self.variable_encoding(data_clean)
                # 5. 缺失值插补
                data_processed = self.impute_missing(data_encoded)
                # 6. 保存预处理后数据
                output_path = f"{self.output_dir}{queue}_processed.csv"
                data_processed.to_csv(output_path, index=False, encoding="utf-8")
                print(f"{queue}队列处理完成，保存至：{output_path}")
            except Exception as e:
                print(f"处理{queue}队列时出错：{e}")

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run()