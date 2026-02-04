import pandas as pd
import numpy as np
import shap
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

class FeatureSelector:
    def __init__(self, processed_dir="./data/processed/", output_dir="./data/features/"):
        self.processed_dir = processed_dir
        self.output_dir = output_dir
        self.target_col = "copd_diagnosis"
        self.qct_cols = None  # 仅筛选QCT特征（单模态模型）

    def load_processed_data(self, queue_name="derivation_train"):
        """加载预处理后的训练集（仅用训练集筛选特征，避免数据泄露）"""
        data = pd.read_csv(f"{self.processed_dir}{queue_name}_processed.csv", encoding="utf-8")
        # 定义QCT特征列（需根据实际数据调整，对应论文47个QCT特征）
        self.qct_cols = [col for col in data.columns if "LAA" in col or "WT" in col or "LD" in col or "WA" in col]
        print(f"加载训练集数据，QCT特征数量：{len(self.qct_cols)}")
        X = data[self.qct_cols].copy()
        y = data[self.target_col].copy()
        return X, y

    def step1_remove_low_variance(self, X):
        """第一步：剔除零/近零方差变量"""
        selector = VarianceThreshold(threshold=0.01)  # 近零方差阈值（可微调）
        X_high_var = selector.fit_transform(X)
        # 筛选后的特征名
        selected_cols = X.columns[selector.get_support()].tolist()
        print(f"第一步筛选：剔除低方差变量，剩余QCT特征数：{len(selected_cols)}")
        return pd.DataFrame(X_high_var, columns=selected_cols), selected_cols

    def step2_shap_selection(self, X, y, top_k=10):
        """第二步：SHAP值筛选TopK特征"""
        # 训练临时XGB模型（用默认超参数）
        temp_model = XGBClassifier(
            max_depth=4, eta=0.036, n_estimators=173,  # 论文最优超参数
            objective="binary:logistic", random_state=42
        )
        temp_model.fit(X, y)
        # 计算SHAP值
        explainer = shap.TreeExplainer(temp_model)
        shap_values = explainer.shap_values(X)
        # 计算平均绝对SHAP值（特征重要性）
        shap_importance = pd.DataFrame({
            "feature": X.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0)
        }).sort_values("mean_abs_shap", ascending=False)
        # 筛选TopK特征
        top_features = shap_importance.head(top_k)["feature"].tolist()
        print(f"第二步筛选：SHAP值Top{top_k}特征：{top_features}")
        # 保存特征重要性结果
        shap_importance.to_csv(f"{self.output_dir}shap_feature_importance.csv", index=False, encoding="utf-8")
        return top_features

    def save_selected_features(self, top_features):
        """保存筛选后的特征名，用于后续建模"""
        with open(f"{self.output_dir}selected_qct_features.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(top_features))
        print(f"筛选后的Top10 QCT特征已保存至：{self.output_dir}selected_qct_features.txt")

    def run(self):
        print("===== 开始特征筛选流程 =====")
        try:
            # 1. 加载训练集QCT特征
            X_qct, y = self.load_processed_data()
            # 2. 第一步：剔除低方差变量
            X_high_var, selected_cols = self.step1_remove_low_variance(X_qct)
            # 3. 第二步：SHAP值筛选Top10特征
            top10_features = self.step2_shap_selection(X_high_var, y, top_k=10)
            # 4. 保存筛选结果
            self.save_selected_features(top10_features)
            print("===== 特征筛选流程完成 =====")
        except Exception as e:
            print(f"特征筛选过程中出错：{e}")

if __name__ == "__main__":
    selector = FeatureSelector()
    selector.run()