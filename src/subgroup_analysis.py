import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

class SubgroupAnalyzer:
    def __init__(self, model_path="./results/models/autocopd_final_model.json", processed_dir="./data/processed/", output_dir="./results/metrics/"):
        self.model = self.load_model(model_path)
        self.processed_dir = processed_dir
        self.output_dir = output_dir
        self.target_col = "copd_diagnosis"
        self.selected_features = self.load_selected_features()

    def load_model(self, model_path):
        """加载训练好的模型"""
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            return model
        except Exception as e:
            print(f"模型加载失败：{e}")
            return None

    def load_selected_features(self):
        """加载筛选后的特征"""
        try:
            with open("./data/features/selected_qct_features.txt", "r", encoding="utf-8") as f:
                features = [line.strip() for line in f.readlines()]
            return features
        except FileNotFoundError:
            print("警告：未找到selected_qct_features.txt文件，使用默认QCT特征")
            return ["whole_lung_LAA950", "whole_lung_LAA910", "bronchus_WT", "bronchus_LD"]

    def bootstrap_auc(self, y_true, y_pred_prob, n_bootstrap=2000, random_state=42):
        """计算AUC及2000次Bootstrap 95%CI"""
        np.random.seed(random_state)
        aucs = []
        n_samples = len(y_true)
        for _ in range(n_bootstrap):
            # 重采样
            idx = resample(range(n_samples), replace=True)
            y_boot_true = y_true.iloc[idx] if isinstance(y_true, pd.Series) else y_true[idx]
            y_boot_pred = y_pred_prob[idx]
            # 计算AUC（避免单类样本）
            if len(np.unique(y_boot_true)) == 2:
                auc = roc_auc_score(y_boot_true, y_boot_pred)
                aucs.append(auc)
        # 计算95%CI（百分位法）
        ci_lower = np.percentile(aucs, 2.5)
        ci_upper = np.percentile(aucs, 97.5)
        mean_auc = np.mean(aucs)
        return mean_auc, ci_lower, ci_upper

    def define_subgroups(self, data):
        """定义亚组"""
        subgroups = {
            "age_group": {
                "name": "年龄组",
                "categories": {
                    "1": "35-49岁",
                    "2": "50-59岁",
                    "3": "60-69岁",
                    "4": "≥70岁"
                }
            },
            "gender": {
                "name": "性别",
                "categories": {
                    "0": "女性",
                    "1": "男性"
                }
            },
            "smoking_group": {
                "name": "吸烟包年组",
                "categories": {
                    "1": "0包年",
                    "2": "1-10包年",
                    "3": "11-30包年",
                    "4": "＞30包年"
                }
            }
        }
        return subgroups

    def analyze_subgroups(self, queue_name="external1"):
        """分析指定队列的亚组性能"""
        print(f"===== 开始{queue_name}队列的亚组分析 =====")
        try:
            # 加载数据
            data = pd.read_csv(f"{self.processed_dir}{queue_name}_processed.csv", encoding="utf-8")
            X = data[self.selected_features].copy()
            y = data[self.target_col].copy()
            # 预测概率
            y_pred_prob = self.model.predict_proba(X)[:, 1]
            # 定义亚组
            subgroups = self.define_subgroups(data)
            # 初始化结果表格
            results_df = pd.DataFrame(columns=[
                "亚组类型", "亚组类别", "样本量", "COPD例数", "AUC", "AUC_95CI"
            ])
            # 逐亚组分析
            for subgroup_col, subgroup_info in subgroups.items():
                if subgroup_col in data.columns:
                    print(f"\n分析{subgroup_info['name']}亚组")
                    for cat_code, cat_name in subgroup_info['categories'].items():
                        # 筛选亚组数据（处理浮点数和整数类型）
                        if subgroup_col == "gender":
                            # gender列特殊处理：先转为整数再转为字符串
                            subgroup_mask = data[subgroup_col].fillna(-1).astype(int).astype(str) == cat_code
                        else:
                            # 其他列保持原处理逻辑
                            subgroup_mask = data[subgroup_col].astype(str).str.strip('.0') == cat_code
                        if subgroup_mask.sum() > 0:
                            y_subgroup = y[subgroup_mask]
                            y_pred_subgroup = y_pred_prob[subgroup_mask]
                            # 计算AUC及95%CI
                            if len(np.unique(y_subgroup)) == 2:
                                auc, auc_lower, auc_upper = self.bootstrap_auc(y_subgroup, y_pred_subgroup)
                                auc_ci = f"{auc_lower:.3f}-{auc_upper:.3f}"
                                print(f"{cat_name}：{len(y_subgroup)}例，AUC={auc:.4f}（{auc_ci}）")
                                # 保存结果
                                results_df.loc[len(results_df)] = [
                                    subgroup_info['name'], cat_name, len(y_subgroup), y_subgroup.sum(),
                                    round(auc, 4), auc_ci
                                ]
                            else:
                                print(f"{cat_name}：{len(y_subgroup)}例，仅单一类别，无法计算AUC")
                                # 保存结果（仅单一类别）
                                results_df.loc[len(results_df)] = [
                                    subgroup_info['name'], cat_name, len(y_subgroup), y_subgroup.sum(),
                                    "仅单一类别", "无法计算"
                                ]
                        else:
                            print(f"{cat_name}：无样本")
                            # 保存结果（无样本）
                            results_df.loc[len(results_df)] = [
                                subgroup_info['name'], cat_name, 0, 0,
                                "无样本", "无样本"
                            ]
                else:
                    print(f"警告：{subgroup_info['name']}（{subgroup_col}）列不存在于数据中")
            # 保存亚组分析结果
            output_path = f"{self.output_dir}subgroup_analysis_{queue_name}.csv"
            results_df.to_csv(output_path, index=False, encoding="utf-8")
            print(f"\n===== 亚组分析完成，结果保存至：{output_path} =====")
            print("\n亚组分析汇总：")
            print(results_df[["亚组类型", "亚组类别", "样本量", "AUC", "AUC_95CI"]].to_string(index=False))
            return results_df
        except Exception as e:
            print(f"亚组分析过程中出错：{e}")
            return None

    def run(self, queue_names=["external1", "external2", "external3", "nlst"]):
        """批量分析多个队列的亚组"""
        print("===== 开始批量亚组分析 =====")
        for queue in queue_names:
            self.analyze_subgroups(queue)
        print("===== 批量亚组分析完成 =====")

if __name__ == "__main__":
    analyzer = SubgroupAnalyzer()
    analyzer.run()