import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import chi2_contingency
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
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
            print(f"模型加载完成：{model_path}")
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

    def load_all_queues(self):
        """加载所有队列数据"""
        queues = {
            "derivation_train": "推导队列-训练集",
            "derivation_val": "推导队列-内部验证集",
            "external1": "中国外部验证集1",
            "external2": "中国外部验证集2",
            "external3": "中国外部验证集3",
            "nlst": "美国NLST队列"
        }
        data_dict = {}
        for queue_code, queue_name in queues.items():
            try:
                data = pd.read_csv(f"{self.processed_dir}{queue_code}_processed.csv", encoding="utf-8")
                X = data[self.selected_features].copy()
                y = data[self.target_col].copy()
                # 预测概率
                y_pred_prob = self.model.predict_proba(X)[:, 1]
                data_dict[queue_code] = {
                    "name": queue_name,
                    "X": X,
                    "y": y,
                    "y_pred_prob": y_pred_prob
                }
                print(f"加载{queue_name}：{len(y)}例，实际COPD率：{y.mean():.3f}，预测COPD率：{y_pred_prob.mean():.3f}")
            except Exception as e:
                print(f"加载{queue_name}失败：{e}")
        return data_dict

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

    def hosmer_lemeshow_test(self, y_true, y_pred_prob, n_bins=10):
        """HL检验（校准度）"""
        # 转换为pandas Series
        y_true_series = pd.Series(y_true)
        y_pred_prob_series = pd.Series(y_pred_prob)
        # 分10组
        groups = pd.qcut(y_pred_prob_series, q=n_bins, duplicates="drop")
        # 计算每组实际阳性数和期望阳性数
        observed = y_true_series.groupby(groups).sum()
        expected = y_pred_prob_series.groupby(groups).sum()
        # 计算每组实际阴性数和期望阴性数
        observed_neg = y_true_series.groupby(groups).count() - observed
        expected_neg = y_pred_prob_series.groupby(groups).count() - expected
        # 构建列联表
        contingency_table = np.array([[observed.iloc[i], observed_neg.iloc[i]] for i in range(len(observed))])
        # 卡方检验
        chi2, p_value, dof, expected_table = chi2_contingency(contingency_table)
        return chi2, p_value

    def decision_curve_analysis(self, y_true, y_pred_prob, thresholds=np.linspace(0, 1, 100)):
        """决策曲线分析（DCA）"""
        n = len(y_true)
        tp = np.array([np.sum((y_pred_prob >= t) & (y_true == 1)) for t in thresholds])
        fp = np.array([np.sum((y_pred_prob >= t) & (y_true == 0)) for t in thresholds])
        # 净获益 = (TP - FP×(t/(1-t)))/n
        net_benefit = (tp - fp * (thresholds / (1 - thresholds + 1e-8))) / n
        # 全部干预的净获益
        all_treat = (np.sum(y_true) - (n - np.sum(y_true)) * (thresholds / (1 - thresholds + 1e-8))) / n
        # 全部不干预的净获益
        no_treat = np.zeros_like(thresholds)
        return thresholds, net_benefit, all_treat, no_treat

    def run_full_evaluation(self):
        """完整评估流程：所有队列+所有指标"""
        print("===== 开始模型完整评估 =====")
        # 1. 加载所有队列
        data_dict = self.load_all_queues()
        if not data_dict:
            print("错误：未加载到任何队列数据")
            return None, None
        # 2. 初始化结果表格
        results_df = pd.DataFrame(columns=[
            "队列名称", "样本量", "COPD例数", "AUC", "AUC_95CI", 
            "Brier评分", "HL检验P值", "校准状态"
        ])
        # 3. 逐队列评估
        for queue_code, queue_data in data_dict.items():
            print(f"\n===== 评估{queue_data['name']} =====")
            y_true = queue_data["y"]
            y_pred_prob = queue_data["y_pred_prob"]
            # （1）AUC及95%CI
            auc, auc_lower, auc_upper = self.bootstrap_auc(y_true, y_pred_prob)
            auc_ci = f"{auc_lower:.3f}-{auc_upper:.3f}"
            print(f"AUC：{auc:.4f}（95%CI：{auc_ci}）")
            # （2）Brier评分（越小越好）
            brier = brier_score_loss(y_true, y_pred_prob)
            print(f"Brier评分：{brier:.4f}（＜0.15最优）")
            # （3）HL检验（校准度）
            chi2, hl_p = self.hosmer_lemeshow_test(y_true, y_pred_prob)
            calibration_status = "良好" if hl_p > 0.05 else "偏差"
            print(f"HL检验：χ²={chi2:.2f}，P值={hl_p:.4f}，校准状态：{calibration_status}")
            # （4）DCA（后续可视化）
            thresholds, net_benefit, all_treat, no_treat = self.decision_curve_analysis(y_true, y_pred_prob)
            # （5）保存结果到表格
            results_df.loc[len(results_df)] = [
                queue_data["name"], len(y_true), y_true.sum(),
                round(auc, 4), auc_ci, round(brier, 4), round(hl_p, 4), calibration_status
            ]
        # 4. 保存评估结果
        results_path = f"{self.output_dir}model_performance_all_queues.csv"
        results_df.to_csv(results_path, index=False, encoding="utf-8")
        print(f"\n===== 评估完成，结果保存至：{results_path} =====")
        print("\n核心性能汇总：")
        print(results_df[["队列名称", "AUC", "AUC_95CI", "Brier评分", "校准状态"]].to_string(index=False))
        return results_df, data_dict

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run_full_evaluation()