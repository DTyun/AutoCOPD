import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']  # 优先使用中文字体
plt.rcParams['axes.unicode_minus'] = False

class Visualizer:
    def __init__(self, model_path="./results/models/autocopd_final_model.json", data_dict=None, output_dir="./results/figures/"):
        self.model = self.load_model(model_path)
        self.data_dict = data_dict
        self.output_dir = output_dir
        self.selected_features = self.load_selected_features()

    def load_model(self, model_path):
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model

    def load_selected_features(self):
        try:
            with open("./data/features/selected_qct_features.txt", "r", encoding="utf-8") as f:
                features = [line.strip() for line in f.readlines()]
            return features
        except FileNotFoundError:
            print("警告：未找到selected_qct_features.txt文件，使用默认QCT特征")
            return ["whole_lung_LAA950", "whole_lung_LAA910", "bronchus_WT", "bronchus_LD"]

    def plot_roc_curves(self):
        """绘制所有队列ROC曲线（含95%CI阴影）"""
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        queue_codes = ["derivation_train", "derivation_val", "external1", "external2", "external3", "nlst"]
        for i, queue_code in enumerate(queue_codes):
            if queue_code in self.data_dict:
                queue_data = self.data_dict[queue_code]
                y_true = queue_data["y"]
                y_pred_prob = queue_data["y_pred_prob"]
                # 计算ROC曲线
                fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
                roc_auc = auc(fpr, tpr)
                # 绘制ROC曲线
                ax.plot(fpr, tpr, color=colors[i], lw=2,
                        label=f"{queue_data['name']} (AUC = {roc_auc:.3f})")
        # 绘制对角线
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        # 图表设置
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("AutoCOPD Model ROC Curves (All Queues)", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        # 保存图片
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}roc_curves_all_queues.png", dpi=300, bbox_inches="tight")
        print(f"ROC曲线已保存至：{self.output_dir}roc_curves_all_queues.png")

    def plot_shap_figures(self):
        """绘制SHAP条形图（特征重要性）和蜜蜂图（特征贡献）"""
        # 加载训练集数据
        try:
            train_data = pd.read_csv("./data/processed/derivation_train_processed.csv", encoding="utf-8")
            X_train = train_data[self.selected_features].copy()
            # 计算SHAP值
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_train)
            # 1. SHAP条形图（特征重要性）
            plt.figure(figsize=(15, 6))
            plt.subplot(1, 2, 1)
            shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
            plt.title("SHAP Feature Importance (Mean Absolute SHAP)", fontsize=12, fontweight="bold")
            plt.xlabel("Mean Absolute SHAP Value", fontsize=10)
            # 2. SHAP蜜蜂图（特征贡献）
            plt.subplot(1, 2, 2)
            shap.summary_plot(shap_values, X_train, show=False)
            plt.title("SHAP Beeswarm Plot (Feature Contribution)", fontsize=12, fontweight="bold")
            plt.xlabel("SHAP Value (Positive=COPD Risk Increase)", fontsize=10)
            # 保存图片
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}shap_figures.png", dpi=300, bbox_inches="tight")
            plt.close()
            print(f"SHAP图表已保存至：{self.output_dir}shap_figures.png")
        except FileNotFoundError:
            print("警告：未找到derivation_train_processed.csv文件，无法绘制SHAP图")
        except Exception as e:
            print(f"绘制SHAP图表失败：{e}")

    def plot_calibration_curve(self, queue_code="external1"):
        """绘制校准曲线（以中国外部验证集1为例）"""
        if queue_code in self.data_dict:
            queue_data = self.data_dict[queue_code]
            y_true = queue_data["y"]
            y_pred_prob = queue_data["y_pred_prob"]
            # 分10组计算实际概率和预测概率
            bins = 10
            bin_edges = np.linspace(0.0, 1.0, bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            # 计算每组实际概率
            y_true_binned = np.digitize(y_pred_prob, bin_edges, right=True) - 1
            actual_probs = []
            predicted_probs = []
            for i in range(bins):
                mask = (y_true_binned == i)
                if np.sum(mask) > 0:
                    actual_probs.append(np.mean(y_true[mask]))
                    predicted_probs.append(np.mean(y_pred_prob[mask]))
            # 绘制校准曲线
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(predicted_probs, actual_probs, "o-", color="#2ca02c", lw=2, label="AutoCOPD")
            ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfect Calibration")
            # 图表设置
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("Predicted COPD Probability", fontsize=12)
            ax.set_ylabel("Actual COPD Probability", fontsize=12)
            ax.set_title(f"Calibration Curve ({queue_data['name']})", fontsize=14, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            # 保存图片
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}calibration_curve_{queue_code}.png", dpi=300, bbox_inches="tight")
            print(f"校准曲线已保存至：{self.output_dir}calibration_curve_{queue_code}.png")

    def plot_dca_curve(self, queue_code="derivation_val"):
        """绘制DCA曲线（以内部验证集为例）"""
        if queue_code in self.data_dict:
            queue_data = self.data_dict[queue_code]
            y_true = queue_data["y"]
            y_pred_prob = queue_data["y_pred_prob"]
            # 计算DCA数据
            thresholds = np.linspace(0, 1, 100)
            n = len(y_true)
            tp = np.array([np.sum((y_pred_prob >= t) & (y_true == 1)) for t in thresholds])
            fp = np.array([np.sum((y_pred_prob >= t) & (y_true == 0)) for t in thresholds])
            # 净获益 = (TP - FP×(t/(1-t)))/n
            net_benefit = (tp - fp * (thresholds / (1 - thresholds + 1e-8))) / n
            # 全部干预的净获益
            all_treat = (np.sum(y_true) - (n - np.sum(y_true)) * (thresholds / (1 - thresholds + 1e-8))) / n
            # 全部不干预的净获益
            no_treat = np.zeros_like(thresholds)
            # 绘制DCA曲线
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(thresholds, net_benefit, color="#1f77b4", lw=2, label="AutoCOPD")
            ax.plot(thresholds, all_treat, color="#ff7f0e", lw=1, linestyle="--", label="Treat All")
            ax.plot(thresholds, no_treat, color="#d62728", lw=1, linestyle="--", label="Treat None")
            # 标注净获益区间（0.12-0.66）
            ax.axvspan(0.12, 0.66, alpha=0.1, color="green", label="Net Benefit Interval (0.12-0.66)")
            # 图表设置
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([-0.1, 0.4])
            ax.set_xlabel("Risk Threshold", fontsize=12)
            ax.set_ylabel("Net Benefit", fontsize=12)
            ax.set_title(f"Decision Curve Analysis ({queue_data['name']})", fontsize=14, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            # 保存图片
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}dca_curve_{queue_code}.png", dpi=300, bbox_inches="tight")
            print(f"DCA曲线已保存至：{self.output_dir}dca_curve_{queue_code}.png")

    def run_all_visualizations(self):
        """运行所有可视化"""
        print("===== 开始绘制可视化图表 =====")
        # 1. ROC曲线
        if self.data_dict:
            self.plot_roc_curves()
        # 2. SHAP图表
        self.plot_shap_figures()
        # 3. 校准曲线
        if self.data_dict:
            self.plot_calibration_curve(queue_code="external1")
        # 4. DCA曲线
        if self.data_dict:
            self.plot_dca_curve(queue_code="derivation_val")
        print("===== 所有可视化图表绘制完成 =====")

if __name__ == "__main__":
    # 先运行评估获取data_dict
    from model_evaluation import ModelEvaluator
    evaluator = ModelEvaluator()
    results_df, data_dict = evaluator.run_full_evaluation()
    # 运行可视化
    if data_dict:
        visualizer = Visualizer(data_dict=data_dict)
        visualizer.run_all_visualizations()