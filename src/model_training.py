import pandas as pd
import numpy as np
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, processed_dir="./data/processed/", feature_dir="./data/features/", model_dir="./results/models/"):
        self.processed_dir = processed_dir
        self.feature_dir = feature_dir
        self.model_dir = model_dir
        self.target_col = "copd_diagnosis"
        self.selected_features = self.load_selected_features()

    def load_selected_features(self):
        """加载筛选后的Top10 QCT特征"""
        try:
            with open(f"{self.feature_dir}selected_qct_features.txt", "r", encoding="utf-8") as f:
                features = [line.strip() for line in f.readlines()]
            print(f"加载筛选后的特征：{features}（共{len(features)}个）")
            return features
        except FileNotFoundError:
            print("警告：未找到selected_qct_features.txt文件，使用默认QCT特征")
            return ["whole_lung_LAA950", "whole_lung_LAA910", "bronchus_WT", "bronchus_LD"]

    def load_training_data(self):
        """加载训练集和内部验证集"""
        # 训练集
        train_data = pd.read_csv(f"{self.processed_dir}derivation_train_processed.csv", encoding="utf-8")
        X_train = train_data[self.selected_features].copy()
        y_train = train_data[self.target_col].copy()
        # 内部验证集
        val_data = pd.read_csv(f"{self.processed_dir}derivation_val_processed.csv", encoding="utf-8")
        X_val = val_data[self.selected_features].copy()
        y_val = val_data[self.target_col].copy()
        print(f"训练集：{X_train.shape[0]}行 × {X_train.shape[1]}列，COPD比例：{y_train.mean():.3f}")
        print(f"内部验证集：{X_val.shape[0]}行 × {X_val.shape[1]}列，COPD比例：{y_val.mean():.3f}")
        return X_train, y_train, X_val, y_val

    def objective(self, params):
        """贝叶斯优化目标函数：10折交叉验证对数损失"""
        # 转换超参数类型
        params["max_depth"] = int(params["max_depth"])
        params["n_estimators"] = int(params["n_estimators"])
        params["subsample"] = float(params["subsample"])
        # 10折交叉验证
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        fold_losses = []
        for train_idx, val_idx in skf.split(self.X_train, self.y_train):
            X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            # 训练模型
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                **params
            )
            model.fit(X_fold_train, y_fold_train, verbose=False)
            # 计算对数损失
            y_pred_prob = model.predict_proba(X_fold_val)[:, 1]
            fold_loss = log_loss(y_fold_val, y_pred_prob)
            fold_losses.append(fold_loss)
        # 返回平均损失
        avg_loss = np.mean(fold_losses)
        print(f"超参数：{params}，10折平均对数损失：{avg_loss:.4f}")
        return {"loss": avg_loss, "status": STATUS_OK}

    def bayesian_optimization(self, max_evals=100):
        """贝叶斯优化调参"""
        # 超参数搜索空间（基于论文范围）
        space = {
            "max_depth": hp.quniform("max_depth", 3, 6, 1),  # 论文最优4
            "eta": hp.uniform("eta", 0.01, 0.1),  # 论文最优0.036
            "n_estimators": hp.quniform("n_estimators", 100, 300, 10),  # 论文最优173
            "subsample": hp.uniform("subsample", 0.7, 1.0),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.7, 1.0),
            "min_child_weight": hp.quniform("min_child_weight", 1, 5, 1),
            "gamma": hp.uniform("gamma", 0, 1)
        }
        # 执行优化
        trials = Trials()
        best = fmin(
            fn=self.objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(42)
        )
        # 转换最优超参数类型
        best_params = {
            "max_depth": int(best["max_depth"]),
            "eta": best["eta"],
            "n_estimators": int(best["n_estimators"]),
            "subsample": best["subsample"],
            "colsample_bytree": best["colsample_bytree"],
            "min_child_weight": int(best["min_child_weight"]),
            "gamma": best["gamma"]
        }
        print(f"\n最优超参数：{best_params}")
        return best_params

    def train_final_model(self, best_params):
        """用最优超参数训练最终模型"""
        # 训练最终模型（训练集+10折交叉验证）
        final_model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            **best_params
        )
        final_model.fit(self.X_train, self.y_train, verbose=False)
        # 在内部验证集评估
        y_val_pred_prob = final_model.predict_proba(self.X_val)[:, 1]
        val_auc = roc_auc_score(self.y_val, y_val_pred_prob)
        val_logloss = log_loss(self.y_val, y_val_pred_prob)
        print(f"\n最终模型内部验证集性能：AUC={val_auc:.4f}，对数损失={val_logloss:.4f}")
        print(f"目标：AUC≥0.860（论文内部验证集AUC=0.860）")
        # 保存模型
        model_path = f"{self.model_dir}autocopd_final_model.json"
        final_model.save_model(model_path)
        print(f"最终模型已保存至：{model_path}")
        # 保存最优超参数
        params_df = pd.DataFrame([best_params])
        params_df.to_csv(f"{self.model_dir}best_hyperparameters.csv", index=False, encoding="utf-8")
        return final_model, val_auc

    def run(self):
        print("===== 开始模型训练流程 =====")
        try:
            # 1. 加载数据
            self.X_train, self.y_train, self.X_val, self.y_val = self.load_training_data()
            # 2. 贝叶斯优化调参
            print("\n===== 贝叶斯优化调参 =====")
            best_params = self.bayesian_optimization(max_evals=100)
            # 3. 训练最终模型
            print("\n===== 训练最终模型 =====")
            final_model, val_auc = self.train_final_model(best_params)
            print("===== 模型训练流程完成 =====")
            return final_model, val_auc
        except Exception as e:
            print(f"模型训练过程中出错：{e}")
            return None, 0

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()