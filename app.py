from flask import Flask, request, jsonify, render_template, send_file
import xgboost as xgb
import pandas as pd
import numpy as np
import io
import os

app = Flask(__name__)

# 加载模型
def load_model():
    model = xgb.XGBClassifier()
    model_path = "./results/models/autocopd_final_model.json"
    if os.path.exists(model_path):
        model.load_model(model_path)
        return model
    else:
        return None

# 加载特征名
def load_selected_features():
    features_path = "./data/features/selected_qct_features.txt"
    if os.path.exists(features_path):
        with open(features_path, "r", encoding="utf-8") as f:
            selected_features = [line.strip() for line in f.readlines()]
        return selected_features
    else:
        # 默认特征
        return ["whole_lung_LAA950", "bronchus_LD", "whole_lung_LAA910"]



@app.route("/")
def index():
    current_model = load_model()
    current_features = load_selected_features()
    return render_template("index.html", features=current_features, model_loaded=current_model is not None)

@app.route("/predict", methods=["POST"])
def predict():
    current_model = load_model()
    current_features = load_selected_features()
    
    if current_model is None:
        return jsonify({"error": "模型加载失败，请先运行模型训练流程"}), 500
    
    data = request.json
    try:
        # 转换为DataFrame
        df = pd.DataFrame([data])[current_features]
        # 预测
        prob = current_model.predict_proba(df)[0, 1]
        diagnosis = "COPD高风险" if prob >= 0.352 else "COPD低风险"  # 论文阈值0.352
        return jsonify({
            "predicted_probability": round(float(prob), 4),
            "diagnosis": diagnosis
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    current_model = load_model()
    current_features = load_selected_features()
    
    if current_model is None:
        return jsonify({"error": "模型加载失败，请先运行模型训练流程"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "请上传CSV文件"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "文件名不能为空"}), 400
    
    try:
        # 读取CSV文件
        df = pd.read_csv(file)
        # 检查特征列
        if not all(feature in df.columns for feature in current_features):
            missing_features = [feature for feature in current_features if feature not in df.columns]
            return jsonify({"error": f"CSV文件缺少必需特征列: {missing_features}"}), 400
        
        # 预测
        X = df[current_features]
        probs = current_model.predict_proba(X)[:, 1]
        diagnoses = ["COPD高风险" if prob >= 0.352 else "COPD低风险" for prob in probs]
        
        # 添加预测结果到DataFrame
        df['predicted_probability'] = probs.round(4)
        df['diagnosis'] = diagnoses
        
        # 生成CSV响应
        output = io.StringIO()
        df.to_csv(output, index=False, encoding='utf-8-sig')
        output.seek(0)
        
        # 创建临时文件
        temp_path = "./batch_predict_results.csv"
        with open(temp_path, 'w', encoding='utf-8-sig') as f:
            f.write(output.getvalue())
        
        return send_file(temp_path, as_attachment=True, download_name="batch_predict_results.csv", mimetype="text/csv")
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/custom_analysis", methods=["POST"])
def custom_analysis():
    current_model = load_model()
    current_features = load_selected_features()
    
    if current_model is None:
        return jsonify({"error": "模型加载失败，请先运行模型训练流程"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "请上传CSV文件"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "文件名不能为空"}), 400
    
    try:
        # 读取CSV文件
        df = pd.read_csv(file)
        # 检查特征列
        if not all(feature in df.columns for feature in current_features):
            missing_features = [feature for feature in current_features if feature not in df.columns]
            return jsonify({"error": f"CSV文件缺少必需特征列: {missing_features}"}), 400
        
        # 预测
        X = df[current_features]
        probs = current_model.predict_proba(X)[:, 1]
        diagnoses = ["COPD高风险" if prob >= 0.352 else "COPD低风险" for prob in probs]
        
        # 添加预测结果到DataFrame
        df['predicted_probability'] = probs.round(4)
        df['diagnosis'] = diagnoses
        
        # 计算统计信息
        total_samples = len(df)
        high_risk_count = sum(1 for d in diagnoses if d == "COPD高风险")
        low_risk_count = sum(1 for d in diagnoses if d == "COPD低风险")
        high_risk_rate = (high_risk_count / total_samples) * 100 if total_samples > 0 else 0
        
        # 生成分析结果
        analysis_result = {
            "total_samples": total_samples,
            "high_risk_count": high_risk_count,
            "low_risk_count": low_risk_count,
            "high_risk_rate": round(high_risk_rate, 2),
            "sample_data": df.head(10).to_dict('records') if total_samples > 0 else []
        }
        
        return jsonify(analysis_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
