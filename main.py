def main():
    print("="*50)
    print("AutoCOPD模型Python复刻全流程启动")
    print("="*50)

    # 1. 数据预处理
    print("\n【步骤1/5】数据预处理...")
    try:
        from src.data_preprocess import DataPreprocessor
        preprocessor = DataPreprocessor()
        preprocessor.run()
        print("✅ 数据预处理完成")
    except Exception as e:
        print(f"❌ 数据预处理失败：{e}")

    # 2. 特征筛选
    print("\n【步骤2/5】特征筛选...")
    try:
        from src.feature_selection import FeatureSelector
        selector = FeatureSelector()
        selector.run()
        print("✅ 特征筛选完成")
    except Exception as e:
        print(f"❌ 特征筛选失败：{e}")

    # 3. 模型训练
    print("\n【步骤3/5】模型训练...")
    try:
        from src.model_training import ModelTrainer
        trainer = ModelTrainer()
        final_model, val_auc = trainer.run()
        print(f"✅ 模型训练完成，内部验证集AUC={val_auc:.4f}")
    except Exception as e:
        print(f"❌ 模型训练失败：{e}")

    # 4. 性能评估
    print("\n【步骤4/5】模型评估...")
    try:
        from src.model_evaluation import ModelEvaluator
        evaluator = ModelEvaluator()
        results_df, data_dict = evaluator.run_full_evaluation()
        print("✅ 模型评估完成")
    except Exception as e:
        print(f"❌ 模型评估失败：{e}")
        data_dict = None

    # 5. 可视化
    print("\n【步骤5/5】绘制可视化图表...")
    try:
        from src.visualization import Visualizer
        if data_dict:
            visualizer = Visualizer(data_dict=data_dict)
            visualizer.run_all_visualizations()
            print("✅ 可视化图表绘制完成")
        else:
            print("⚠️  未获取到数据字典，跳过可视化步骤")
    except Exception as e:
        print(f"❌ 可视化失败：{e}")

    # 6. 亚组分析（可选）
    print("\n【步骤6/6】亚组分析（可选）...")
    try:
        from src.subgroup_analysis import SubgroupAnalyzer
        analyzer = SubgroupAnalyzer()
        analyzer.run()
        print("✅ 亚组分析完成")
    except Exception as e:
        print(f"❌ 亚组分析失败：{e}")

    print("\n" + "="*50)
    print("AutoCOPD模型Python复刻全流程完成！")
    print(f"结果文件保存至：./results/")
    print("="*50)

if __name__ == "__main__":
    main()