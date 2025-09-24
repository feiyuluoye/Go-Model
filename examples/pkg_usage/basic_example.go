package main

import (
	"fmt"
	"log"

	"github.com/feiyuluoye/Go-Model/pkg/gomodel"
)

func main() {
	fmt.Println("=== Go-Model Package 基础使用示例 ===")

	// 1. 快速训练示例
	fmt.Println("\n1. 快速训练示例")
	quickTrainingExample()

	// 2. 完整API使用示例
	fmt.Println("\n2. 完整API使用示例")
	fullAPIExample()

	// 3. 数据处理示例
	fmt.Println("\n3. 数据处理示例")
	dataProcessingExample()

	// 4. 模型比较示例
	fmt.Println("\n4. 模型比较示例")
	modelComparisonExample()
}

func quickTrainingExample() {
	// 准备简单的线性数据
	features := [][]float64{
		{1.0, 2.0},
		{2.0, 3.0},
		{3.0, 4.0},
		{4.0, 5.0},
		{5.0, 6.0},
	}
	target := []float64{5.0, 8.0, 11.0, 14.0, 17.0} // y = 3*x1 + 2*x2 - 1

	// 快速训练OLS模型
	result, err := gomodel.QuickTrain(features, target, gomodel.OLS)
	if err != nil {
		log.Printf("快速训练失败: %v", err)
		return
	}

	fmt.Printf("算法: %s\n", result.Algorithm)
	fmt.Printf("训练R²: %.4f\n", result.TrainingScore)
	fmt.Printf("RMSE: %.4f\n", result.Metrics["rmse"])

	// 快速预测
	testFeatures := [][]float64{
		{6.0, 7.0},
		{7.0, 8.0},
	}

	predictions, err := gomodel.QuickPredict(features, target, testFeatures, gomodel.OLS)
	if err != nil {
		log.Printf("快速预测失败: %v", err)
		return
	}

	fmt.Printf("预测结果: %.2f, %.2f\n", predictions[0], predictions[1])
}

func fullAPIExample() {
	// 创建客户端
	config := &gomodel.ClientConfig{
		DefaultValidation: &gomodel.ValidationConfig{
			Method:     "kfold",
			KFolds:     5,
			RandomSeed: 42,
		},
		RandomSeed: 42,
		Verbose:    true,
	}
	client := gomodel.NewClient(config)

	// 创建数据工具
	dataUtils := gomodel.NewDataUtils(42)

	// 生成合成数据
	data, err := dataUtils.GenerateSyntheticData(100, 3, 0.1, "linear")
	if err != nil {
		log.Printf("生成数据失败: %v", err)
		return
	}

	// 数据预处理
	normalizedData, err := dataUtils.Normalize(data)
	if err != nil {
		log.Printf("数据标准化失败: %v", err)
		return
	}

	// 分割训练测试集
	trainData, testData, err := dataUtils.SplitTrainTest(normalizedData, 0.2, true)
	if err != nil {
		log.Printf("数据分割失败: %v", err)
		return
	}

	// 配置Ridge回归模型
	modelConfig := &gomodel.ModelConfig{
		Algorithm:    gomodel.Ridge,
		Parameters:   map[string]interface{}{"lambda": 1.0},
		LossFunction: gomodel.R2,
		Validation: &gomodel.ValidationConfig{
			Method:     "kfold",
			KFolds:     5,
			RandomSeed: 42,
		},
	}

	// 训练模型
	result, err := client.Train(trainData, modelConfig)
	if err != nil {
		log.Printf("模型训练失败: %v", err)
		return
	}

	fmt.Printf("算法: %s\n", result.Algorithm)
	fmt.Printf("训练R²: %.4f\n", result.TrainingScore)
	if result.ValidationScore != nil {
		fmt.Printf("验证R²: %.4f\n", *result.ValidationScore)
	}

	// 交叉验证结果
	if result.CrossValidation != nil {
		cv := result.CrossValidation
		fmt.Printf("交叉验证: %.4f ± %.4f (%d折)\n", 
			cv.MeanScore, cv.StdScore, cv.FoldCount)
	}

	// 在测试集上评估
	fmt.Printf("测试集样本数: %d\n", testData.Target.Len())
}

func dataProcessingExample() {
	dataUtils := gomodel.NewDataUtils(42)

	// 创建示例数据
	features := [][]float64{
		{1.0, 10.0, 100.0},
		{2.0, 20.0, 200.0},
		{3.0, 30.0, 300.0},
		{4.0, 40.0, 400.0},
		{5.0, 50.0, 500.0},
	}
	target := []float64{111.0, 222.0, 333.0, 444.0, 555.0}

	data, err := dataUtils.CreateFromArrays(features, target, 
		[]string{"小", "中", "大"}, "总和")
	if err != nil {
		log.Printf("创建数据失败: %v", err)
		return
	}

	// 获取数据摘要
	summary := dataUtils.GetDataSummary(data)
	fmt.Printf("数据形状: %d样本, %d特征\n", summary["samples"], summary["features"])

	// 标准化数据
	normalizedData, err := dataUtils.Normalize(data)
	if err != nil {
		log.Printf("标准化失败: %v", err)
		return
	}

	// 缩放数据
	scaledData, err := dataUtils.Scale(data)
	if err != nil {
		log.Printf("缩放失败: %v", err)
		return
	}

	fmt.Printf("原始数据第一行: %.2f, %.2f, %.2f\n", 
		data.Features.At(0, 0), data.Features.At(0, 1), data.Features.At(0, 2))
	fmt.Printf("标准化后第一行: %.2f, %.2f, %.2f\n", 
		normalizedData.Features.At(0, 0), normalizedData.Features.At(0, 1), normalizedData.Features.At(0, 2))
	fmt.Printf("缩放后第一行: %.2f, %.2f, %.2f\n", 
		scaledData.Features.At(0, 0), scaledData.Features.At(0, 1), scaledData.Features.At(0, 2))
}

func modelComparisonExample() {
	dataUtils := gomodel.NewDataUtils(42)
	client := gomodel.NewClient(nil)

	// 生成测试数据
	data, err := dataUtils.GenerateSyntheticData(80, 2, 0.15, "linear")
	if err != nil {
		log.Printf("生成数据失败: %v", err)
		return
	}

	// 测试多个算法
	algorithms := []gomodel.AlgorithmType{
		gomodel.OLS,
		gomodel.Ridge,
		gomodel.Lasso,
	}

	fmt.Printf("算法性能比较:\n")
	fmt.Printf("%-12s %-10s %-10s %-10s\n", "算法", "R²", "RMSE", "MAE")
	fmt.Printf("%-12s %-10s %-10s %-10s\n", "----", "--", "----", "---")

	for _, alg := range algorithms {
		config := gomodel.GetDefaultConfig(alg)
		config.Validation = &gomodel.ValidationConfig{
			Method:     "kfold",
			KFolds:     5,
			RandomSeed: 42,
		}

		result, err := client.Train(data, config)
		if err != nil {
			log.Printf("训练%s失败: %v", alg, err)
			continue
		}

		validationScore := "N/A"
		if result.ValidationScore != nil {
			validationScore = fmt.Sprintf("%.4f", *result.ValidationScore)
		}

		fmt.Printf("%-12s %-10s %-10.4f %-10.4f\n", 
			alg, validationScore, result.Metrics["rmse"], result.Metrics["r2"])
	}

	// 获取算法信息
	fmt.Printf("\n支持的算法信息:\n")
	allInfo := gomodel.GetAllAlgorithmsInfo()
	for alg, info := range allInfo {
		fmt.Printf("- %s: %s (%s)\n", alg, info["description"], info["type"])
	}
}
