package main

import (
	"fmt"
	"log"

	"github.com/feiyuluoye/Go-Model/pkg/gomodel"
)

func main() {
	fmt.Println("=== Go-Model Package 高级使用示例 ===")

	// 1. 模型管理示例
	fmt.Println("\n1. 模型管理示例")
	modelManagementExample()

	// 2. 交叉验证示例
	fmt.Println("\n2. 交叉验证示例")
	crossValidationExample()

	// 3. 非线性模型示例
	fmt.Println("\n3. 非线性模型示例")
	nonlinearModelExample()

	// 4. 分类模型示例
	fmt.Println("\n4. 分类模型示例")
	classificationExample()
}

func modelManagementExample() {
	manager := gomodel.NewModelManager()
	dataUtils := gomodel.NewDataUtils(42)

	// 生成训练数据
	trainData, err := dataUtils.GenerateSyntheticData(100, 3, 0.1, "linear")
	if err != nil {
		log.Printf("生成训练数据失败: %v", err)
		return
	}

	// 训练多个模型
	algorithms := []gomodel.AlgorithmType{
		gomodel.OLS,
		gomodel.Ridge,
		gomodel.Lasso,
	}

	modelIDs := make([]string, 0)

	for _, alg := range algorithms {
		config := gomodel.GetDefaultConfig(alg)
		
		trainedModel, err := manager.TrainModel(config, trainData)
		if err != nil {
			log.Printf("训练%s模型失败: %v", alg, err)
			continue
		}

		modelIDs = append(modelIDs, trainedModel.ID)
		fmt.Printf("训练完成 - %s: R² = %.4f\n", alg, trainedModel.Performance["training_score"])
	}

	// 获取模型列表
	modelList := manager.GetModelList()
	fmt.Printf("\n已训练模型数量: %d\n", len(modelList))

	// 比较模型性能
	if len(modelIDs) > 1 {
		comparison, err := manager.CompareModels(modelIDs, "training_score")
		if err != nil {
			log.Printf("模型比较失败: %v", err)
		} else {
			fmt.Printf("\n模型性能比较 (training_score):\n")
			for modelID, score := range comparison {
				fmt.Printf("模型 %s: %.4f\n", modelID[:8], score)
			}
		}
	}

	// 生成测试数据并预测
	testData, err := dataUtils.GenerateSyntheticData(20, 3, 0.05, "linear")
	if err != nil {
		log.Printf("生成测试数据失败: %v", err)
		return
	}

	if len(modelIDs) > 0 {
		// 准备测试特征
		testFeatures := gomodel.MatrixToArrays(testData.Features)
		
		predictions, err := manager.PredictWithModel(modelIDs[0], testFeatures)
		if err != nil {
			log.Printf("预测失败: %v", err)
		} else {
			fmt.Printf("\n预测结果前3个: %.2f, %.2f, %.2f\n", 
				predictions.Predictions[0], predictions.Predictions[1], predictions.Predictions[2])
		}

		// 在测试数据上评估模型
		metrics, err := manager.EvaluateModelOnTestData(modelIDs[0], testData)
		if err != nil {
			log.Printf("测试评估失败: %v", err)
		} else {
			fmt.Printf("测试集评估 - R²: %.4f, RMSE: %.4f\n", 
				metrics["r2_score"], metrics["rmse"])
		}
	}
}

func crossValidationExample() {
	manager := gomodel.NewModelManager()
	dataUtils := gomodel.NewDataUtils(42)

	// 生成数据
	data, err := dataUtils.GenerateSyntheticData(150, 4, 0.2, "linear")
	if err != nil {
		log.Printf("生成数据失败: %v", err)
		return
	}

	// 标准化数据
	normalizedData, err := dataUtils.Normalize(data)
	if err != nil {
		log.Printf("数据标准化失败: %v", err)
		return
	}

	// 测试不同的Ridge参数
	lambdaValues := []float64{0.1, 1.0, 10.0, 100.0}
	
	fmt.Printf("Ridge回归参数调优 (5折交叉验证):\n")
	fmt.Printf("%-10s %-12s %-12s\n", "Lambda", "Mean Score", "Std Score")
	fmt.Printf("%-10s %-12s %-12s\n", "------", "----------", "---------")

	bestScore := -1.0
	bestLambda := 0.0

	for _, lambda := range lambdaValues {
		config := &gomodel.ModelConfig{
			Algorithm:    gomodel.Ridge,
			Parameters:   map[string]interface{}{"lambda": lambda},
			LossFunction: gomodel.R2,
		}

		cvResult, err := manager.CrossValidateModel(config, normalizedData, 5)
		if err != nil {
			log.Printf("交叉验证失败 (lambda=%.1f): %v", lambda, err)
			continue
		}

		fmt.Printf("%-10.1f %-12.4f %-12.4f\n", 
			lambda, cvResult.MeanScore, cvResult.StdScore)

		if cvResult.MeanScore > bestScore {
			bestScore = cvResult.MeanScore
			bestLambda = lambda
		}
	}

	fmt.Printf("\n最佳参数: lambda = %.1f, 分数 = %.4f\n", bestLambda, bestScore)
}

func nonlinearModelExample() {
	dataUtils := gomodel.NewDataUtils(42)
	client := gomodel.NewClient(nil)

	// 生成多项式数据
	polyData, err := dataUtils.GenerateSyntheticData(80, 1, 0.1, "polynomial")
	if err != nil {
		log.Printf("生成多项式数据失败: %v", err)
		return
	}

	// 测试不同度数的多项式
	degrees := []int{1, 2, 3, 4}
	
	fmt.Printf("多项式回归度数比较:\n")
	fmt.Printf("%-6s %-10s %-10s\n", "度数", "训练R²", "RMSE")
	fmt.Printf("%-6s %-10s %-10s\n", "----", "------", "----")

	for _, degree := range degrees {
		config := &gomodel.ModelConfig{
			Algorithm:    gomodel.Polynomial,
			Parameters:   map[string]interface{}{"degree": degree},
			LossFunction: gomodel.R2,
		}

		result, err := client.Train(polyData, config)
		if err != nil {
			log.Printf("训练多项式模型失败 (度数=%d): %v", degree, err)
			continue
		}

		fmt.Printf("%-6d %-10.4f %-10.4f\n", 
			degree, result.TrainingScore, result.Metrics["rmse"])
	}

	// 测试其他非线性模型
	fmt.Printf("\n非线性模型比较:\n")
	fmt.Printf("%-12s %-10s %-10s\n", "模型", "训练R²", "RMSE")
	fmt.Printf("%-12s %-10s %-10s\n", "----", "------", "----")

	nonlinearAlgorithms := []gomodel.AlgorithmType{
		gomodel.Exponential,
		gomodel.Logarithmic,
		gomodel.Power,
	}

	for _, alg := range nonlinearAlgorithms {
		// 为不同模型生成适合的数据
		var testData *gomodel.TrainingData
		switch alg {
		case gomodel.Exponential:
			testData, _ = dataUtils.GenerateSyntheticData(60, 1, 0.1, "linear")
		case gomodel.Logarithmic:
			testData, _ = dataUtils.GenerateSyntheticData(60, 1, 0.1, "linear")
		case gomodel.Power:
			testData, _ = dataUtils.GenerateSyntheticData(60, 1, 0.1, "linear")
		}

		if testData == nil {
			continue
		}

		config := gomodel.GetDefaultConfig(alg)
		result, err := client.Train(testData, config)
		if err != nil {
			log.Printf("训练%s模型失败: %v", alg, err)
			continue
		}

		fmt.Printf("%-12s %-10.4f %-10.4f\n", 
			alg, result.TrainingScore, result.Metrics["rmse"])
	}
}

func classificationExample() {
	dataUtils := gomodel.NewDataUtils(42)
	client := gomodel.NewClient(nil)

	// 生成分类数据
	classData, err := dataUtils.GenerateSyntheticData(200, 2, 0.1, "classification")
	if err != nil {
		log.Printf("生成分类数据失败: %v", err)
		return
	}

	// 标准化特征
	normalizedData, err := dataUtils.Normalize(classData)
	if err != nil {
		log.Printf("数据标准化失败: %v", err)
		return
	}

	// 分割训练测试集
	trainData, testData, err := dataUtils.SplitTrainTest(normalizedData, 0.3, true)
	if err != nil {
		log.Printf("数据分割失败: %v", err)
		return
	}

	// 配置逻辑回归
	config := &gomodel.ModelConfig{
		Algorithm: gomodel.Logistic,
		Parameters: map[string]interface{}{
			"learning_rate":  0.01,
			"max_iterations": 1000,
			"tolerance":      1e-6,
		},
		LossFunction: gomodel.Accuracy,
		Validation: &gomodel.ValidationConfig{
			Method:     "kfold",
			KFolds:     5,
			RandomSeed: 42,
		},
	}

	// 训练模型
	result, err := client.Train(trainData, config)
	if err != nil {
		log.Printf("逻辑回归训练失败: %v", err)
		return
	}

	fmt.Printf("逻辑回归结果:\n")
	fmt.Printf("训练准确率: %.4f\n", result.TrainingScore)
	if result.ValidationScore != nil {
		fmt.Printf("验证准确率: %.4f\n", *result.ValidationScore)
	}

	// 交叉验证结果
	if result.CrossValidation != nil {
		cv := result.CrossValidation
		fmt.Printf("交叉验证准确率: %.4f ± %.4f\n", cv.MeanScore, cv.StdScore)
	}

	fmt.Printf("训练集大小: %d, 测试集大小: %d\n", 
		trainData.Target.Len(), testData.Target.Len())

	// 显示数据摘要
	summary := dataUtils.GetDataSummary(classData)
	fmt.Printf("数据摘要: %d样本, %d特征\n", summary["samples"], summary["features"])
}
