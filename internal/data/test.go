package data

import (
	"log"
	"testing"
)

func TestDataPipeline(t *testing.T) {
	// 加载CSV数据
	dataset, err := LoadCSV("data.csv", true, "target")
	if err != nil {
		log.Fatalf("加载数据失败: %v", err)
	}

	// 数据标准化
	scaler := NewStandardScaler()
	scaledData, err := scaler.FitTransform(dataset)
	if err != nil {
		log.Fatalf("标准化数据失败: %v", err)
	}

	// 添加多项式特征
	polyData, err := AddPolynomialFeatures(scaledData, 2)
	if err != nil {
		log.Fatalf("添加多项式特征失败: %v", err)
	}

	// 分割训练集和测试集
	trainData, testData, err := TrainTestSplit(polyData, 0.2)
	if err != nil {
		log.Fatalf("分割数据集失败: %v", err)
	}

	// 验证分割结果
	if trainData == nil || testData == nil {
		t.Error("训练集或测试集为空")
	}

	// 验证数据维度
	if trainData.NumSamples() == 0 || testData.NumSamples() == 0 {
		t.Error("训练集或测试集样本数为0")
	}

	// 现在可以使用trainData训练模型，使用testData评估模型
	t.Logf("数据管道测试完成: 训练集样本数=%d, 测试集样本数=%d",
		trainData.NumSamples(), testData.NumSamples())
}
