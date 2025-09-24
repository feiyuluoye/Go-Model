package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/feiyuluoye/Go-Model/internal/models"
	"gonum.org/v1/gonum/mat"
)

func main() {
	fmt.Println("=== PLS 回归示例 ===")

	// 生成高维数据，其中特征之间有相关性
	n := 80
	p := 20 // 高维特征

	XData := make([]float64, n*p)
	yData := make([]float64, n)

	rand.Seed(42)
	
	// 创建潜在变量
	for i := 0; i < n; i++ {
		// 两个潜在因子
		factor1 := rand.NormFloat64()
		factor2 := rand.NormFloat64()
		
		// y主要由这两个因子决定
		yData[i] = 2*factor1 + 1.5*factor2 + rand.NormFloat64()*0.3
		
		// X的特征是这些因子的线性组合加噪声
		for j := 0; j < p; j++ {
			if j < 8 {
				// 前8个特征主要与factor1相关
				XData[i*p+j] = factor1 + rand.NormFloat64()*0.5
			} else if j < 16 {
				// 中间8个特征主要与factor2相关
				XData[i*p+j] = factor2 + rand.NormFloat64()*0.5
			} else {
				// 最后4个特征是噪声
				XData[i*p+j] = rand.NormFloat64()
			}
		}
	}

	X := mat.NewDense(n, p, XData)
	y := mat.NewVecDense(n, yData)

	fmt.Printf("数据集大小: %d 样本, %d 特征\n", n, p)
	fmt.Printf("数据特点: 高维特征，存在潜在结构\n\n")

	// 比较不同的成分数量
	components := []int{1, 2, 5, 10}

	for _, numComp := range components {
		fmt.Printf("--- 成分数量 = %d ---\n", numComp)
		
		model := models.NewPLS(numComp)
		
		err := model.Fit(X, y)
		if err != nil {
			log.Fatalf("训练失败: %v", err)
		}

		r2 := model.Score(X, y)
		fmt.Printf("R² 分数: %.4f\n", r2)

		params := model.GetParameters()
		fmt.Printf("模型参数已保存，包含 %d 个成分\n\n", numComp)
	}

	// 使用最佳模型进行预测
	bestModel := models.NewPLS(2) // 使用2个成分
	bestModel.Fit(X, y)

	fmt.Println("预测示例 (使用 2 个成分):")
	
	// 创建测试数据
	testData := make([]float64, 3*p)
	rand.Seed(123)
	
	for i := 0; i < 3; i++ {
		factor1 := float64(i-1) // -1, 0, 1
		factor2 := float64(1-i) // 1, 0, -1
		
		for j := 0; j < p; j++ {
			if j < 8 {
				testData[i*p+j] = factor1 + rand.NormFloat64()*0.1
			} else if j < 16 {
				testData[i*p+j] = factor2 + rand.NormFloat64()*0.1
			} else {
				testData[i*p+j] = rand.NormFloat64()*0.1
			}
		}
	}
	
	testX := mat.NewDense(3, p, testData)
	predictions := bestModel.Predict(testX)
	
	for i := 0; i < 3; i++ {
		factor1 := float64(i-1)
		factor2 := float64(1-i)
		expected := 2*factor1 + 1.5*factor2
		pred := predictions.At(i, 0)
		
		fmt.Printf("测试 %d: 潜在因子 [%.1f, %.1f] -> 预测: %.4f, 期望: %.4f\n", 
			i+1, factor1, factor2, pred, expected)
	}

	fmt.Println("\n=== PLS 示例完成 ===")
}
