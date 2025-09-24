package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"strings"

	"github.com/feiyuluoye/Go-Model/internal/models"
	"gonum.org/v1/gonum/mat"
)

func main() {
	fmt.Println("=== Logistic 回归示例 ===")

	// 生成二分类数据
	n := 200
	p := 2

	XData := make([]float64, n*p)
	yData := make([]float64, n)

	rand.Seed(42)
	
	for i := 0; i < n; i++ {
		x1 := rand.Float64()*6 - 3 // [-3, 3]
		x2 := rand.Float64()*6 - 3 // [-3, 3]
		
		XData[i*p] = x1
		XData[i*p+1] = x2
		
		// 决策边界: 2*x1 + x2 - 1 > 0 则为类别1
		logit := 2*x1 + x2 - 1
		prob := 1.0 / (1.0 + math.Exp(-logit))
		
		// 根据概率生成标签
		if rand.Float64() < prob {
			yData[i] = 1.0
		} else {
			yData[i] = 0.0
		}
	}

	X := mat.NewDense(n, p, XData)
	y := mat.NewVecDense(n, yData)

	fmt.Printf("数据集大小: %d 样本, %d 特征\n", n, p)
	fmt.Printf("真实决策边界: 2*x1 + x2 - 1 = 0\n")
	
	// 计算类别分布
	class0, class1 := 0, 0
	for i := 0; i < n; i++ {
		if yData[i] == 0 {
			class0++
		} else {
			class1++
		}
	}
	fmt.Printf("类别分布: 类别0: %d, 类别1: %d\n\n", class0, class1)

	// 创建并训练 Logistic 模型
	model := models.NewLogistic()
	
	fmt.Println("训练 Logistic 回归模型...")
	err := model.Fit(X, y)
	if err != nil {
		log.Fatalf("训练失败: %v", err)
	}

	// 获取模型参数
	params := model.GetParameters()
	coeffs := params["coefficients"].([]float64)
	intercept := params["intercept"].(float64)

	fmt.Printf("学到的系数: [%.4f, %.4f]\n", coeffs[0], coeffs[1])
	fmt.Printf("学到的截距: %.4f\n", intercept)
	fmt.Printf("学到的决策边界: %.4f*x1 + %.4f*x2 + %.4f = 0\n", 
		coeffs[0], coeffs[1], intercept)

	// 计算准确率
	accuracy := model.Score(X, y)
	fmt.Printf("训练集准确率: %.4f\n\n", accuracy)

	// 进行预测
	fmt.Println("预测示例:")
	testX := mat.NewDense(5, 2, []float64{
		2.0, 1.0,   // 2*2 + 1 - 1 = 4 > 0, 应该是类别1
		-1.0, -1.0, // 2*(-1) + (-1) - 1 = -4 < 0, 应该是类别0
		0.0, 1.0,   // 2*0 + 1 - 1 = 0, 边界上
		1.0, -1.0,  // 2*1 + (-1) - 1 = 0, 边界上
		-2.0, 2.0,  // 2*(-2) + 2 - 1 = -3 < 0, 应该是类别0
	})

	probabilities := model.Predict(testX)
	
	fmt.Printf("%-15s %-15s %-15s %-15s\n", "输入", "概率", "预测类别", "真实期望")
	fmt.Println(strings.Repeat("-", 60))
	
	for i := 0; i < 5; i++ {
		x1, x2 := testX.At(i, 0), testX.At(i, 1)
		prob := probabilities.At(i, 0)
		predClass := 0
		if prob >= 0.5 {
			predClass = 1
		}
		
		// 计算真实期望
		trueLogit := 2*x1 + x2 - 1
		expectedClass := 0
		if trueLogit > 0 {
			expectedClass = 1
		}
		
		fmt.Printf("[%.1f, %.1f]     %.4f          %d               %d\n", 
			x1, x2, prob, predClass, expectedClass)
	}

	fmt.Println("\n=== Logistic 示例完成 ===")
}
