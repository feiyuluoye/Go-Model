package main

import (
	"fmt"
	"github.com/feiyuluoye/Go-Model/internal/regression/linear"
	"log"
)

func main() {
	fmt.Println("OLS回归算法示例")
	fmt.Println("==================")

	// 创建示例数据（避免完全线性相关的特征）
	X := [][]float64{
		{1.0, 2.0},
		{2.0, 1.0},
		{3.0, 4.0},
		{4.0, 3.0},
		{5.0, 6.0},
	}
	y := []float64{3.0, 4.0, 7.0, 8.0, 11.0}

	fmt.Printf("特征矩阵 X: %v\n", X)
	fmt.Printf("目标变量 y: %v\n", y)

	// 创建OLS回归器
	ols := linear.NewOLS(true)

	// 训练模型
	fmt.Println("\n训练OLS模型...")
	err := ols.Fit(X, y)
	if err != nil {
		log.Fatalf("训练失败: %v", err)
	}
	fmt.Println("训练完成!")

	// 显示模型参数
	fmt.Printf("\n模型参数:\n")
	fmt.Printf("截距 (Intercept): %.4f\n", ols.GetIntercept())
	fmt.Printf("系数 (Coefficients): %v\n", ols.GetCoefficients())

	// 进行预测
	fmt.Println("\n进行预测...")
	testX := [][]float64{
		{6.0, 7.0},
		{7.0, 8.0},
	}
	predictions, err := ols.Predict(testX)
	if err != nil {
		log.Fatalf("预测失败: %v", err)
	}

	fmt.Printf("测试数据: %v\n", testX)
	fmt.Printf("预测结果: %v\n", predictions)

	// 计算模型评分
	fmt.Println("\n评估模型...")
	score, err := ols.Score(X, y)
	if err != nil {
		log.Fatalf("评分失败: %v", err)
	}
	fmt.Printf("R²评分: %.4f\n", score)

	// 验证预测结果
	fmt.Println("\n验证预测:")
	expected1 := 13.0 // 6*1 + 7*1 + 1 = 13
	expected2 := 15.0 // 7*1 + 8*1 + 1 = 15
	fmt.Printf("预测值1: %.2f, 期望值: %.2f, 误差: %.2f\n",
		predictions[0], expected1, predictions[0]-expected1)
	fmt.Printf("预测值2: %.2f, 期望值: %.2f, 误差: %.2f\n",
		predictions[1], expected2, predictions[1]-expected2)

	fmt.Println("\n示例完成!")
}
