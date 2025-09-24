package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/feiyuluoye/Go-Model/internal/models"
	"gonum.org/v1/gonum/mat"
)

func main() {
	fmt.Println("=== Exponential 回归示例 ===")

	// 生成指数数据: y = 2 * exp(0.5 * x) + noise
	n := 80
	
	XData := make([]float64, n)
	yData := make([]float64, n)

	rand.Seed(42)
	
	for i := 0; i < n; i++ {
		x := rand.Float64()*4 - 1 // [-1, 3] 避免指数增长过快
		XData[i] = x
		
		// 真实指数关系: y = 2 * exp(0.5 * x)
		y_true := 2 * math.Exp(0.5*x)
		// 添加相对噪声（避免绝对噪声在大值时影响过大）
		noise := rand.NormFloat64() * y_true * 0.1
		yData[i] = y_true + noise
		
		// 确保y值为正（指数回归要求）
		if yData[i] <= 0 {
			yData[i] = 0.1
		}
	}

	X := mat.NewDense(n, 1, XData)
	y := mat.NewVecDense(n, yData)

	fmt.Printf("数据集大小: %d 样本\n", n)
	fmt.Printf("真实指数函数: y = 2 * exp(0.5 * x)\n")
	fmt.Printf("真实参数: a = 2.0, b = 0.5\n\n")

	// 创建并训练指数回归模型
	model := models.NewExponential()
	
	fmt.Println("训练 Exponential 回归模型...")
	err := model.Fit(X, y)
	if err != nil {
		log.Fatalf("训练失败: %v", err)
	}

	// 获取模型参数
	params := model.GetParameters()
	a := params["a"].(float64)
	b := params["b"].(float64)

	fmt.Printf("学到的参数: a = %.4f, b = %.4f\n", a, b)
	fmt.Printf("学到的函数: y = %.4f * exp(%.4f * x)\n", a, b)

	// 计算R²分数
	r2 := model.Score(X, y)
	fmt.Printf("R² 分数: %.4f\n\n", r2)

	// 进行预测
	fmt.Println("预测示例:")
	testX := mat.NewDense(6, 1, []float64{-1.0, -0.5, 0.0, 0.5, 1.0, 2.0})
	predictions := model.Predict(testX)
	
	fmt.Printf("%-8s %-12s %-12s %-12s\n", "x", "预测值", "真实值", "相对误差")
	fmt.Println("----------------------------------------")
	
	for i := 0; i < 6; i++ {
		x := testX.At(i, 0)
		pred := predictions.At(i, 0)
		expected := 2 * math.Exp(0.5*x)
		relError := math.Abs(pred-expected) / expected * 100
		
		fmt.Printf("%-8.1f %-12.4f %-12.4f %-11.2f%%\n", 
			x, pred, expected, relError)
	}

	fmt.Println("\n=== Exponential 示例完成 ===")
}
