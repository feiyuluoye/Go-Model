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
	fmt.Println("=== Power 回归示例 ===")

	// 生成幂函数数据: y = 2 * x^1.5 + noise
	n := 80
	
	XData := make([]float64, n)
	yData := make([]float64, n)

	rand.Seed(42)
	
	for i := 0; i < n; i++ {
		x := rand.Float64()*4 + 0.5 // [0.5, 4.5] 确保x > 0
		XData[i] = x
		
		// 真实幂函数关系: y = 2 * x^1.5
		y_true := 2 * math.Pow(x, 1.5)
		// 添加相对噪声
		noise := rand.NormFloat64() * y_true * 0.1
		yData[i] = y_true + noise
		
		// 确保y值为正（幂回归要求）
		if yData[i] <= 0 {
			yData[i] = 0.1
		}
	}

	X := mat.NewDense(n, 1, XData)
	y := mat.NewVecDense(n, yData)

	fmt.Printf("数据集大小: %d 样本\n", n)
	fmt.Printf("真实幂函数: y = 2 * x^1.5\n")
	fmt.Printf("真实参数: a = 2.0, b = 1.5\n\n")

	// 创建并训练幂回归模型
	model := models.NewPower()
	
	fmt.Println("训练 Power 回归模型...")
	err := model.Fit(X, y)
	if err != nil {
		log.Fatalf("训练失败: %v", err)
	}

	// 获取模型参数
	params := model.GetParameters()
	a := params["a"].(float64)
	b := params["b"].(float64)

	fmt.Printf("学到的参数: a = %.4f, b = %.4f\n", a, b)
	fmt.Printf("学到的函数: y = %.4f * x^%.4f\n", a, b)

	// 计算R²分数
	r2 := model.Score(X, y)
	fmt.Printf("R² 分数: %.4f\n\n", r2)

	// 进行预测
	fmt.Println("预测示例:")
	testX := mat.NewDense(6, 1, []float64{0.5, 1.0, 1.5, 2.0, 3.0, 4.0})
	predictions := model.Predict(testX)
	
	fmt.Printf("%-8s %-12s %-12s %-12s\n", "x", "预测值", "真实值", "相对误差")
	fmt.Println("----------------------------------------")
	
	for i := 0; i < 6; i++ {
		x := testX.At(i, 0)
		pred := predictions.At(i, 0)
		expected := 2 * math.Pow(x, 1.5)
		relError := math.Abs(pred-expected) / expected * 100
		
		fmt.Printf("%-8.1f %-12.4f %-12.4f %-11.2f%%\n", 
			x, pred, expected, relError)
	}

	fmt.Println("\n=== Power 示例完成 ===")
}
