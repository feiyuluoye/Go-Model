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
	fmt.Println("=== Logarithmic 回归示例 ===")

	// 生成对数数据: y = 3 * ln(x) + 2 + noise
	n := 80
	
	XData := make([]float64, n)
	yData := make([]float64, n)

	rand.Seed(42)
	
	for i := 0; i < n; i++ {
		x := rand.Float64()*9 + 1 // [1, 10] 确保x > 0
		XData[i] = x
		
		// 真实对数关系: y = 3 * ln(x) + 2
		y_true := 3*math.Log(x) + 2
		noise := rand.NormFloat64() * 0.5
		yData[i] = y_true + noise
	}

	X := mat.NewDense(n, 1, XData)
	y := mat.NewVecDense(n, yData)

	fmt.Printf("数据集大小: %d 样本\n", n)
	fmt.Printf("真实对数函数: y = 3 * ln(x) + 2\n")
	fmt.Printf("真实参数: a = 3.0, b = 2.0\n\n")

	// 创建并训练对数回归模型
	model := models.NewLogarithmic()
	
	fmt.Println("训练 Logarithmic 回归模型...")
	err := model.Fit(X, y)
	if err != nil {
		log.Fatalf("训练失败: %v", err)
	}

	// 获取模型参数
	params := model.GetParameters()
	a := params["a"].(float64)
	b := params["b"].(float64)

	fmt.Printf("学到的参数: a = %.4f, b = %.4f\n", a, b)
	fmt.Printf("学到的函数: y = %.4f * ln(x) + %.4f\n", a, b)

	// 计算R²分数
	r2 := model.Score(X, y)
	fmt.Printf("R² 分数: %.4f\n\n", r2)

	// 进行预测
	fmt.Println("预测示例:")
	testX := mat.NewDense(6, 1, []float64{1.0, 2.0, 3.0, 5.0, 7.0, 10.0})
	predictions := model.Predict(testX)
	
	fmt.Printf("%-8s %-12s %-12s %-12s\n", "x", "预测值", "真实值", "绝对误差")
	fmt.Println("----------------------------------------")
	
	for i := 0; i < 6; i++ {
		x := testX.At(i, 0)
		pred := predictions.At(i, 0)
		expected := 3*math.Log(x) + 2
		absError := math.Abs(pred - expected)
		
		fmt.Printf("%-8.1f %-12.4f %-12.4f %-12.4f\n", 
			x, pred, expected, absError)
	}

	fmt.Println("\n=== Logarithmic 示例完成 ===")
}
