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
	fmt.Println("=== Polynomial 回归示例 ===")

	// 生成多项式数据: y = 2 + 3*x - 0.5*x^2 + 0.1*x^3 + noise
	n := 100
	
	XData := make([]float64, n)
	yData := make([]float64, n)

	rand.Seed(42)
	
	for i := 0; i < n; i++ {
		x := rand.Float64()*6 - 3 // [-3, 3]
		XData[i] = x
		
		// 真实多项式: y = 2 + 3*x - 0.5*x^2 + 0.1*x^3
		y_true := 2 + 3*x - 0.5*x*x + 0.1*x*x*x
		noise := rand.NormFloat64() * 0.5
		yData[i] = y_true + noise
	}

	X := mat.NewDense(n, 1, XData)
	y := mat.NewVecDense(n, yData)

	fmt.Printf("数据集大小: %d 样本\n", n)
	fmt.Printf("真实多项式: y = 2 + 3*x - 0.5*x² + 0.1*x³\n\n")

	// 比较不同的多项式度数
	degrees := []int{1, 2, 3, 5}

	for _, degree := range degrees {
		fmt.Printf("--- 多项式度数 = %d ---\n", degree)
		
		model := models.NewPolynomial(degree)
		
		err := model.Fit(X, y)
		if err != nil {
			log.Fatalf("训练失败: %v", err)
		}

		params := model.GetParameters()
		coeffs := params["coefficients"].([]float64)

		fmt.Printf("学到的系数: [")
		for i, coeff := range coeffs {
			if i > 0 {
				fmt.Printf(", ")
			}
			fmt.Printf("%.4f", coeff)
		}
		fmt.Printf("]\n")

		r2 := model.Score(X, y)
		fmt.Printf("R² 分数: %.4f\n\n", r2)
	}

	// 使用最佳模型进行预测
	bestModel := models.NewPolynomial(3)
	bestModel.Fit(X, y)

	fmt.Println("预测示例 (使用 3次多项式):")
	testX := mat.NewDense(5, 1, []float64{-2.0, -1.0, 0.0, 1.0, 2.0})
	predictions := bestModel.Predict(testX)
	
	for i := 0; i < 5; i++ {
		x := testX.At(i, 0)
		pred := predictions.At(i, 0)
		expected := 2 + 3*x - 0.5*x*x + 0.1*x*x*x
		
		fmt.Printf("x = %4.1f -> 预测: %7.4f, 期望: %7.4f, 误差: %6.4f\n", 
			x, pred, expected, math.Abs(pred-expected))
	}

	fmt.Println("\n=== Polynomial 示例完成 ===")
}
