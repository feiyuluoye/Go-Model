package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/feiyuluoye/Go-Model/internal/models"
	"gonum.org/v1/gonum/mat"
)

func main() {
	fmt.Println("=== Ridge 回归示例 ===")

	// 生成带有多重共线性的数据
	n := 100
	p := 5

	XData := make([]float64, n*p)
	yData := make([]float64, n)

	rand.Seed(42)
	for i := 0; i < n; i++ {
		// 创建相关特征来演示Ridge的正则化效果
		x1 := rand.Float64()*4 - 2
		x2 := x1 + rand.NormFloat64()*0.1 // x2与x1高度相关
		x3 := rand.Float64()*4 - 2
		x4 := x3 + rand.NormFloat64()*0.1 // x4与x3高度相关
		x5 := rand.Float64()*4 - 2

		XData[i*p] = x1
		XData[i*p+1] = x2
		XData[i*p+2] = x3
		XData[i*p+3] = x4
		XData[i*p+4] = x5

		// 真实关系：y = 1*x1 + 1*x2 + 2*x3 + 2*x4 + 0.5*x5 + noise
		noise := rand.NormFloat64() * 0.5
		yData[i] = x1 + x2 + 2*x3 + 2*x4 + 0.5*x5 + noise
	}

	X := mat.NewDense(n, p, XData)
	y := mat.NewVecDense(n, yData)

	fmt.Printf("数据集大小: %d 样本, %d 特征 (包含多重共线性)\n", n, p)
	fmt.Printf("真实系数: [1.0, 1.0, 2.0, 2.0, 0.5]\n\n")

	// 比较不同的正则化强度
	lambdas := []float64{0.0, 0.1, 1.0, 10.0}

	for _, lambda := range lambdas {
		fmt.Printf("--- Lambda = %.1f ---\n", lambda)
		
		model := models.NewRidge(lambda)
		
		err := model.Fit(X, y)
		if err != nil {
			log.Fatalf("训练失败: %v", err)
		}

		params := model.GetParameters()
		coeffs := params["coefficients"].([]float64)
		intercept := params["intercept"].(float64)

		fmt.Printf("系数: [")
		for i, coeff := range coeffs {
			if i > 0 {
				fmt.Printf(", ")
			}
			fmt.Printf("%.4f", coeff)
		}
		fmt.Printf("]\n")
		fmt.Printf("截距: %.4f\n", intercept)

		r2 := model.Score(X, y)
		fmt.Printf("R² 分数: %.4f\n\n", r2)
	}

	// 使用最佳模型进行预测
	bestModel := models.NewRidge(1.0)
	bestModel.Fit(X, y)

	fmt.Println("预测示例 (使用 λ=1.0):")
	testX := mat.NewDense(2, 5, []float64{
		1.0, 1.0, 1.0, 1.0, 1.0, // 预期: 1+1+2+2+0.5 = 6.5
		0.0, 0.0, 0.0, 0.0, 0.0, // 预期: 0
	})

	predictions := bestModel.Predict(testX)
	
	for i := 0; i < 2; i++ {
		fmt.Printf("输入 %d: ", i+1)
		for j := 0; j < 5; j++ {
			fmt.Printf("%.1f ", testX.At(i, j))
		}
		fmt.Printf("-> 预测: %.4f\n", predictions.At(i, 0))
	}

	fmt.Println("\n=== Ridge 示例完成 ===")
}
