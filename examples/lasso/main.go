package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/feiyuluoye/Go-Model/internal/models"
	"gonum.org/v1/gonum/mat"
)

func main() {
	fmt.Println("=== Lasso 回归示例 ===")

	// 生成稀疏数据（只有部分特征有用）
	n := 100
	p := 10

	XData := make([]float64, n*p)
	yData := make([]float64, n)

	rand.Seed(42)
	// 真实系数：只有前3个特征有用，其余为0
	trueCoeffs := []float64{2.0, -1.5, 3.0, 0, 0, 0, 0, 0, 0, 0}

	for i := 0; i < n; i++ {
		var y_true float64
		for j := 0; j < p; j++ {
			x := rand.NormFloat64()
			XData[i*p+j] = x
			y_true += trueCoeffs[j] * x
		}
		
		// 添加噪声
		noise := rand.NormFloat64() * 0.5
		yData[i] = y_true + noise
	}

	X := mat.NewDense(n, p, XData)
	y := mat.NewVecDense(n, yData)

	fmt.Printf("数据集大小: %d 样本, %d 特征\n", n, p)
	fmt.Printf("真实系数: 前3个有效 [2.0, -1.5, 3.0], 其余为0\n\n")

	// 比较不同的正则化强度
	lambdas := []float64{0.01, 0.1, 0.5, 1.0}

	for _, lambda := range lambdas {
		fmt.Printf("--- Lambda = %.2f ---\n", lambda)
		
		model := models.NewLasso(lambda)
		
		err := model.Fit(X, y)
		if err != nil {
			log.Fatalf("训练失败: %v", err)
		}

		params := model.GetParameters()
		coeffs := params["coefficients"].([]float64)
		intercept := params["intercept"].(float64)

		fmt.Printf("学到的系数: [")
		nonZeroCount := 0
		for i, coeff := range coeffs {
			if i > 0 {
				fmt.Printf(", ")
			}
			fmt.Printf("%.4f", coeff)
			if coeff != 0 {
				nonZeroCount++
			}
		}
		fmt.Printf("]\n")
		fmt.Printf("截距: %.4f\n", intercept)
		fmt.Printf("非零系数数量: %d/%d\n", nonZeroCount, len(coeffs))

		r2 := model.Score(X, y)
		fmt.Printf("R² 分数: %.4f\n\n", r2)
	}

	// 使用最佳模型进行预测
	bestModel := models.NewLasso(0.1)
	bestModel.Fit(X, y)

	fmt.Println("预测示例 (使用 λ=0.1):")
	testX := mat.NewDense(2, 10, []float64{
		1.0, -1.0, 1.0, 0.5, -0.5, 0.2, -0.2, 0.1, -0.1, 0.05, // 混合输入
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,      // 全零输入
	})

	predictions := bestModel.Predict(testX)
	
	for i := 0; i < 2; i++ {
		fmt.Printf("输入 %d: [", i+1)
		for j := 0; j < 3; j++ { // 只显示前3个重要特征
			if j > 0 {
				fmt.Printf(", ")
			}
			fmt.Printf("%.1f", testX.At(i, j))
		}
		fmt.Printf(", ...] -> 预测: %.4f\n", predictions.At(i, 0))
	}

	fmt.Println("\n=== Lasso 示例完成 ===")
}
