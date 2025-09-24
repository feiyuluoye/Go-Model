package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/feiyuluoye/Go-Model/internal/models"
	"gonum.org/v1/gonum/mat"
)

func main() {
	fmt.Println("=== OLS 回归示例 ===")

	// 生成示例数据：y = 2*x1 + 3*x2 + 1 + noise
	n := 100 // 样本数
	p := 2   // 特征数

	// 创建特征矩阵 X
	XData := make([]float64, n*p)
	yData := make([]float64, n)

	rand.Seed(42)
	for i := 0; i < n; i++ {
		x1 := rand.Float64()*10 - 5 // [-5, 5]
		x2 := rand.Float64()*10 - 5 // [-5, 5]
		
		XData[i*p] = x1
		XData[i*p+1] = x2
		
		// 真实关系：y = 2*x1 + 3*x2 + 1 + noise
		noise := rand.NormFloat64() * 0.5
		yData[i] = 2*x1 + 3*x2 + 1 + noise
	}

	X := mat.NewDense(n, p, XData)
	y := mat.NewVecDense(n, yData)

	fmt.Printf("数据集大小: %d 样本, %d 特征\n", n, p)
	fmt.Printf("真实系数: [2.0, 3.0], 截距: 1.0\n\n")

	// 创建并训练 OLS 模型
	model := models.NewOLS()
	
	fmt.Println("训练 OLS 模型...")
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

	// 计算训练集上的 R² 分数
	r2 := model.Score(X, y)
	fmt.Printf("训练集 R² 分数: %.4f\n\n", r2)

	// 进行预测
	fmt.Println("预测示例:")
	testX := mat.NewDense(3, 2, []float64{
		1.0, 2.0,  // 预期: 2*1 + 3*2 + 1 = 9
		-1.0, 1.0, // 预期: 2*(-1) + 3*1 + 1 = 2
		0.0, 0.0,  // 预期: 2*0 + 3*0 + 1 = 1
	})

	predictions := model.Predict(testX)
	
	for i := 0; i < 3; i++ {
		x1, x2 := testX.At(i, 0), testX.At(i, 1)
		pred := predictions.At(i, 0)
		expected := 2*x1 + 3*x2 + 1
		fmt.Printf("输入: [%.1f, %.1f] -> 预测: %.4f, 期望: %.4f\n", 
			x1, x2, pred, expected)
	}

	fmt.Println("\n=== OLS 示例完成 ===")
}
