package nonlinear

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
)

// Logarithmic 对数回归模型实现 y = a * ln(x) + b
type Logarithmic struct {
	A         float64 // 系数a
	B         float64 // 系数b
	isTrained bool
}

// NewLogarithmic 创建新的对数回归模型
func NewLogarithmic() *Logarithmic {
	return &Logarithmic{
		isTrained: false,
	}
}

// Fit 训练对数回归模型
func (l *Logarithmic) Fit(X *mat.Dense, y *mat.VecDense) error {
	n, cols := X.Dims()
	if cols != 1 {
		return fmt.Errorf("logarithmic regression requires single feature input")
	}

	// 检查所有x值都为正（对数变换需要）
	for i := 0; i < n; i++ {
		if X.At(i, 0) <= 0 {
			return fmt.Errorf("logarithmic regression requires all x values to be positive")
		}
	}

	// 创建设计矩阵 [ln(x), 1] 用于线性回归 y = a*ln(x) + b
	XDesign := mat.NewDense(n, 2, nil)
	for i := 0; i < n; i++ {
		XDesign.Set(i, 0, math.Log(X.At(i, 0))) // ln(x)
		XDesign.Set(i, 1, 1.0)                  // 截距项
	}

	// 求解正规方程：beta = (X^T X)^-1 X^T y
	var XTX mat.Dense
	XTX.Mul(XDesign.T(), XDesign)

	var invXTX mat.Dense
	if err := invXTX.Inverse(&XTX); err != nil {
		return fmt.Errorf("singular matrix in logarithmic regression: %v", err)
	}

	var XTY mat.VecDense
	XTY.MulVec(XDesign.T(), y)

	beta := mat.NewVecDense(2, nil)
	beta.MulVec(&invXTX, &XTY)

	// 提取参数：a = beta[0], b = beta[1]
	l.A = beta.At(0, 0)
	l.B = beta.At(1, 0)
	l.isTrained = true

	return nil
}

// Predict 使用训练好的对数回归模型进行预测
func (l *Logarithmic) Predict(X *mat.Dense) *mat.VecDense {
	n, cols := X.Dims()
	if cols != 1 {
		panic("logarithmic regression requires single feature input")
	}

	predictions := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		x := X.At(i, 0)
		if x <= 0 {
			panic("logarithmic regression requires positive x values for prediction")
		}
		y := l.A*math.Log(x) + l.B
		predictions.SetVec(i, y)
	}

	return predictions
}

// Score 计算模型评分 (R²)
func (l *Logarithmic) Score(X *mat.Dense, y *mat.VecDense) float64 {
	yPred := l.Predict(X)
	var ssTotal, ssRes float64
	ymean := 0.0

	n, _ := y.Dims()
	for i := 0; i < n; i++ {
		ymean += y.At(i, 0)
	}
	ymean /= float64(n)

	for i := 0; i < n; i++ {
		diff := y.At(i, 0) - ymean
		ssTotal += diff * diff
		diff = y.At(i, 0) - yPred.At(i, 0)
		ssRes += diff * diff
	}

	if ssTotal == 0 {
		return 1.0
	}
	return 1 - ssRes/ssTotal
}

// GetParameters 返回模型参数
func (l *Logarithmic) GetParameters() map[string]interface{} {
	params := make(map[string]interface{})
	params["a"] = l.A
	params["b"] = l.B
	return params
}

// GetModelType 返回模型类型名称
func (l *Logarithmic) GetModelType() string {
	return "Logarithmic"
}
