package nonlinear

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
)

// Exponential 指数回归模型实现 y = a * exp(b * x)
type Exponential struct {
	A         float64 // 系数a
	B         float64 // 系数b
	isTrained bool
}

// NewExponential 创建新的指数回归模型
func NewExponential() *Exponential {
	return &Exponential{
		isTrained: false,
	}
}

// Fit 训练指数回归模型使用线性化
func (e *Exponential) Fit(X *mat.Dense, y *mat.VecDense) error {
	n, cols := X.Dims()
	if cols != 1 {
		return fmt.Errorf("exponential regression requires single feature input")
	}

	// 检查所有y值都为正（对数变换需要）
	for i := 0; i < n; i++ {
		if y.At(i, 0) <= 0 {
			return fmt.Errorf("exponential regression requires all y values to be positive")
		}
	}

	// 通过取y的自然对数进行线性化
	lnY := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		lnY.SetVec(i, math.Log(y.At(i, 0)))
	}

	// 创建设计矩阵 [1, x] 用于线性回归 ln(y) = ln(a) + b*x
	XDesign := mat.NewDense(n, 2, nil)
	for i := 0; i < n; i++ {
		XDesign.Set(i, 0, 1.0)           // 截距项
		XDesign.Set(i, 1, X.At(i, 0))   // x值
	}

	// 求解正规方程：beta = (X^T X)^-1 X^T lnY
	var XTX mat.Dense
	XTX.Mul(XDesign.T(), XDesign)

	var invXTX mat.Dense
	if err := invXTX.Inverse(&XTX); err != nil {
		return fmt.Errorf("singular matrix in exponential regression: %v", err)
	}

	var XTY mat.VecDense
	XTY.MulVec(XDesign.T(), lnY)

	beta := mat.NewVecDense(2, nil)
	beta.MulVec(&invXTX, &XTY)

	// 提取参数：ln(a) = beta[0], b = beta[1]
	e.A = math.Exp(beta.At(0, 0))
	e.B = beta.At(1, 0)
	e.isTrained = true

	return nil
}

// Predict 使用训练好的指数回归模型进行预测
func (e *Exponential) Predict(X *mat.Dense) *mat.VecDense {
	n, cols := X.Dims()
	if cols != 1 {
		panic("exponential regression requires single feature input")
	}

	predictions := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		x := X.At(i, 0)
		y := e.A * math.Exp(e.B*x)
		predictions.SetVec(i, y)
	}

	return predictions
}

// Score 计算模型评分 (R²)
func (e *Exponential) Score(X *mat.Dense, y *mat.VecDense) float64 {
	yPred := e.Predict(X)
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
func (e *Exponential) GetParameters() map[string]interface{} {
	params := make(map[string]interface{})
	params["a"] = e.A
	params["b"] = e.B
	return params
}

// GetModelType 返回模型类型名称
func (e *Exponential) GetModelType() string {
	return "Exponential"
}
