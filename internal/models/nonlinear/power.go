package nonlinear

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
)

// Power 幂回归模型实现 y = a * x^b
type Power struct {
	A         float64 // 系数a
	B         float64 // 指数b
	isTrained bool
}

// NewPower 创建新的幂回归模型
func NewPower() *Power {
	return &Power{
		isTrained: false,
	}
}

// Fit 训练幂回归模型使用线性化
func (p *Power) Fit(X *mat.Dense, y *mat.VecDense) error {
	n, cols := X.Dims()
	if cols != 1 {
		return fmt.Errorf("power regression requires single feature input")
	}

	// 检查所有x和y值都为正（对数变换需要）
	for i := 0; i < n; i++ {
		if X.At(i, 0) <= 0 {
			return fmt.Errorf("power regression requires all x values to be positive")
		}
		if y.At(i, 0) <= 0 {
			return fmt.Errorf("power regression requires all y values to be positive")
		}
	}

	// 通过取对数进行线性化：ln(y) = ln(a) + b*ln(x)
	lnX := mat.NewVecDense(n, nil)
	lnY := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		lnX.SetVec(i, math.Log(X.At(i, 0)))
		lnY.SetVec(i, math.Log(y.At(i, 0)))
	}

	// 创建设计矩阵 [1, ln(x)] 用于线性回归 ln(y) = ln(a) + b*ln(x)
	XDesign := mat.NewDense(n, 2, nil)
	for i := 0; i < n; i++ {
		XDesign.Set(i, 0, 1.0)           // 截距项
		XDesign.Set(i, 1, lnX.At(i, 0))  // ln(x)
	}

	// 求解正规方程：beta = (X^T X)^-1 X^T lnY
	var XTX mat.Dense
	XTX.Mul(XDesign.T(), XDesign)

	var invXTX mat.Dense
	if err := invXTX.Inverse(&XTX); err != nil {
		return fmt.Errorf("singular matrix in power regression: %v", err)
	}

	var XTY mat.VecDense
	XTY.MulVec(XDesign.T(), lnY)

	beta := mat.NewVecDense(2, nil)
	beta.MulVec(&invXTX, &XTY)

	// 提取参数：ln(a) = beta[0], b = beta[1]
	p.A = math.Exp(beta.At(0, 0))
	p.B = beta.At(1, 0)
	p.isTrained = true

	return nil
}

// Predict 使用训练好的幂回归模型进行预测
func (p *Power) Predict(X *mat.Dense) *mat.VecDense {
	n, cols := X.Dims()
	if cols != 1 {
		panic("power regression requires single feature input")
	}

	predictions := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		x := X.At(i, 0)
		if x <= 0 {
			panic("power regression requires positive x values for prediction")
		}
		y := p.A * math.Pow(x, p.B)
		predictions.SetVec(i, y)
	}

	return predictions
}

// Score 计算模型评分 (R²)
func (p *Power) Score(X *mat.Dense, y *mat.VecDense) float64 {
	yPred := p.Predict(X)
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
func (p *Power) GetParameters() map[string]interface{} {
	params := make(map[string]interface{})
	params["a"] = p.A
	params["b"] = p.B
	return params
}

// GetModelType 返回模型类型名称
func (p *Power) GetModelType() string {
	return "Power"
}
