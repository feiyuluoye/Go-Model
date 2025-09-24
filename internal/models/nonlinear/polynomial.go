package nonlinear

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
)

// Polynomial 多项式回归模型实现
type Polynomial struct {
	Coefficients *mat.VecDense
	Degree       int
	isTrained    bool
}

// NewPolynomial 创建新的多项式回归模型
func NewPolynomial(degree int) *Polynomial {
	return &Polynomial{
		Degree:    degree,
		isTrained: false,
	}
}

// Fit 训练多项式回归模型
func (p *Polynomial) Fit(X *mat.Dense, y *mat.VecDense) error {
	n, cols := X.Dims()
	if cols != 1 {
		return fmt.Errorf("polynomial regression requires single feature input")
	}

	// 转换X为多项式特征
	XPoly := mat.NewDense(n, p.Degree+1, nil)
	for i := 0; i < n; i++ {
		x := X.At(i, 0)
		for j := 0; j <= p.Degree; j++ {
			XPoly.Set(i, j, math.Pow(x, float64(j)))
		}
	}

	// 求解正规方程：beta = (X^T X)^-1 X^T y
	var XTX mat.Dense
	XTX.Mul(XPoly.T(), XPoly)

	var invXTX mat.Dense
	if err := invXTX.Inverse(&XTX); err != nil {
		return fmt.Errorf("singular matrix in polynomial regression - try reducing degree: %v", err)
	}

	var XTy mat.VecDense
	XTy.MulVec(XPoly.T(), y)

	// 存储系数
	p.Coefficients = mat.NewVecDense(p.Degree+1, nil)
	p.Coefficients.MulVec(&invXTX, &XTy)

	p.isTrained = true
	return nil
}

// Predict 使用训练好的多项式回归模型进行预测
func (p *Polynomial) Predict(X *mat.Dense) *mat.VecDense {
	n, cols := X.Dims()
	if cols != 1 {
		panic("polynomial regression requires single feature input")
	}

	predictions := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		x := X.At(i, 0)
		y := 0.0
		for j := 0; j <= p.Degree; j++ {
			y += p.Coefficients.At(j, 0) * math.Pow(x, float64(j))
		}
		predictions.SetVec(i, y)
	}

	return predictions
}

// Score 计算模型评分 (R²)
func (p *Polynomial) Score(X *mat.Dense, y *mat.VecDense) float64 {
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
func (p *Polynomial) GetParameters() map[string]interface{} {
	params := make(map[string]interface{})
	params["degree"] = p.Degree

	if p.Coefficients != nil {
		coeffs := make([]float64, p.Coefficients.Len())
		for i := 0; i < p.Coefficients.Len(); i++ {
			coeffs[i] = p.Coefficients.At(i, 0)
		}
		params["coefficients"] = coeffs
	}

	return params
}

// GetModelType 返回模型类型名称
func (p *Polynomial) GetModelType() string {
	return "Polynomial"
}
