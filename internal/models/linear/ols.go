package linear

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

// OLS 普通最小二乘法回归模型
type OLS struct {
	Coefficients *mat.VecDense
	Intercept    float64
	isTrained    bool
}

// NewOLS 创建新的OLS回归器
func NewOLS() *OLS {
	return &OLS{
		isTrained: false,
	}
}

// Fit 训练OLS模型
func (o *OLS) Fit(X *mat.Dense, y *mat.VecDense) error {
	n, p := X.Dims()
	if n == 0 || p == 0 {
		return fmt.Errorf("empty feature matrix")
	}
	
	yRows, _ := y.Dims()
	if yRows != n {
		return fmt.Errorf("mismatched dimensions: X has %d samples, y has %d", n, yRows)
	}

	// 添加截距项
	XWithIntercept := mat.NewDense(n, p+1, nil)
	for i := 0; i < n; i++ {
		XWithIntercept.Set(i, 0, 1.0) // 截距项
		for j := 0; j < p; j++ {
			XWithIntercept.Set(i, j+1, X.At(i, j))
		}
	}

	// 使用正规方程: (X^T * X)^-1 * X^T * y
	var XTX mat.Dense
	XTX.Mul(XWithIntercept.T(), XWithIntercept)

	var invXTX mat.Dense
	if err := invXTX.Inverse(&XTX); err != nil {
		return fmt.Errorf("failed to invert matrix: %v", err)
	}

	var XTy mat.VecDense
	XTy.MulVec(XWithIntercept.T(), y)

	// 计算系数
	coefficients := mat.NewVecDense(p+1, nil)
	coefficients.MulVec(&invXTX, &XTy)

	// 提取截距和系数
	o.Intercept = coefficients.AtVec(0)
	o.Coefficients = mat.NewVecDense(p, nil)
	for i := 0; i < p; i++ {
		o.Coefficients.SetVec(i, coefficients.AtVec(i+1))
	}

	o.isTrained = true
	return nil
}

// Predict 使用训练好的模型进行预测
func (o *OLS) Predict(X *mat.Dense) *mat.VecDense {
	n, p := X.Dims()
	predictions := mat.NewVecDense(n, nil)

	for i := 0; i < n; i++ {
		prediction := o.Intercept
		for j := 0; j < p; j++ {
			prediction += X.At(i, j) * o.Coefficients.AtVec(j)
		}
		predictions.SetVec(i, prediction)
	}

	return predictions
}

// Score 计算模型评分 (R²)
func (o *OLS) Score(X *mat.Dense, y *mat.VecDense) float64 {
	predictions := o.Predict(X)
	
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
		diff = y.At(i, 0) - predictions.At(i, 0)
		ssRes += diff * diff
	}

	if ssTotal == 0 {
		return 1.0 // 完美拟合
	}
	return 1 - ssRes/ssTotal
}

// GetParameters 返回模型参数
func (o *OLS) GetParameters() map[string]interface{} {
	params := make(map[string]interface{})
	params["intercept"] = o.Intercept
	
	if o.Coefficients != nil {
		coeffs := make([]float64, o.Coefficients.Len())
		for i := 0; i < o.Coefficients.Len(); i++ {
			coeffs[i] = o.Coefficients.AtVec(i)
		}
		params["coefficients"] = coeffs
	}
	
	return params
}

// GetModelType 返回模型类型名称
func (o *OLS) GetModelType() string {
	return "OLS"
}
