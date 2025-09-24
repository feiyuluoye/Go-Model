package linear

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

// Lasso Lasso回归模型实现
type Lasso struct {
	Coefficients *mat.VecDense
	Intercept    float64
	Lambda       float64
	MaxIter      int
	Tol          float64
	isTrained    bool
}

// NewLasso 创建新的Lasso模型
func NewLasso(lambda float64) *Lasso {
	return &Lasso{
		Lambda:    lambda,
		MaxIter:   1000,
		Tol:       1e-4,
		isTrained: false,
	}
}

// Fit 训练Lasso模型使用坐标下降法
func (l *Lasso) Fit(X *mat.Dense, y *mat.VecDense) error {
	n, p := X.Dims()

	// 添加截距项
	XWithIntercept := mat.NewDense(n, p+1, nil)
	for i := 0; i < n; i++ {
		XWithIntercept.Set(i, 0, 1.0) // 截距项
		for j := 0; j < p; j++ {
			XWithIntercept.Set(i, j+1, X.At(i, j))
		}
	}

	// 初始化系数
	beta := mat.NewVecDense(p+1, nil)
	for i := 0; i < p+1; i++ {
		beta.SetVec(i, 0.0)
	}

	// 坐标下降算法
	for iter := 0; iter < l.MaxIter; iter++ {
		betaOld := mat.VecDenseCopyOf(beta)
		
		for j := 0; j < p+1; j++ {
			// 对截距项不进行正则化
			lambda := l.Lambda
			if j == 0 {
				lambda = 0
			}

			// 计算 rho = (1/n) * X_j^T (y - X_{-j} beta_{-j})
			var rho float64
			for i := 0; i < n; i++ {
				pred := 0.0
				for k := 0; k < p+1; k++ {
					if k != j {
						pred += XWithIntercept.At(i, k) * beta.AtVec(k)
					}
				}
				rho += XWithIntercept.At(i, j) * (y.At(i, 0) - pred)
			}
			rho /= float64(n)

			// 计算 X_j^T X_j / n
			xjNorm := 0.0
			for i := 0; i < n; i++ {
				xjNorm += XWithIntercept.At(i, j) * XWithIntercept.At(i, j)
			}
			xjNorm /= float64(n)

			// 软阈值操作
			if xjNorm > 0 {
				threshold := lambda / xjNorm
				if rho > threshold {
					beta.SetVec(j, (rho-threshold)/xjNorm)
				} else if rho < -threshold {
					beta.SetVec(j, (rho+threshold)/xjNorm)
				} else {
					beta.SetVec(j, 0.0)
				}
			}
		}

		// 检查收敛性
		maxDiff := 0.0
		for i := 0; i < p+1; i++ {
			diff := math.Abs(beta.AtVec(i) - betaOld.AtVec(i))
			if diff > maxDiff {
				maxDiff = diff
			}
		}
		if maxDiff < l.Tol {
			break
		}
	}

	// 提取截距和系数
	l.Intercept = beta.AtVec(0)
	l.Coefficients = mat.NewVecDense(p, nil)
	for i := 0; i < p; i++ {
		l.Coefficients.SetVec(i, beta.AtVec(i+1))
	}

	l.isTrained = true
	return nil
}

// Predict 使用训练好的模型进行预测
func (l *Lasso) Predict(X *mat.Dense) *mat.VecDense {
	n, p := X.Dims()
	predictions := mat.NewVecDense(n, nil)

	for i := 0; i < n; i++ {
		prediction := l.Intercept
		for j := 0; j < p; j++ {
			prediction += X.At(i, j) * l.Coefficients.AtVec(j)
		}
		predictions.SetVec(i, prediction)
	}

	return predictions
}

// Score 计算模型评分 (R²)
func (l *Lasso) Score(X *mat.Dense, y *mat.VecDense) float64 {
	predictions := l.Predict(X)
	
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
		return 1.0
	}
	return 1 - ssRes/ssTotal
}

// GetParameters 返回模型参数
func (l *Lasso) GetParameters() map[string]interface{} {
	params := make(map[string]interface{})
	params["lambda"] = l.Lambda
	params["intercept"] = l.Intercept
	
	if l.Coefficients != nil {
		coeffs := make([]float64, l.Coefficients.Len())
		for i := 0; i < l.Coefficients.Len(); i++ {
			coeffs[i] = l.Coefficients.AtVec(i)
		}
		params["coefficients"] = coeffs
	}
	
	return params
}

// GetModelType 返回模型类型名称
func (l *Lasso) GetModelType() string {
	return "Lasso"
}
