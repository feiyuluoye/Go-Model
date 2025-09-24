package linear

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

// Ridge Ridge回归模型实现
type Ridge struct {
	Coefficients *mat.VecDense
	Intercept    float64
	Lambda       float64 // 正则化参数
	isTrained    bool
}

// NewRidge 创建新的Ridge模型
func NewRidge(lambda float64) *Ridge {
	return &Ridge{
		Lambda:    lambda,
		isTrained: false,
	}
}

// Fit 训练Ridge模型
func (r *Ridge) Fit(X *mat.Dense, y *mat.VecDense) error {
	n, p := X.Dims()

	// 添加截距项
	XWithIntercept := mat.NewDense(n, p+1, nil)
	for i := 0; i < n; i++ {
		XWithIntercept.Set(i, 0, 1.0) // 截距项
		for j := 0; j < p; j++ {
			XWithIntercept.Set(i, j+1, X.At(i, j))
		}
	}

	// 计算 X^T X + λI
	var XTX mat.Dense
	XTX.Mul(XWithIntercept.T(), XWithIntercept)

	// 创建正则化矩阵（不对截距项正则化）
	identity := mat.NewDiagDense(p+1, nil)
	for i := 1; i < p+1; i++ { // 跳过截距项
		identity.SetDiag(i, r.Lambda)
	}

	// 添加正则化项
	XTXSymmetric := mat.NewSymDense(p+1, nil)
	for i := 0; i < p+1; i++ {
		for j := i; j < p+1; j++ {
			val := XTX.At(i, j)
			if i == j && i > 0 { // 非截距项的对角线元素
				val += r.Lambda
			}
			XTXSymmetric.SetSym(i, j, val)
		}
	}

	// 使用Cholesky分解求解
	var cholesky mat.Cholesky
	if ok := cholesky.Factorize(XTXSymmetric); !ok {
		// 如果Cholesky分解失败，尝试添加小的正则化项
		for i := 0; i < p+1; i++ {
			XTXSymmetric.SetSym(i, i, XTXSymmetric.At(i, i)+1e-10)
		}
		if ok := cholesky.Factorize(XTXSymmetric); !ok {
			return fmt.Errorf("matrix is not positive definite")
		}
	}

	// 计算 X^T y
	var XTy mat.VecDense
	XTy.MulVec(XWithIntercept.T(), y)

	// 求解线性方程组
	coefficients := mat.NewVecDense(p+1, nil)
	if err := cholesky.SolveVecTo(coefficients, &XTy); err != nil {
		return fmt.Errorf("failed to solve linear system: %v", err)
	}

	// 提取截距和系数
	r.Intercept = coefficients.AtVec(0)
	r.Coefficients = mat.NewVecDense(p, nil)
	for i := 0; i < p; i++ {
		r.Coefficients.SetVec(i, coefficients.AtVec(i+1))
	}

	r.isTrained = true
	return nil
}

// Predict 使用训练好的模型进行预测
func (r *Ridge) Predict(X *mat.Dense) *mat.VecDense {
	n, p := X.Dims()
	predictions := mat.NewVecDense(n, nil)

	for i := 0; i < n; i++ {
		prediction := r.Intercept
		for j := 0; j < p; j++ {
			prediction += X.At(i, j) * r.Coefficients.AtVec(j)
		}
		predictions.SetVec(i, prediction)
	}

	return predictions
}

// Score 计算模型评分 (R²)
func (r *Ridge) Score(X *mat.Dense, y *mat.VecDense) float64 {
	predictions := r.Predict(X)
	
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
func (r *Ridge) GetParameters() map[string]interface{} {
	params := make(map[string]interface{})
	params["lambda"] = r.Lambda
	params["intercept"] = r.Intercept
	
	if r.Coefficients != nil {
		coeffs := make([]float64, r.Coefficients.Len())
		for i := 0; i < r.Coefficients.Len(); i++ {
			coeffs[i] = r.Coefficients.AtVec(i)
		}
		params["coefficients"] = coeffs
	}
	
	return params
}

// GetModelType 返回模型类型名称
func (r *Ridge) GetModelType() string {
	return "Ridge"
}
