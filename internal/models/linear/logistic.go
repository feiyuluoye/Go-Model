package linear

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

// Logistic 逻辑回归模型实现
type Logistic struct {
	Coefficients *mat.VecDense
	Intercept    float64
	MaxIter      int
	Tol          float64
	LearningRate float64
	isTrained    bool
}

// NewLogistic 创建新的逻辑回归模型
func NewLogistic() *Logistic {
	return &Logistic{
		MaxIter:      1000,
		Tol:          1e-4,
		LearningRate: 0.01,
		isTrained:    false,
	}
}

// sigmoid 函数
func sigmoid(z float64) float64 {
	// 防止溢出
	if z > 30 {
		return 1.0
	} else if z < -30 {
		return 0.0
	}
	return 1.0 / (1.0 + math.Exp(-z))
}

// Fit 训练逻辑回归模型使用梯度下降
func (l *Logistic) Fit(X *mat.Dense, y *mat.VecDense) error {
	n, p := X.Dims()

	// 添加截距项
	XWithIntercept := mat.NewDense(n, p+1, nil)
	for i := 0; i < n; i++ {
		XWithIntercept.Set(i, 0, 1.0) // 截距项
		for j := 0; j < p; j++ {
			XWithIntercept.Set(i, j+1, X.At(i, j))
		}
	}

	// 初始化参数
	theta := mat.NewVecDense(p+1, nil)
	for i := 0; i < p+1; i++ {
		theta.SetVec(i, 0.0)
	}

	// 梯度下降
	for iter := 0; iter < l.MaxIter; iter++ {
		thetaOld := mat.VecDenseCopyOf(theta)

		// 前向传播：计算预测值
		predictions := mat.NewVecDense(n, nil)
		for i := 0; i < n; i++ {
			var z float64
			for j := 0; j < p+1; j++ {
				z += XWithIntercept.At(i, j) * theta.AtVec(j)
			}
			predictions.SetVec(i, sigmoid(z))
		}

		// 计算梯度
		gradient := mat.NewVecDense(p+1, nil)
		for j := 0; j < p+1; j++ {
			var sum float64
			for i := 0; i < n; i++ {
				error := predictions.At(i, 0) - y.At(i, 0)
				sum += XWithIntercept.At(i, j) * error
			}
			gradient.SetVec(j, sum/float64(n))
		}

		// 更新参数
		theta.AddScaledVec(theta, -l.LearningRate, gradient)

		// 检查收敛性
		maxDiff := 0.0
		for i := 0; i < p+1; i++ {
			diff := math.Abs(theta.AtVec(i) - thetaOld.AtVec(i))
			if diff > maxDiff {
				maxDiff = diff
			}
		}
		if maxDiff < l.Tol {
			break
		}
	}

	// 提取截距和系数
	l.Intercept = theta.AtVec(0)
	l.Coefficients = mat.NewVecDense(p, nil)
	for i := 0; i < p; i++ {
		l.Coefficients.SetVec(i, theta.AtVec(i+1))
	}

	l.isTrained = true
	return nil
}

// Predict 预测概率
func (l *Logistic) Predict(X *mat.Dense) *mat.VecDense {
	n, p := X.Dims()
	predictions := mat.NewVecDense(n, nil)

	for i := 0; i < n; i++ {
		z := l.Intercept
		for j := 0; j < p; j++ {
			z += X.At(i, j) * l.Coefficients.AtVec(j)
		}
		predictions.SetVec(i, sigmoid(z))
	}

	return predictions
}

// PredictClass 预测分类（0或1）
func (l *Logistic) PredictClass(X *mat.Dense, threshold float64) *mat.VecDense {
	probabilities := l.Predict(X)
	n, _ := probabilities.Dims()
	classifications := mat.NewVecDense(n, nil)

	for i := 0; i < n; i++ {
		if probabilities.At(i, 0) >= threshold {
			classifications.SetVec(i, 1.0)
		} else {
			classifications.SetVec(i, 0.0)
		}
	}

	return classifications
}

// Score 计算准确率
func (l *Logistic) Score(X *mat.Dense, y *mat.VecDense) float64 {
	predictions := l.PredictClass(X, 0.5)
	n, _ := y.Dims()
	correct := 0

	for i := 0; i < n; i++ {
		if predictions.At(i, 0) == y.At(i, 0) {
			correct++
		}
	}

	return float64(correct) / float64(n)
}

// GetParameters 返回模型参数
func (l *Logistic) GetParameters() map[string]interface{} {
	params := make(map[string]interface{})
	params["intercept"] = l.Intercept
	params["max_iter"] = l.MaxIter
	params["tol"] = l.Tol
	params["learning_rate"] = l.LearningRate
	
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
func (l *Logistic) GetModelType() string {
	return "Logistic"
}
