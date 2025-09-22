package linear

import (
	"fmt"
	"math"
)

// OLS 普通最小二乘法回归模型
type OLS struct {
	Coefficients []float64
	Intercept    float64
	FitIntercept bool
}

// NewOLS 创建新的OLS回归器
func NewOLS(fitIntercept bool) *OLS {
	return &OLS{
		FitIntercept: fitIntercept,
	}
}

// Fit 训练OLS模型
func (o *OLS) Fit(X [][]float64, y []float64) error {
	if len(X) == 0 || len(X[0]) == 0 {
		return fmt.Errorf("empty feature matrix")
	}
	if len(y) != len(X) {
		return fmt.Errorf("mismatched dimensions: X has %d samples, y has %d", len(X), len(y))
	}

	nSamples := len(X)
	nFeatures := len(X[0])

	// 如果需要截距项，添加一列1
	if o.FitIntercept {
		X = o.addIntercept(X)
		nFeatures++
	}

	// 使用正规方程: (X^T * X)^-1 * X^T * y
	xtx := make([][]float64, nFeatures)
	for i := range xtx {
		xtx[i] = make([]float64, nFeatures)
	}

	xty := make([]float64, nFeatures)

	// 计算 X^T * X 和 X^T * y
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			for k := 0; k < nFeatures; k++ {
				xtx[j][k] += X[i][j] * X[i][k]
			}
			xty[j] += X[i][j] * y[i]
		}
	}

	// 求解线性方程组
	coefficients, err := o.solveLinearSystem(xtx, xty)
	if err != nil {
		return fmt.Errorf("failed to solve linear system: %v", err)
	}

	if o.FitIntercept {
		o.Intercept = coefficients[0]
		o.Coefficients = coefficients[1:]
	} else {
		o.Intercept = 0
		o.Coefficients = coefficients
	}

	return nil
}

// Predict 使用训练好的模型进行预测
func (o *OLS) Predict(X [][]float64) ([]float64, error) {
	if len(o.Coefficients) == 0 {
		return nil, fmt.Errorf("model not trained")
	}
	if len(X) == 0 {
		return nil, fmt.Errorf("empty input")
	}
	if len(X[0]) != len(o.Coefficients) {
		return nil, fmt.Errorf("mismatched feature dimensions")
	}

	predictions := make([]float64, len(X))
	for i, sample := range X {
		prediction := o.Intercept
		for j, feature := range sample {
			prediction += feature * o.Coefficients[j]
		}
		predictions[i] = prediction
	}

	return predictions, nil
}

// Score 计算模型评分 (R²)
func (o *OLS) Score(X [][]float64, y []float64) (float64, error) {
	predictions, err := o.Predict(X)
	if err != nil {
		return 0, err
	}

	// 计算总平方和
	var meanY, sst float64
	for _, val := range y {
		meanY += val
	}
	meanY /= float64(len(y))
	for _, val := range y {
		sst += (val - meanY) * (val - meanY)
	}

	// 计算残差平方和
	var sse float64
	for i, pred := range predictions {
		residual := y[i] - pred
		sse += residual * residual
	}

	// R² = 1 - SSE/SST
	if sst == 0 {
		return 1, nil // 完美拟合
	}
	return 1 - sse/sst, nil
}

// addIntercept 添加截距项（一列1）
func (o *OLS) addIntercept(X [][]float64) [][]float64 {
	result := make([][]float64, len(X))
	for i := range X {
		result[i] = make([]float64, len(X[i])+1)
		result[i][0] = 1 // 截距项
		copy(result[i][1:], X[i])
	}
	return result
}

// solveLinearSystem 使用高斯消元法求解线性方程组
func (o *OLS) solveLinearSystem(A [][]float64, b []float64) ([]float64, error) {
	n := len(b)

	// 创建增广矩阵
	augmented := make([][]float64, n)
	for i := range augmented {
		augmented[i] = make([]float64, n+1)
		copy(augmented[i][:n], A[i])
		augmented[i][n] = b[i]
	}

	// 前向消元
	for i := 0; i < n; i++ {
		// 寻找主元
		maxRow := i
		for k := i + 1; k < n; k++ {
			if math.Abs(augmented[k][i]) > math.Abs(augmented[maxRow][i]) {
				maxRow = k
			}
		}

		// 交换行
		augmented[i], augmented[maxRow] = augmented[maxRow], augmented[i]

		// 检查主元是否为0
		if math.Abs(augmented[i][i]) < 1e-10 {
			return nil, fmt.Errorf("matrix is singular")
		}

		// 消元
		for k := i + 1; k < n; k++ {
			factor := augmented[k][i] / augmented[i][i]
			for j := i; j <= n; j++ {
				augmented[k][j] -= factor * augmented[i][j]
			}
		}
	}

	// 回代
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		x[i] = augmented[i][n]
		for j := i + 1; j < n; j++ {
			x[i] -= augmented[i][j] * x[j]
		}
		x[i] /= augmented[i][i]
	}

	return x, nil
}

// GetCoefficients 返回模型系数
func (o *OLS) GetCoefficients() []float64 {
	return o.Coefficients
}

// GetIntercept 返回截距
func (o *OLS) GetIntercept() float64 {
	return o.Intercept
}
