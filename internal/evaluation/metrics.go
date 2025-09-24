package evaluation

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/mat"
)

// MSE 计算均方误差 (Mean Squared Error)
func MSE(yTrue, yPred []float64) (float64, error) {
	if len(yTrue) != len(yPred) {
		return 0, errors.New("预测值和真实值长度不匹配")
	}

	n := float64(len(yTrue))
	var sumSquaredError float64

	for i := range yTrue {
		diff := yTrue[i] - yPred[i]
		sumSquaredError += diff * diff
	}

	return sumSquaredError / n, nil
}

// RMSE 计算均方根误差 (Root Mean Squared Error)
func RMSE(yTrue, yPred []float64) (float64, error) {
	mse, err := MSE(yTrue, yPred)
	if err != nil {
		return 0, err
	}
	return math.Sqrt(mse), nil
}

// MAE 计算平均绝对误差 (Mean Absolute Error)
func MAE(yTrue, yPred []float64) (float64, error) {
	if len(yTrue) != len(yPred) {
		return 0, errors.New("预测值和真实值长度不匹配")
	}

	n := float64(len(yTrue))
	var sumAbsoluteError float64

	for i := range yTrue {
		diff := math.Abs(yTrue[i] - yPred[i])
		sumAbsoluteError += diff
	}

	return sumAbsoluteError / n, nil
}

// R2Score 计算决定系数 (Coefficient of Determination, R²)
func R2Score(yTrue, yPred []float64) (float64, error) {
	if len(yTrue) != len(yPred) {
		return 0, errors.New("预测值和真实值长度不匹配")
	}

	// 计算真实值的均值
	var yMean float64
	for _, y := range yTrue {
		yMean += y
	}
	yMean /= float64(len(yTrue))

	// 计算总平方和 (SST) 和残差平方和 (SSE)
	var sst, sse float64
	for i := range yTrue {
		sst += math.Pow(yTrue[i]-yMean, 2)
		sse += math.Pow(yTrue[i]-yPred[i], 2)
	}

	// 防止除零错误
	if sst == 0 {
		return 1.0, nil // 完美拟合
	}

	// R² = 1 - (SSE/SST)
	return 1.0 - (sse / sst), nil
}

// MSEMat 使用gonum矩阵计算均方误差
func MSEMat(yTrue, yPred *mat.VecDense) float64 {
	n := yTrue.Len()
	var sumSquaredError float64

	for i := 0; i < n; i++ {
		diff := yTrue.At(i, 0) - yPred.At(i, 0)
		sumSquaredError += diff * diff
	}

	return sumSquaredError / float64(n)
}

// RMSEMat 使用gonum矩阵计算均方根误差
func RMSEMat(yTrue, yPred *mat.VecDense) float64 {
	return math.Sqrt(MSEMat(yTrue, yPred))
}

// MAEMat 使用gonum矩阵计算平均绝对误差
func MAEMat(yTrue, yPred *mat.VecDense) float64 {
	n := yTrue.Len()
	var sumAbsoluteError float64

	for i := 0; i < n; i++ {
		diff := math.Abs(yTrue.At(i, 0) - yPred.At(i, 0))
		sumAbsoluteError += diff
	}

	return sumAbsoluteError / float64(n)
}

// R2ScoreMat 使用gonum矩阵计算决定系数
func R2ScoreMat(yTrue, yPred *mat.VecDense) float64 {
	n := yTrue.Len()

	// 计算真实值的均值
	var yMean float64
	for i := 0; i < n; i++ {
		yMean += yTrue.At(i, 0)
	}
	yMean /= float64(n)

	// 计算总平方和 (SST) 和残差平方和 (SSE)
	var sst, sse float64
	for i := 0; i < n; i++ {
		sst += math.Pow(yTrue.At(i, 0)-yMean, 2)
		sse += math.Pow(yTrue.At(i, 0)-yPred.At(i, 0), 2)
	}

	// 防止除零错误
	if sst == 0 {
		return 1.0 // 完美拟合
	}

	// R² = 1 - (SSE/SST)
	return 1.0 - (sse / sst)
}

// EvaluateModel 计算所有评估指标并返回结果映射
func EvaluateModel(yTrue, yPred []float64) (map[string]float64, error) {
	metrics := make(map[string]float64)

	r2, err := R2Score(yTrue, yPred)
	if err != nil {
		return nil, err
	}
	metrics["r2"] = r2

	mse, err := MSE(yTrue, yPred)
	if err != nil {
		return nil, err
	}
	metrics["mse"] = mse

	rmse, err := RMSE(yTrue, yPred)
	if err != nil {
		return nil, err
	}
	metrics["rmse"] = rmse

	mae, err := MAE(yTrue, yPred)
	if err != nil {
		return nil, err
	}
	metrics["mae"] = mae

	return metrics, nil
}

// EvaluateModelMat 使用gonum矩阵计算所有评估指标
func EvaluateModelMat(yTrue, yPred *mat.VecDense) map[string]float64 {
	metrics := make(map[string]float64)

	metrics["r2"] = R2ScoreMat(yTrue, yPred)
	metrics["mse"] = MSEMat(yTrue, yPred)
	metrics["rmse"] = RMSEMat(yTrue, yPred)
	metrics["mae"] = MAEMat(yTrue, yPred)

	return metrics
}
