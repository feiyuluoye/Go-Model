package data

import (
	"errors"
	"github.com/feiyuluoye/Go-Model/internal/types"
	"math"
)

// StandardScaler 实现数据标准化（z-score标准化）
type StandardScaler struct {
	Mean   []float64
	StdDev []float64
	Fitted bool
}

// NewStandardScaler 创建一个新的StandardScaler实例
func NewStandardScaler() *StandardScaler {
	return &StandardScaler{
		Fitted: false,
	}
}

// Fit 计算特征的均值和标准差
func (sc *StandardScaler) Fit(data *types.Dataset) error {
	if data == nil || !data.IsValid() {
		return errors.New("无效的数据集")
	}

	nSamples := data.NumSamples()
	nFeatures := data.NumFeatures()

	sc.Mean = make([]float64, nFeatures)
	sc.StdDev = make([]float64, nFeatures)

	// 计算均值
	for i := 0; i < nFeatures; i++ {
		sum := 0.0
		for j := 0; j < nSamples; j++ {
			sum += data.Features[j][i]
		}
		sc.Mean[i] = sum / float64(nSamples)
	}

	// 计算标准差
	for i := 0; i < nFeatures; i++ {
		sumSq := 0.0
		for j := 0; j < nSamples; j++ {
			diff := data.Features[j][i] - sc.Mean[i]
			sumSq += diff * diff
		}
		variance := sumSq / float64(nSamples)
		sc.StdDev[i] = math.Sqrt(variance)
	}

	sc.Fitted = true
	return nil
}

// Transform 使用计算好的均值和标准差对数据进行标准化
func (sc *StandardScaler) Transform(data *types.Dataset) (*types.Dataset, error) {
	if !sc.Fitted {
		return nil, errors.New("scaler尚未拟合，请先调用Fit方法")
	}

	if data == nil || !data.IsValid() {
		return nil, errors.New("无效的数据集")
	}

	nSamples := data.NumSamples()
	nFeatures := data.NumFeatures()

	if nFeatures != len(sc.Mean) {
		return nil, errors.New("特征数量不匹配")
	}

	// 创建标准化后的特征矩阵
	scaledFeatures := make([][]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		scaledFeatures[i] = make([]float64, nFeatures)
		for j := 0; j < nFeatures; j++ {
			if sc.StdDev[j] > 0 {
				scaledFeatures[i][j] = (data.Features[i][j] - sc.Mean[j]) / sc.StdDev[j]
			} else {
				scaledFeatures[i][j] = 0.0 // 如果标准差为0，设置为0
			}
		}
	}

	// 创建新的数据集
	return types.NewDataset(scaledFeatures, data.Target, data.FeatureNames), nil
}

// FitTransform 结合Fit和Transform一步完成
func (sc *StandardScaler) FitTransform(data *types.Dataset) (*types.Dataset, error) {
	err := sc.Fit(data)
	if err != nil {
		return nil, err
	}
	return sc.Transform(data)
}

// MinMaxScaler 实现数据归一化（Min-Max归一化）
type MinMaxScaler struct {
	Min    []float64
	Max    []float64
	Fitted bool
}

// NewMinMaxScaler 创建一个新的MinMaxScaler实例
func NewMinMaxScaler() *MinMaxScaler {
	return &MinMaxScaler{
		Fitted: false,
	}
}

// Fit 计算特征的最小值和最大值
func (sc *MinMaxScaler) Fit(data *types.Dataset) error {
	if data == nil || !data.IsValid() {
		return errors.New("无效的数据集")
	}

	nSamples := data.NumSamples()
	nFeatures := data.NumFeatures()

	sc.Min = make([]float64, nFeatures)
	sc.Max = make([]float64, nFeatures)

	// 初始化min和max
	for i := 0; i < nFeatures; i++ {
		sc.Min[i] = data.Features[0][i]
		sc.Max[i] = data.Features[0][i]
		for j := 1; j < nSamples; j++ {
			if data.Features[j][i] < sc.Min[i] {
				sc.Min[i] = data.Features[j][i]
			}
			if data.Features[j][i] > sc.Max[i] {
				sc.Max[i] = data.Features[j][i]
			}
		}
	}

	sc.Fitted = true
	return nil
}

// Transform 使用计算好的最小值和最大值对数据进行归一化
func (sc *MinMaxScaler) Transform(data *types.Dataset) (*types.Dataset, error) {
	if !sc.Fitted {
		return nil, errors.New("scaler尚未拟合，请先调用Fit方法")
	}

	if data == nil || !data.IsValid() {
		return nil, errors.New("无效的数据集")
	}

	nSamples := data.NumSamples()
	nFeatures := data.NumFeatures()

	if nFeatures != len(sc.Min) {
		return nil, errors.New("特征数量不匹配")
	}

	// 创建归一化后的特征矩阵
	normalizedFeatures := make([][]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		normalizedFeatures[i] = make([]float64, nFeatures)
		for j := 0; j < nFeatures; j++ {
			if sc.Max[j] > sc.Min[j] {
				normalizedFeatures[i][j] = (data.Features[i][j] - sc.Min[j]) / (sc.Max[j] - sc.Min[j])
			} else {
				normalizedFeatures[i][j] = 0.0 // 如果最大值等于最小值，设置为0
			}
		}
	}

	// 创建新的数据集
	return types.NewDataset(normalizedFeatures, data.Target, data.FeatureNames), nil
}

// FitTransform 结合Fit和Transform一步完成
func (sc *MinMaxScaler) FitTransform(data *types.Dataset) (*types.Dataset, error) {
	err := sc.Fit(data)
	if err != nil {
		return nil, err
	}
	return sc.Transform(data)
}
