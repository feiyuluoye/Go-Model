package data

import (
	"errors"
	"fmt"
	"github.com/feiyuluoye/Go-Model/internal/types"
	"math"
)

// PolynomialFeatures 生成多项式特征
type PolynomialFeatures struct {
	Degree int
}

// NewPolynomialFeatures 创建一个新的PolynomialFeatures实例
func NewPolynomialFeatures(degree int) (*PolynomialFeatures, error) {
	if degree < 1 {
		return nil, errors.New("多项式次数必须大于等于1")
	}
	return &PolynomialFeatures{
		Degree: degree,
	}, nil
}

// Transform 将原始特征转换为多项式特征
func (pf *PolynomialFeatures) Transform(data *types.Dataset) (*types.Dataset, error) {
	if data == nil || !data.IsValid() {
		return nil, errors.New("无效的数据集")
	}

	nSamples := data.NumSamples()
	nFeatures := data.NumFeatures()

	// 计算多项式特征的数量
	// 对于degree次多项式，特征数量为 (n_features + degree) choose degree
	// 这里我们简化计算，只考虑交互项
	newFeatureCount := 0
	for d := 1; d <= pf.Degree; d++ {
		newFeatureCount += int(math.Pow(float64(nFeatures), float64(d)))
	}

	// 创建新的特征矩阵
	newFeatures := make([][]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		newFeatures[i] = make([]float64, newFeatureCount)
		featureIndex := 0

		// 添加原始特征 (degree=1)
		for j := 0; j < nFeatures; j++ {
			newFeatures[i][featureIndex] = data.Features[i][j]
			featureIndex++
		}

		// 添加高阶特征
		for d := 2; d <= pf.Degree; d++ {
			// 生成所有可能的d次组合
			featureIndex = generateCombinations(data.Features[i], d, newFeatures[i], featureIndex)
		}
	}

	// 生成新的特征名称
	newFeatureNames := make([]string, newFeatureCount)
	featureIndex := 0

	// 添加原始特征名称
	for j := 0; j < nFeatures; j++ {
		newFeatureNames[featureIndex] = data.FeatureNames[j]
		featureIndex++
	}

	// 添加高阶特征名称
	for d := 2; d <= pf.Degree; d++ {
		featureIndex = generateCombinationNames(data.FeatureNames, d, newFeatureNames, featureIndex)
	}

	// 创建新的数据集
	return types.NewDataset(newFeatures, data.Target, newFeatureNames), nil
}

// generateCombinations 生成所有可能的特征组合的乘积
func generateCombinations(features []float64, degree int, result []float64, startIndex int) int {
	if degree == 1 {
		for i := 0; i < len(features); i++ {
			result[startIndex+i] = features[i]
		}
		return startIndex + len(features)
	}

	index := startIndex
	for i := 0; i < len(features); i++ {
		// 对于每个特征，递归地生成其与其他特征的组合
		remainingFeatures := features[i:]
		temp := make([]float64, len(remainingFeatures))
		for j := 0; j < len(remainingFeatures); j++ {
			temp[j] = features[i] * remainingFeatures[j]
		}
		index = generateCombinations(temp, degree-1, result, index)
	}
	return index
}

// generateCombinationNames 生成组合特征的名称
func generateCombinationNames(featureNames []string, degree int, result []string, startIndex int) int {
	if degree == 1 {
		for i := 0; i < len(featureNames); i++ {
			result[startIndex+i] = featureNames[i]
		}
		return startIndex + len(featureNames)
	}

	index := startIndex
	for i := 0; i < len(featureNames); i++ {
		remainingNames := featureNames[i:]
		tempNames := make([]string, len(remainingNames))
		for j := 0; j < len(remainingNames); j++ {
			tempNames[j] = fmt.Sprintf("%s*%s", featureNames[i], remainingNames[j])
		}
		index = generateCombinationNames(tempNames, degree-1, result, index)
	}
	return index
}

// AddPolynomialFeatures 向数据集添加多项式特征
func AddPolynomialFeatures(data *types.Dataset, degree int) (*types.Dataset, error) {
	pf, err := NewPolynomialFeatures(degree)
	if err != nil {
		return nil, err
	}
	return pf.Transform(data)
}

// AddInteractionTerms 添加特征交互项
func AddInteractionTerms(data *types.Dataset) (*types.Dataset, error) {
	if data == nil || !data.IsValid() {
		return nil, errors.New("无效的数据集")
	}

	nSamples := data.NumSamples()
	nFeatures := data.NumFeatures()

	// 计算交互项的数量: n*(n-1)/2
	interactionCount := nFeatures * (nFeatures - 1) / 2

	// 创建新的特征矩阵
	newFeatures := make([][]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		newFeatures[i] = make([]float64, nFeatures+interactionCount)
		// 复制原始特征
		copy(newFeatures[i], data.Features[i])
		index := nFeatures

		// 添加交互项
		for j := 0; j < nFeatures; j++ {
			for k := j + 1; k < nFeatures; k++ {
				newFeatures[i][index] = data.Features[i][j] * data.Features[i][k]
				index++
			}
		}
	}

	// 生成新的特征名称
	newFeatureNames := make([]string, nFeatures+interactionCount)
	copy(newFeatureNames, data.FeatureNames)
	index := nFeatures

	// 添加交互项名称
	for j := 0; j < nFeatures; j++ {
		for k := j + 1; k < nFeatures; k++ {
			newFeatureNames[index] = fmt.Sprintf("%s*%s", data.FeatureNames[j], data.FeatureNames[k])
			index++
		}
	}

	// 创建新的数据集
	return types.NewDataset(newFeatures, data.Target, newFeatureNames), nil
}

// DropLowVarianceFeatures 删除低方差特征
func DropLowVarianceFeatures(data *types.Dataset, threshold float64) (*types.Dataset, error) {
	if data == nil || !data.IsValid() {
		return nil, errors.New("无效的数据集")
	}

	nSamples := data.NumSamples()
	nFeatures := data.NumFeatures()

	// 计算每个特征的方差
	variances := make([]float64, nFeatures)
	for i := 0; i < nFeatures; i++ {
		// 计算均值
		mean := 0.0
		for j := 0; j < nSamples; j++ {
			mean += data.Features[j][i]
		}
		mean /= float64(nSamples)

		// 计算方差
		variance := 0.0
		for j := 0; j < nSamples; j++ {
			diff := data.Features[j][i] - mean
			variance += diff * diff
		}
		variance /= float64(nSamples)
		variances[i] = variance
	}

	// 选择方差大于阈值的特征
	selectedFeatures := [][]float64{}
	selectedNames := []string{}

	for i := 0; i < nFeatures; i++ {
		if variances[i] > threshold {
			selectedNames = append(selectedNames, data.FeatureNames[i])
			// 收集所有样本的这个特征
			featureValues := make([]float64, nSamples)
			for j := 0; j < nSamples; j++ {
				featureValues[j] = data.Features[j][i]
			}
			selectedFeatures = append(selectedFeatures, featureValues)
		}
	}

	// 转置特征矩阵 (从[特征][样本] 到 [样本][特征])
	transposedFeatures := make([][]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		transposedFeatures[i] = make([]float64, len(selectedFeatures))
		for j := 0; j < len(selectedFeatures); j++ {
			transposedFeatures[i][j] = selectedFeatures[j][i]
		}
	}

	// 创建新的数据集
	return types.NewDataset(transposedFeatures, data.Target, selectedNames), nil
}
