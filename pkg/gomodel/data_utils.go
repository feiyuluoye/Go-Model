package gomodel

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/feiyuluoye/Go-Model/internal/data"
	"github.com/feiyuluoye/Go-Model/internal/types"
	"gonum.org/v1/gonum/mat"
)

// DataUtils 提供数据处理相关的实用工具
type DataUtils struct {
	randomSeed int64
}

// NewDataUtils 创建数据工具实例
func NewDataUtils(randomSeed int64) *DataUtils {
	if randomSeed == 0 {
		randomSeed = time.Now().UnixNano()
	}
	return &DataUtils{randomSeed: randomSeed}
}

// LoadFromCSV 从CSV文件加载数据
func (du *DataUtils) LoadFromCSV(filePath string, targetColumn interface{}, hasHeader bool) (*TrainingData, error) {
	dataset, err := data.LoadCSV(filePath, hasHeader, targetColumn)
	if err != nil {
		return nil, &Error{
			Code:    ErrInvalidData,
			Message: "failed to load CSV data",
			Details: err.Error(),
		}
	}

	return du.convertToTrainingData(dataset), nil
}

// LoadFromJSON 从JSON文件加载数据
func (du *DataUtils) LoadFromJSON(filePath string) (*TrainingData, error) {
	dataset, err := data.LoadJSON(filePath)
	if err != nil {
		return nil, &Error{
			Code:    ErrInvalidData,
			Message: "failed to load JSON data",
			Details: err.Error(),
		}
	}

	return du.convertToTrainingData(dataset), nil
}

// CreateFromArrays 从数组创建训练数据
func (du *DataUtils) CreateFromArrays(features [][]float64, target []float64, featureNames []string, targetName string) (*TrainingData, error) {
	if len(features) == 0 || len(target) == 0 {
		return nil, &Error{
			Code:    ErrInvalidData,
			Message: "features and target cannot be empty",
		}
	}

	if len(features) != len(target) {
		return nil, &Error{
			Code:    ErrInvalidData,
			Message: fmt.Sprintf("features length (%d) must match target length (%d)", len(features), len(target)),
		}
	}

	// 创建特征矩阵
	featureMatrix := mat.NewDense(len(features), len(features[0]), nil)
	for i, row := range features {
		for j, val := range row {
			featureMatrix.Set(i, j, val)
		}
	}

	// 创建目标向量
	targetVector := mat.NewVecDense(len(target), target)

	return &TrainingData{
		Features:     featureMatrix,
		Target:       targetVector,
		FeatureNames: featureNames,
		TargetName:   targetName,
	}, nil
}

// SplitTrainTest 分割训练和测试数据
func (du *DataUtils) SplitTrainTest(data *TrainingData, testSize float64, shuffle bool) (*TrainingData, *TrainingData, error) {
	if testSize <= 0 || testSize >= 1 {
		return nil, nil, &Error{
			Code:    ErrInvalidParameters,
			Message: "test size must be between 0 and 1",
		}
	}

	r, c := data.Features.Dims()
	testCount := int(float64(r) * testSize)
	trainCount := r - testCount

	// 创建索引
	indices := make([]int, r)
	for i := range indices {
		indices[i] = i
	}

	// 如果需要打乱
	if shuffle {
		rand.Seed(du.randomSeed)
		rand.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
	}

	// 分割索引
	trainIndices := indices[:trainCount]
	testIndices := indices[trainCount:]

	// 创建训练数据
	trainFeatures := mat.NewDense(trainCount, c, nil)
	trainTarget := mat.NewVecDense(trainCount, nil)
	for i, idx := range trainIndices {
		for j := 0; j < c; j++ {
			trainFeatures.Set(i, j, data.Features.At(idx, j))
		}
		trainTarget.SetVec(i, data.Target.AtVec(idx))
	}

	// 创建测试数据
	testFeatures := mat.NewDense(testCount, c, nil)
	testTarget := mat.NewVecDense(testCount, nil)
	for i, idx := range testIndices {
		for j := 0; j < c; j++ {
			testFeatures.Set(i, j, data.Features.At(idx, j))
		}
		testTarget.SetVec(i, data.Target.AtVec(idx))
	}

	trainData := &TrainingData{
		Features:     trainFeatures,
		Target:       trainTarget,
		FeatureNames: data.FeatureNames,
		TargetName:   data.TargetName,
	}

	testData := &TrainingData{
		Features:     testFeatures,
		Target:       testTarget,
		FeatureNames: data.FeatureNames,
		TargetName:   data.TargetName,
	}

	return trainData, testData, nil
}

// Normalize 标准化特征数据 (z-score normalization)
func (du *DataUtils) Normalize(data *TrainingData) (*TrainingData, error) {
	r, c := data.Features.Dims()
	normalizedFeatures := mat.NewDense(r, c, nil)

	// 对每个特征列进行标准化
	for j := 0; j < c; j++ {
		// 计算均值和标准差
		mean, std := du.calculateColumnStats(data.Features, j)
		
		// 标准化该列
		for i := 0; i < r; i++ {
			originalValue := data.Features.At(i, j)
			normalizedValue := (originalValue - mean) / std
			normalizedFeatures.Set(i, j, normalizedValue)
		}
	}

	return &TrainingData{
		Features:     normalizedFeatures,
		Target:       data.Target,
		FeatureNames: data.FeatureNames,
		TargetName:   data.TargetName,
	}, nil
}

// Scale 缩放特征数据到[0,1]范围 (min-max scaling)
func (du *DataUtils) Scale(data *TrainingData) (*TrainingData, error) {
	r, c := data.Features.Dims()
	scaledFeatures := mat.NewDense(r, c, nil)

	// 对每个特征列进行缩放
	for j := 0; j < c; j++ {
		// 计算最小值和最大值
		min, max := du.calculateColumnMinMax(data.Features, j)
		
		// 缩放该列
		for i := 0; i < r; i++ {
			originalValue := data.Features.At(i, j)
			scaledValue := (originalValue - min) / (max - min)
			scaledFeatures.Set(i, j, scaledValue)
		}
	}

	return &TrainingData{
		Features:     scaledFeatures,
		Target:       data.Target,
		FeatureNames: data.FeatureNames,
		TargetName:   data.TargetName,
	}, nil
}

// RemoveOutliers 移除异常值
func (du *DataUtils) RemoveOutliers(data *TrainingData, method string, threshold float64) (*TrainingData, error) {
	switch method {
	case "iqr":
		return du.removeOutliersIQR(data, threshold)
	case "zscore":
		return du.removeOutliersZScore(data, threshold)
	default:
		return nil, &Error{
			Code:    ErrInvalidParameters,
			Message: fmt.Sprintf("unsupported outlier removal method: %s", method),
		}
	}
}

// GenerateSyntheticData 生成合成数据用于测试
func (du *DataUtils) GenerateSyntheticData(samples int, features int, noiseLevel float64, dataType string) (*TrainingData, error) {
	rand.Seed(du.randomSeed)
	
	switch dataType {
	case "linear":
		return du.generateLinearData(samples, features, noiseLevel)
	case "polynomial":
		return du.generatePolynomialData(samples, features, noiseLevel)
	case "classification":
		return du.generateClassificationData(samples, features, noiseLevel)
	default:
		return nil, &Error{
			Code:    ErrInvalidParameters,
			Message: fmt.Sprintf("unsupported synthetic data type: %s", dataType),
		}
	}
}

// GetDataSummary 获取数据摘要统计信息
func (du *DataUtils) GetDataSummary(data *TrainingData) map[string]interface{} {
	r, c := data.Features.Dims()
	
	summary := map[string]interface{}{
		"samples":  r,
		"features": c,
		"feature_stats": make(map[string]map[string]float64),
		"target_stats":  make(map[string]float64),
	}

	// 特征统计
	featureStats := make(map[string]map[string]float64)
	for j := 0; j < c; j++ {
		featureName := fmt.Sprintf("feature_%d", j)
		if j < len(data.FeatureNames) && data.FeatureNames[j] != "" {
			featureName = data.FeatureNames[j]
		}
		
		mean, std := du.calculateColumnStats(data.Features, j)
		min, max := du.calculateColumnMinMax(data.Features, j)
		
		featureStats[featureName] = map[string]float64{
			"mean": mean,
			"std":  std,
			"min":  min,
			"max":  max,
		}
	}
	summary["feature_stats"] = featureStats

	// 目标变量统计
	targetMean, targetStd := du.calculateVectorStats(data.Target)
	targetMin, targetMax := du.calculateVectorMinMax(data.Target)
	
	summary["target_stats"] = map[string]float64{
		"mean": targetMean,
		"std":  targetStd,
		"min":  targetMin,
		"max":  targetMax,
	}

	return summary
}

// 辅助方法

func (du *DataUtils) convertToTrainingData(dataset *types.Dataset) *TrainingData {
	// 转换特征矩阵
	r := len(dataset.Features)
	c := len(dataset.Features[0])
	featureMatrix := mat.NewDense(r, c, nil)
	
	for i, row := range dataset.Features {
		for j, val := range row {
			featureMatrix.Set(i, j, val)
		}
	}

	// 转换目标向量
	targetVector := mat.NewVecDense(len(dataset.Target), dataset.Target)

	return &TrainingData{
		Features: featureMatrix,
		Target:   targetVector,
	}
}

func (du *DataUtils) calculateColumnStats(matrix *mat.Dense, col int) (mean, std float64) {
	r, _ := matrix.Dims()
	
	// 计算均值
	sum := 0.0
	for i := 0; i < r; i++ {
		sum += matrix.At(i, col)
	}
	mean = sum / float64(r)

	// 计算标准差
	sumSquares := 0.0
	for i := 0; i < r; i++ {
		diff := matrix.At(i, col) - mean
		sumSquares += diff * diff
	}
	std = math.Sqrt(sumSquares / float64(r))

	return mean, std
}

func (du *DataUtils) calculateColumnMinMax(matrix *mat.Dense, col int) (min, max float64) {
	r, _ := matrix.Dims()
	
	min = matrix.At(0, col)
	max = matrix.At(0, col)
	
	for i := 1; i < r; i++ {
		val := matrix.At(i, col)
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
	}

	return min, max
}

func (du *DataUtils) calculateVectorStats(vector *mat.VecDense) (mean, std float64) {
	n := vector.Len()
	
	// 计算均值
	sum := 0.0
	for i := 0; i < n; i++ {
		sum += vector.AtVec(i)
	}
	mean = sum / float64(n)

	// 计算标准差
	sumSquares := 0.0
	for i := 0; i < n; i++ {
		diff := vector.AtVec(i) - mean
		sumSquares += diff * diff
	}
	std = math.Sqrt(sumSquares / float64(n))

	return mean, std
}

func (du *DataUtils) calculateVectorMinMax(vector *mat.VecDense) (min, max float64) {
	n := vector.Len()
	
	min = vector.AtVec(0)
	max = vector.AtVec(0)
	
	for i := 1; i < n; i++ {
		val := vector.AtVec(i)
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
	}

	return min, max
}

func (du *DataUtils) removeOutliersIQR(data *TrainingData, multiplier float64) (*TrainingData, error) {
	// IQR方法移除异常值的实现
	// 这里简化实现，实际应该计算四分位数
	return data, nil
}

func (du *DataUtils) removeOutliersZScore(data *TrainingData, threshold float64) (*TrainingData, error) {
	// Z-score方法移除异常值的实现
	// 这里简化实现，实际应该计算z-score并过滤
	return data, nil
}

func (du *DataUtils) generateLinearData(samples, features int, noiseLevel float64) (*TrainingData, error) {
	// 生成线性关系的合成数据
	X := make([][]float64, samples)
	y := make([]float64, samples)
	
	// 生成随机系数
	coefficients := make([]float64, features)
	for i := range coefficients {
		coefficients[i] = rand.Float64()*4 - 2 // [-2, 2]
	}
	
	for i := 0; i < samples; i++ {
		X[i] = make([]float64, features)
		target := 0.0
		
		for j := 0; j < features; j++ {
			X[i][j] = rand.Float64()*10 - 5 // [-5, 5]
			target += coefficients[j] * X[i][j]
		}
		
		// 添加噪声
		noise := rand.NormFloat64() * noiseLevel
		y[i] = target + noise
	}
	
	return du.CreateFromArrays(X, y, nil, "target")
}

func (du *DataUtils) generatePolynomialData(samples, features int, noiseLevel float64) (*TrainingData, error) {
	// 生成多项式关系的合成数据
	X := make([][]float64, samples)
	y := make([]float64, samples)
	
	for i := 0; i < samples; i++ {
		X[i] = make([]float64, features)
		target := 0.0
		
		for j := 0; j < features; j++ {
			X[i][j] = rand.Float64()*4 - 2 // [-2, 2]
			// 简单的二次关系
			target += X[i][j] + 0.5*X[i][j]*X[i][j]
		}
		
		// 添加噪声
		noise := rand.NormFloat64() * noiseLevel
		y[i] = target + noise
	}
	
	return du.CreateFromArrays(X, y, nil, "target")
}

func (du *DataUtils) generateClassificationData(samples, features int, noiseLevel float64) (*TrainingData, error) {
	// 生成分类数据
	X := make([][]float64, samples)
	y := make([]float64, samples)
	
	for i := 0; i < samples; i++ {
		X[i] = make([]float64, features)
		sum := 0.0
		
		for j := 0; j < features; j++ {
			X[i][j] = rand.Float64()*4 - 2 // [-2, 2]
			sum += X[i][j]
		}
		
		// 简单的线性决策边界
		if sum > 0 {
			y[i] = 1.0
		} else {
			y[i] = 0.0
		}
		
		// 添加一些噪声（翻转标签）
		if rand.Float64() < noiseLevel {
			y[i] = 1.0 - y[i]
		}
	}
	
	return du.CreateFromArrays(X, y, nil, "class")
}
