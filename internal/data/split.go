package data

import (
	"errors"
	"go-model/pkg/types"
	"math/rand"
	"time"
)

// SplitDataset 将数据集分割为训练集和测试集
// testRatio: 测试集比例（0-1之间）
// shuffle: 是否随机打乱数据
func SplitDataset(data *types.Dataset, testRatio float64, shuffle bool) (*types.Dataset, *types.Dataset, error) {
	if data == nil || !data.IsValid() {
		return nil, nil, errors.New("无效的数据集")
	}

	if testRatio <= 0 || testRatio >= 1 {
		return nil, nil, errors.New("测试集比例必须在0和1之间")
	}

	nSamples := data.NumSamples()
	testSize := int(float64(nSamples) * testRatio)
	trainSize := nSamples - testSize

	// 创建索引数组
	indices := make([]int, nSamples)
	for i := 0; i < nSamples; i++ {
		indices[i] = i
	}

	// 随机打乱索引
	if shuffle {
		rand.Seed(time.Now().UnixNano())
		rand.Shuffle(nSamples, func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
	}

	// 创建训练集
	trainFeatures := make([][]float64, trainSize)
	trainTarget := make([]float64, trainSize)

	for i := 0; i < trainSize; i++ {
		idx := indices[i]
		trainFeatures[i] = make([]float64, len(data.Features[idx]))
		copy(trainFeatures[i], data.Features[idx])
		trainTarget[i] = data.Target[idx]
	}

	// 创建测试集
	testFeatures := make([][]float64, testSize)
	testTarget := make([]float64, testSize)

	for i := 0; i < testSize; i++ {
		idx := indices[trainSize+i]
		testFeatures[i] = make([]float64, len(data.Features[idx]))
		copy(testFeatures[i], data.Features[idx])
		testTarget[i] = data.Target[idx]
	}

	// 创建训练集和测试集数据集
	trainDataset := types.NewDataset(trainFeatures, trainTarget, data.FeatureNames)
	testDataset := types.NewDataset(testFeatures, testTarget, data.FeatureNames)

	return trainDataset, testDataset, nil
}

// TrainTestSplit 是SplitDataset的便捷包装函数，默认打乱数据
func TrainTestSplit(data *types.Dataset, testRatio float64) (*types.Dataset, *types.Dataset, error) {
	return SplitDataset(data, testRatio, true)
}

// CrossValidationSplit 将数据集分割为k折交叉验证的折
func CrossValidationSplit(data *types.Dataset, k int) ([]*types.Dataset, []*types.Dataset, error) {
	if data == nil || !data.IsValid() {
		return nil, nil, errors.New("无效的数据集")
	}

	if k <= 1 {
		return nil, nil, errors.New("交叉验证的折数必须大于1")
	}

	nSamples := data.NumSamples()
	if k > nSamples {
		return nil, nil, errors.New("交叉验证的折数不能大于样本数量")
	}

	// 创建索引数组并打乱
	indices := make([]int, nSamples)
	for i := 0; i < nSamples; i++ {
		indices[i] = i
	}

	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(nSamples, func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	// 计算每折的大小
	foldSize := nSamples / k
	extraSamples := nSamples % k

	// 准备训练集和测试集的折
	trainFolds := make([]*types.Dataset, k)
	testFolds := make([]*types.Dataset, k)

	// 分割数据
	start := 0
	for i := 0; i < k; i++ {
		// 计算当前折的大小
		size := foldSize
		if i < extraSamples {
			size++
		}

		// 创建当前折的测试集
		testIndices := indices[start : start+size]
		testFeatures := make([][]float64, len(testIndices))
		testTarget := make([]float64, len(testIndices))

		for j, idx := range testIndices {
			testFeatures[j] = make([]float64, len(data.Features[idx]))
			copy(testFeatures[j], data.Features[idx])
			testTarget[j] = data.Target[idx]
		}

		testFolds[i] = types.NewDataset(testFeatures, testTarget, data.FeatureNames)

		// 创建当前折的训练集（除了测试集的所有数据）
		trainIndices := make([]int, 0, nSamples-size)
		trainIndices = append(trainIndices, indices[:start]...)
		trainIndices = append(trainIndices, indices[start+size:]...)

		trainFeatures := make([][]float64, len(trainIndices))
		trainTarget := make([]float64, len(trainIndices))

		for j, idx := range trainIndices {
			trainFeatures[j] = make([]float64, len(data.Features[idx]))
			copy(trainFeatures[j], data.Features[idx])
			trainTarget[j] = data.Target[idx]
		}

		trainFolds[i] = types.NewDataset(trainFeatures, trainTarget, data.FeatureNames)

		// 更新起始位置
		start += size
	}

	return trainFolds, testFolds, nil
}
