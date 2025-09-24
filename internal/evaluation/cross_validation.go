package evaluation

import (
	"errors"
	"fmt"
	"github.com/feiyuluoye/Go-Model/internal/types"
	"math"
	"math/rand"
	"time"
)

// 定义模型接口，用于交叉验证
// 注意：这个接口需要与项目中现有的模型实现兼容

type Model interface {
	Fit(X [][]float64, y []float64) error
	Predict(X [][]float64) ([]float64, error)
}

// KFoldCrossValidation 执行k折交叉验证
func KFoldCrossValidation(model Model, X [][]float64, y []float64, k int) (map[string]float64, error) {
	if k <= 1 {
		return nil, errors.New("折数必须大于1")
	}

	if len(X) != len(y) {
		return nil, errors.New("特征矩阵和目标变量长度不匹配")
	}

	nSamples := len(X)
	if k > nSamples {
		return nil, errors.New("折数不能大于样本数量")
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

	// 存储每折的评估指标
	foldMetrics := make([]map[string]float64, k)

	// 执行k折交叉验证
	start := 0
	for fold := 0; fold < k; fold++ {
		// 计算当前折的大小
		size := foldSize
		if fold < extraSamples {
			size++
		}

		// 分割训练集和测试集
		testIndices := indices[start : start+size]
		trainIndices := make([]int, 0, nSamples-size)
		trainIndices = append(trainIndices, indices[:start]...)
		trainIndices = append(trainIndices, indices[start+size:]...)

		// 创建训练集
		trainX := make([][]float64, len(trainIndices))
		trainY := make([]float64, len(trainIndices))
		for i, idx := range trainIndices {
			trainX[i] = make([]float64, len(X[idx]))
			copy(trainX[i], X[idx])
			trainY[i] = y[idx]
		}

		// 创建测试集
		testX := make([][]float64, len(testIndices))
		testY := make([]float64, len(testIndices))
		for i, idx := range testIndices {
			testX[i] = make([]float64, len(X[idx]))
			copy(testX[i], X[idx])
			testY[i] = y[idx]
		}

		// 训练模型
		modelCopy := cloneModel(model)
		err := modelCopy.Fit(trainX, trainY)
		if err != nil {
			return nil, fmt.Errorf("折 %d 训练失败: %v", fold, err)
		}

		// 预测
		predictions, err := modelCopy.Predict(testX)
		if err != nil {
			return nil, fmt.Errorf("折 %d 预测失败: %v", fold, err)
		}

		// 评估
		metrics, err := EvaluateModel(testY, predictions)
		if err != nil {
			return nil, fmt.Errorf("折 %d 评估失败: %v", fold, err)
		}

		foldMetrics[fold] = metrics
		start += size
	}

	// 计算平均指标
	averageMetrics := make(map[string]float64)
	metricNames := []string{"r2", "mse", "rmse", "mae"}

	for _, name := range metricNames {
		var sum float64
		for _, metrics := range foldMetrics {
			sum += metrics[name]
		}
		averageMetrics[name] = sum / float64(k)
	}

	// 添加标准差
	for _, name := range metricNames {
		var sumSquaredDiff float64
		mean := averageMetrics[name]
		for _, metrics := range foldMetrics {
			diff := metrics[name] - mean
			sumSquaredDiff += diff * diff
		}
		stdDev := math.Sqrt(sumSquaredDiff / float64(k))
		averageMetrics[name+"_std"] = stdDev
	}

	return averageMetrics, nil
}

// LeaveOneOutCrossValidation 执行留一法交叉验证
func LeaveOneOutCrossValidation(model Model, X [][]float64, y []float64) (map[string]float64, error) {
	return KFoldCrossValidation(model, X, y, len(X))
}

// 为了简单起见，这里提供一个模型克隆函数
// 注意：在实际实现中，您可能需要根据具体的模型类型实现更复杂的克隆逻辑
func cloneModel(model Model) Model {
	// 这个实现是简化版的，在实际使用时需要根据具体模型类型进行扩展
	// 这里假设model是一个可以直接使用的模型实例
	return model
}

// CrossValidateDataset 使用Dataset进行交叉验证
func CrossValidateDataset(model Model, dataset *types.Dataset, k int) (map[string]float64, error) {
	if dataset == nil || !dataset.IsValid() {
		return nil, errors.New("无效的数据集")
	}

	return KFoldCrossValidation(model, dataset.Features, dataset.Target, k)
}
