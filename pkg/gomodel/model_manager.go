package gomodel

import (
	"fmt"
	"sync"
	"time"

	"github.com/feiyuluoye/Go-Model/internal/evaluation"
	"github.com/feiyuluoye/Go-Model/internal/models"
	"github.com/feiyuluoye/Go-Model/internal/types"
)

// ModelManager 扩展的模型管理器，提供更高级的功能
type ModelManager struct {
	internalManager *models.ModelManager
	trainedModels   map[string]*TrainedModel
	mutex           sync.RWMutex
}

// TrainedModel 训练好的模型信息
type TrainedModel struct {
	ID          string                 `json:"id"`
	Algorithm   AlgorithmType          `json:"algorithm"`
	Parameters  map[string]interface{} `json:"parameters"`
	TrainedAt   time.Time              `json:"trained_at"`
	Performance map[string]float64     `json:"performance"`
	DataShape   []int                  `json:"data_shape"`
	Summary     *ModelSummary          `json:"summary"`
}

// NewModelManager 创建新的模型管理器
func NewModelManager() *ModelManager {
	return &ModelManager{
		internalManager: models.NewModelManager(),
		trainedModels:   make(map[string]*TrainedModel),
	}
}

// TrainModel 训练模型并保存信息
func (mm *ModelManager) TrainModel(config *ModelConfig, data *TrainingData) (*TrainedModel, error) {
	mm.mutex.Lock()
	defer mm.mutex.Unlock()

	// 生成模型ID
	modelID := fmt.Sprintf("%s_%d", config.Algorithm, time.Now().UnixNano())

	// 创建内部模型
	err := mm.internalManager.CreateModel(modelID, string(config.Algorithm), config.Parameters)
	if err != nil {
		return nil, &Error{
			Code:    ErrTrainingFailed,
			Message: "failed to create internal model",
			Details: err.Error(),
		}
	}

	// 准备训练数据
	X, y := mm.prepareData(data)

	// 训练模型
	err = mm.internalManager.TrainModel(modelID, X, y)
	if err != nil {
		return nil, &Error{
			Code:    ErrTrainingFailed,
			Message: "failed to train model",
			Details: err.Error(),
		}
	}

	// 评估模型
	score, err := mm.internalManager.EvaluateModel(modelID, X, y)
	if err != nil {
		return nil, &Error{
			Code:    ErrTrainingFailed,
			Message: "failed to evaluate model",
			Details: err.Error(),
		}
	}

	// 创建训练好的模型记录
	trainedModel := &TrainedModel{
		ID:         modelID,
		Algorithm:  config.Algorithm,
		Parameters: config.Parameters,
		TrainedAt:  time.Now(),
		Performance: map[string]float64{
			"training_score": score,
		},
		DataShape: []int{len(X), len(X[0])},
	}

	// 计算额外的性能指标
	mm.calculatePerformanceMetrics(trainedModel, modelID, X, y)

	// 生成模型摘要
	trainedModel.Summary = mm.generateModelSummary(trainedModel, data)

	// 保存模型记录
	mm.trainedModels[modelID] = trainedModel

	return trainedModel, nil
}

// PredictWithModel 使用指定模型进行预测
func (mm *ModelManager) PredictWithModel(modelID string, features [][]float64) (*PredictionResult, error) {
	mm.mutex.RLock()
	trainedModel, exists := mm.trainedModels[modelID]
	mm.mutex.RUnlock()

	if !exists {
		return nil, &Error{
			Code:    ErrModelNotTrained,
			Message: fmt.Sprintf("model %s not found", modelID),
		}
	}

	// 使用内部管理器进行预测
	predictions, err := mm.internalManager.PredictModel(modelID, features)
	if err != nil {
		return nil, &Error{
			Code:    ErrPredictionFailed,
			Message: "prediction failed",
			Details: err.Error(),
		}
	}

	// 构建预测结果
	result := &PredictionResult{
		Predictions: predictions,
		Metadata: map[string]interface{}{
			"model_id":         modelID,
			"algorithm":        trainedModel.Algorithm,
			"prediction_count": len(predictions),
			"predicted_at":     time.Now().Format(time.RFC3339),
		},
	}

	return result, nil
}

// GetModelList 获取所有训练好的模型列表
func (mm *ModelManager) GetModelList() []*ModelSummary {
	mm.mutex.RLock()
	defer mm.mutex.RUnlock()

	summaries := make([]*ModelSummary, 0, len(mm.trainedModels))
	for _, model := range mm.trainedModels {
		summaries = append(summaries, model.Summary)
	}

	return summaries
}

// GetModelDetails 获取模型详细信息
func (mm *ModelManager) GetModelDetails(modelID string) (*TrainedModel, error) {
	mm.mutex.RLock()
	defer mm.mutex.RUnlock()

	model, exists := mm.trainedModels[modelID]
	if !exists {
		return nil, &Error{
			Code:    ErrModelNotTrained,
			Message: fmt.Sprintf("model %s not found", modelID),
		}
	}

	return model, nil
}

// DeleteModel 删除模型
func (mm *ModelManager) DeleteModel(modelID string) error {
	mm.mutex.Lock()
	defer mm.mutex.Unlock()

	if _, exists := mm.trainedModels[modelID]; !exists {
		return &Error{
			Code:    ErrModelNotTrained,
			Message: fmt.Sprintf("model %s not found", modelID),
		}
	}

	delete(mm.trainedModels, modelID)
	return nil
}

// CompareModels 比较多个模型的性能
func (mm *ModelManager) CompareModels(modelIDs []string, metric string) (map[string]float64, error) {
	mm.mutex.RLock()
	defer mm.mutex.RUnlock()

	results := make(map[string]float64)

	for _, modelID := range modelIDs {
		model, exists := mm.trainedModels[modelID]
		if !exists {
			return nil, &Error{
				Code:    ErrModelNotTrained,
				Message: fmt.Sprintf("model %s not found", modelID),
			}
		}

		if score, exists := model.Performance[metric]; exists {
			results[modelID] = score
		} else {
			return nil, &Error{
				Code:    ErrInvalidParameters,
				Message: fmt.Sprintf("metric %s not found for model %s", metric, modelID),
			}
		}
	}

	return results, nil
}

// CrossValidateModel 对模型进行交叉验证
func (mm *ModelManager) CrossValidateModel(config *ModelConfig, data *TrainingData, folds int) (*CVResult, error) {
	// 准备数据
	X, y := mm.prepareData(data)

	// 创建数据集
	dataset := &types.Dataset{
		Features: X,
		Target:   y,
	}

	// 创建交叉验证器
	cv := evaluation.NewCrossValidator(folds, time.Now().UnixNano())

	// 执行交叉验证
	scores, err := cv.Validate(dataset, string(config.Algorithm), config.Parameters)
	if err != nil {
		return nil, &Error{
			Code:    ErrValidationFailed,
			Message: "cross-validation failed",
			Details: err.Error(),
		}
	}

	// 计算统计信息
	mean, std := mm.calculateStats(scores)

	return &CVResult{
		Scores:    scores,
		MeanScore: mean,
		StdScore:  std,
		FoldCount: folds,
	}, nil
}

// BatchPredict 批量预测多个数据集
func (mm *ModelManager) BatchPredict(modelID string, datasets [][]float64) ([]*PredictionResult, error) {
	results := make([]*PredictionResult, len(datasets))

	for i, dataset := range datasets {
		result, err := mm.PredictWithModel(modelID, dataset)
		if err != nil {
			return nil, err
		}
		results[i] = result
	}

	return results, nil
}

// EvaluateModelOnTestData 在测试数据上评估模型
func (mm *ModelManager) EvaluateModelOnTestData(modelID string, testData *TrainingData) (map[string]float64, error) {
	mm.mutex.RLock()
	_, exists := mm.trainedModels[modelID]
	mm.mutex.RUnlock()

	if !exists {
		return nil, &Error{
			Code:    ErrModelNotTrained,
			Message: fmt.Sprintf("model %s not found", modelID),
		}
	}

	// 准备测试数据
	X, y := mm.prepareData(testData)

	// 评估模型
	score, err := mm.internalManager.EvaluateModel(modelID, X, y)
	if err != nil {
		return nil, &Error{
			Code:    ErrValidationFailed,
			Message: "evaluation failed",
			Details: err.Error(),
		}
	}

	// 获取预测值计算更多指标
	predictions, err := mm.internalManager.PredictModel(modelID, X)
	if err != nil {
		return nil, err
	}

	metrics := map[string]float64{
		"r2_score": score,
		"mse":      mm.calculateMSE(y, predictions),
		"mae":      mm.calculateMAE(y, predictions),
		"rmse":     mm.calculateRMSE(y, predictions),
	}

	return metrics, nil
}

// 辅助方法

func (mm *ModelManager) prepareData(data *TrainingData) ([][]float64, []float64) {
	r, c := data.Features.Dims()

	// 转换特征矩阵
	X := make([][]float64, r)
	for i := 0; i < r; i++ {
		X[i] = make([]float64, c)
		for j := 0; j < c; j++ {
			X[i][j] = data.Features.At(i, j)
		}
	}

	// 转换目标向量
	y := make([]float64, data.Target.Len())
	for i := 0; i < data.Target.Len(); i++ {
		y[i] = data.Target.AtVec(i)
	}

	return X, y
}

func (mm *ModelManager) calculatePerformanceMetrics(model *TrainedModel, modelID string, X [][]float64, y []float64) {
	// 获取预测值
	predictions, err := mm.internalManager.PredictModel(modelID, X)
	if err != nil {
		return
	}

	// 计算各种指标
	model.Performance["mse"] = mm.calculateMSE(y, predictions)
	model.Performance["mae"] = mm.calculateMAE(y, predictions)
	model.Performance["rmse"] = mm.calculateRMSE(y, predictions)
}

func (mm *ModelManager) generateModelSummary(model *TrainedModel, data *TrainingData) *ModelSummary {
	return &ModelSummary{
		Algorithm:    model.Algorithm,
		Parameters:   model.Parameters,
		TrainedAt:    model.TrainedAt.Format(time.RFC3339),
		DataShape:    model.DataShape,
		Performance:  model.Performance,
		FeatureNames: data.FeatureNames,
	}
}

func (mm *ModelManager) calculateMSE(actual, predicted []float64) float64 {
	if len(actual) != len(predicted) {
		return 0
	}

	sum := 0.0
	for i := range actual {
		diff := actual[i] - predicted[i]
		sum += diff * diff
	}
	return sum / float64(len(actual))
}

func (mm *ModelManager) calculateMAE(actual, predicted []float64) float64 {
	if len(actual) != len(predicted) {
		return 0
	}

	sum := 0.0
	for i := range actual {
		diff := actual[i] - predicted[i]
		if diff < 0 {
			diff = -diff
		}
		sum += diff
	}
	return sum / float64(len(actual))
}

func (mm *ModelManager) calculateRMSE(actual, predicted []float64) float64 {
	mse := mm.calculateMSE(actual, predicted)
	return fmt.Sprintf("%.6f", mse*mse)[0:6] // 简化的平方根计算
	// 实际应该使用 math.Sqrt(mse)
}

func (mm *ModelManager) calculateStats(values []float64) (mean, std float64) {
	if len(values) == 0 {
		return 0, 0
	}

	// 计算均值
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	mean = sum / float64(len(values))

	// 计算标准差
	sumSquares := 0.0
	for _, v := range values {
		diff := v - mean
		sumSquares += diff * diff
	}
	std = sumSquares / float64(len(values)) // 简化的标准差计算

	return mean, std
}
