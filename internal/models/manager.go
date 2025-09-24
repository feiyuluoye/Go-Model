package models

import (
	"fmt"
	"sync"

	"gonum.org/v1/gonum/mat"
)

// ModelManager 模型管理器
type ModelManager struct {
	models map[string]Model
	mu     sync.RWMutex
	nextID int
}

// NewModelManager 创建新的模型管理器
func NewModelManager() *ModelManager {
	return &ModelManager{
		models: make(map[string]Model),
		nextID: 1,
	}
}

// CreateModel 创建模型
func (mm *ModelManager) CreateModel(config *ModelConfig) (Model, error) {
	switch config.ModelType {
	case "ols":
		return NewOLS(), nil
	case "ridge":
		alpha := 1.0
		if param, ok := config.Parameters["alpha"]; ok {
			if a, ok := param.(float64); ok {
				alpha = a
			}
		}
		return NewRidge(alpha), nil
	case "lasso":
		alpha := 1.0
		if param, ok := config.Parameters["alpha"]; ok {
			if a, ok := param.(float64); ok {
				alpha = a
			}
		}
		return NewLasso(alpha), nil
	case "logistic":
		return NewLogistic(), nil
	case "pls":
		numComponents := 2
		if param, ok := config.Parameters["num_components"]; ok {
			if n, ok := param.(int); ok {
				numComponents = n
			}
		}
		return NewPLS(numComponents), nil
	case "polynomial":
		degree := 2
		if param, ok := config.Parameters["degree"]; ok {
			if d, ok := param.(int); ok {
				degree = d
			}
		}
		return NewPolynomial(degree), nil
	case "exponential":
		return NewExponential(), nil
	case "logarithmic":
		return NewLogarithmic(), nil
	case "power":
		return NewPower(), nil
	default:
		return nil, ModelError{
			Code:    ErrorCodeInvalidInput,
			Message: fmt.Sprintf("不支持的模型类型: %s", config.ModelType),
			Details: map[string]interface{}{
				"supported_models": []string{"ols", "ridge", "lasso", "logistic", "pls", "polynomial", "exponential", "logarithmic", "power"},
			},
		}
	}
}

// TrainModel 训练模型
func (mm *ModelManager) TrainModel(config *ModelConfig, X *mat.Dense, y *mat.VecDense) (*TrainingResult, error) {
	// 创建模型
	model, err := mm.CreateModel(config)
	if err != nil {
		return nil, err
	}

	// 训练模型
	if err := model.Fit(X, y); err != nil {
		return nil, ModelError{
			Code:    ErrorCodeTrainingFailed,
			Message: fmt.Sprintf("模型训练失败: %v", err),
			Details: map[string]interface{}{
				"model_type": config.ModelType,
			},
		}
	}

	// 计算训练得分
	score := model.Score(X, y)

	// 存储模型
	modelID := mm.addModel(model)

	// 准备结果
	result := &TrainingResult{
		ModelID:       modelID,
		TrainingScore: score,
		Metrics: map[string]float64{
			"r2": score,
		},
		ModelInfo: &ModelInfo{
			ModelType:  model.GetModelType(),
			Parameters: model.GetParameters(),
			IsTrained:  true,
		},
	}

	return result, nil
}

// Predict 使用模型进行预测
func (mm *ModelManager) Predict(modelID string, X *mat.Dense) (*PredictionResult, error) {
	model, exists := mm.getModel(modelID)
	if !exists {
		return nil, ModelError{
			Code:    ErrorCodeModelNotFound,
			Message: fmt.Sprintf("模型不存在: %s", modelID),
			Details: map[string]interface{}{
				"model_id": modelID,
			},
		}
	}

	predictions := model.Predict(X)
	
	// 转换为slice
	n, _ := predictions.Dims()
	predSlice := make([]float64, n)
	for i := 0; i < n; i++ {
		predSlice[i] = predictions.At(i, 0)
	}

	return &PredictionResult{
		Predictions: predSlice,
		ModelID:     modelID,
	}, nil
}

// Evaluate 评估模型
func (mm *ModelManager) Evaluate(modelID string, X *mat.Dense, y *mat.VecDense) (*EvaluationResult, error) {
	model, exists := mm.getModel(modelID)
	if !exists {
		return nil, ModelError{
			Code:    ErrorCodeModelNotFound,
			Message: fmt.Sprintf("模型不存在: %s", modelID),
			Details: map[string]interface{}{
				"model_id": modelID,
			},
		}
	}

	score := model.Score(X, y)

	return &EvaluationResult{
		Metrics: map[string]float64{
			"r2": score,
		},
		ModelID: modelID,
	}, nil
}

// GetModelInfo 获取模型信息
func (mm *ModelManager) GetModelInfo(modelID string) (*ModelInfo, error) {
	model, exists := mm.getModel(modelID)
	if !exists {
		return nil, ModelError{
			Code:    ErrorCodeModelNotFound,
			Message: fmt.Sprintf("模型不存在: %s", modelID),
			Details: map[string]interface{}{
				"model_id": modelID,
			},
		}
	}

	return &ModelInfo{
		ModelType:  model.GetModelType(),
		Parameters: model.GetParameters(),
		IsTrained:  true,
	}, nil
}

// 内部方法：添加模型
func (mm *ModelManager) addModel(model Model) string {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	modelID := fmt.Sprintf("model_%d", mm.nextID)
	mm.models[modelID] = model
	mm.nextID++
	return modelID
}

// 内部方法：获取模型
func (mm *ModelManager) getModel(modelID string) (Model, bool) {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	model, exists := mm.models[modelID]
	return model, exists
}
