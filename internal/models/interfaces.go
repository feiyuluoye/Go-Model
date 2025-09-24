package models

import (
	"gonum.org/v1/gonum/mat"
)

// Model 统一的模型接口
type Model interface {
	// Fit 训练模型
	Fit(X *mat.Dense, y *mat.VecDense) error
	// Predict 预测
	Predict(X *mat.Dense) *mat.VecDense
	// Score 计算R²分数
	Score(X *mat.Dense, y *mat.VecDense) float64
	// GetParameters 获取模型参数
	GetParameters() map[string]interface{}
	// GetModelType 获取模型类型
	GetModelType() string
}

// ModelInfo 模型信息
type ModelInfo struct {
	ModelType    string                 `json:"model_type"`
	Parameters   map[string]interface{} `json:"parameters"`
	IsTrained    bool                   `json:"is_trained"`
}

// ModelConfig 模型配置
type ModelConfig struct {
	ModelType  string                 `json:"model_type"`
	Parameters map[string]interface{} `json:"parameters"`
}

// TrainingResult 训练结果
type TrainingResult struct {
	ModelID       string             `json:"model_id"`
	TrainingScore float64            `json:"training_score"`
	Metrics       map[string]float64 `json:"metrics"`
	ModelInfo     *ModelInfo         `json:"model_info"`
}

// PredictionResult 预测结果
type PredictionResult struct {
	Predictions []float64          `json:"predictions"`
	ModelID     string             `json:"model_id"`
	Metrics     map[string]float64 `json:"metrics,omitempty"`
}

// EvaluationResult 评估结果
type EvaluationResult struct {
	Metrics map[string]float64 `json:"metrics"`
	ModelID string             `json:"model_id"`
}

// ErrorCode 错误码
type ErrorCode int

const (
	ErrorCodeInvalidInput ErrorCode = iota + 1
	ErrorCodeModelNotFound
	ErrorCodeTrainingFailed
	ErrorCodePredictionFailed
	ErrorCodeEvaluationFailed
)

// ModelError 模型错误
type ModelError struct {
	Code    ErrorCode              `json:"code"`
	Message string                 `json:"message"`
	Details map[string]interface{} `json:"details,omitempty"`
}

func (e ModelError) Error() string {
	return e.Message
}
