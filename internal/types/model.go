package types

// ModelConfig 模型配置参数
type ModelConfig struct {
	ModelType string            `json:"model_type" yaml:"model_type"`
	Params    map[string]string `json:"params" yaml:"params"`
}

// NewModelConfig 创建新的模型配置
func NewModelConfig(modelType string, params map[string]string) *ModelConfig {
	if params == nil {
		params = make(map[string]string)
	}
	return &ModelConfig{
		ModelType: modelType,
		Params:    params,
	}
}

// TrainingResult 训练结果
type TrainingResult struct {
	ModelID       string             `json:"model_id"`
	TrainingScore float64            `json:"training_score"`
	Metrics       map[string]float64 `json:"metrics"`
	Status        string             `json:"status"`
	Message       string             `json:"message"`
}

// PredictionResult 预测结果
type PredictionResult struct {
	Predictions []float64 `json:"predictions"`
	Status      string    `json:"status"`
	Message     string    `json:"message"`
}

// EvaluationResult 评估结果
type EvaluationResult struct {
	Metrics map[string]float64 `json:"metrics"`
	Status  string             `json:"status"`
	Message string             `json:"message"`
}

// ModelInfo 模型信息
type ModelInfo struct {
	ModelType    string            `json:"model_type"`
	Parameters   map[string]string `json:"parameters"`
	Coefficients []float64         `json:"coefficients"`
	Intercept    float64           `json:"intercept"`
	Status       string            `json:"status"`
	Message      string            `json:"message"`
}
