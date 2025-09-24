package gomodel

import (
	"fmt"
	"time"

	"github.com/feiyuluoye/Go-Model/internal/data"
	"github.com/feiyuluoye/Go-Model/internal/evaluation"
	"github.com/feiyuluoye/Go-Model/internal/models"
	"github.com/feiyuluoye/Go-Model/internal/types"
	"gonum.org/v1/gonum/mat"
)

// Client 是Go-Model库的主要客户端接口
type Client struct {
	manager *models.ModelManager
	config  *ClientConfig
}

// ClientConfig 客户端配置
type ClientConfig struct {
	DefaultValidation *ValidationConfig `json:"default_validation"`
	RandomSeed        int64             `json:"random_seed"`
	Verbose           bool              `json:"verbose"`
}

// NewClient 创建新的客户端实例
func NewClient(config *ClientConfig) *Client {
	if config == nil {
		config = &ClientConfig{
			DefaultValidation: &ValidationConfig{
				Method:     "holdout",
				TestSize:   0.2,
				RandomSeed: time.Now().UnixNano(),
			},
			RandomSeed: time.Now().UnixNano(),
			Verbose:    false,
		}
	}

	return &Client{
		manager: models.NewModelManager(),
		config:  config,
	}
}

// Train 训练模型
func (c *Client) Train(data *TrainingData, config *ModelConfig) (*ModelResult, error) {
	if data == nil || config == nil {
		return nil, &Error{
			Code:    ErrInvalidParameters,
			Message: "training data and model config cannot be nil",
		}
	}

	// 验证算法类型
	if !c.isValidAlgorithm(config.Algorithm) {
		return nil, &Error{
			Code:    ErrInvalidAlgorithm,
			Message: fmt.Sprintf("unsupported algorithm: %s", config.Algorithm),
		}
	}

	// 验证数据
	if err := c.validateData(data); err != nil {
		return nil, err
	}

	// 创建模型
	modelID := fmt.Sprintf("%s_%d", config.Algorithm, time.Now().UnixNano())
	err := c.manager.CreateModel(modelID, string(config.Algorithm), config.Parameters)
	if err != nil {
		return nil, &Error{
			Code:    ErrTrainingFailed,
			Message: "failed to create model",
			Details: err.Error(),
		}
	}

	// 准备训练数据
	X, y := c.prepareTrainingData(data)

	// 执行训练
	err = c.manager.TrainModel(modelID, X, y)
	if err != nil {
		return nil, &Error{
			Code:    ErrTrainingFailed,
			Message: "failed to train model",
			Details: err.Error(),
		}
	}

	// 计算训练分数
	trainingScore, err := c.manager.EvaluateModel(modelID, X, y)
	if err != nil {
		return nil, &Error{
			Code:    ErrTrainingFailed,
			Message: "failed to evaluate training score",
			Details: err.Error(),
		}
	}

	// 构建结果
	result := &ModelResult{
		Algorithm:     config.Algorithm,
		Parameters:    config.Parameters,
		TrainingScore: trainingScore,
		Metrics:       make(map[string]float64),
		ModelInfo:     make(map[string]interface{}),
	}

	// 计算额外指标
	c.calculateMetrics(result, modelID, X, y, config.LossFunction)

	// 执行验证（如果配置了）
	if config.Validation != nil {
		err = c.performValidation(result, modelID, data, config)
		if err != nil {
			return nil, err
		}
	}

	// 获取模型信息
	modelInfo, err := c.manager.GetModelInfo(modelID)
	if err == nil {
		result.ModelInfo["model_type"] = modelInfo.ModelType
		result.ModelInfo["created_at"] = modelInfo.CreatedAt
		result.ModelInfo["trained"] = modelInfo.Trained
	}

	return result, nil
}

// Predict 使用训练好的模型进行预测
func (c *Client) Predict(modelID string, features *mat.Dense) (*PredictionResult, error) {
	if features == nil {
		return nil, &Error{
			Code:    ErrInvalidData,
			Message: "features cannot be nil",
		}
	}

	// 转换数据格式
	r, c_count := features.Dims()
	X := make([][]float64, r)
	for i := 0; i < r; i++ {
		X[i] = make([]float64, c_count)
		for j := 0; j < c_count; j++ {
			X[i][j] = features.At(i, j)
		}
	}

	// 执行预测
	predictions, err := c.manager.PredictModel(modelID, X)
	if err != nil {
		return nil, &Error{
			Code:    ErrPredictionFailed,
			Message: "failed to make predictions",
			Details: err.Error(),
		}
	}

	result := &PredictionResult{
		Predictions: predictions,
		Metadata:    make(map[string]interface{}),
	}

	// 添加元数据
	result.Metadata["model_id"] = modelID
	result.Metadata["prediction_count"] = len(predictions)
	result.Metadata["predicted_at"] = time.Now().Format(time.RFC3339)

	return result, nil
}

// TrainAndPredict 训练模型并立即进行预测
func (c *Client) TrainAndPredict(trainData *TrainingData, testFeatures *mat.Dense, config *ModelConfig) (*ModelResult, *PredictionResult, error) {
	// 训练模型
	result, err := c.Train(trainData, config)
	if err != nil {
		return nil, nil, err
	}

	// 生成临时模型ID进行预测
	modelID := fmt.Sprintf("%s_%d", config.Algorithm, time.Now().UnixNano())
	
	// 重新创建和训练模型用于预测
	err = c.manager.CreateModel(modelID, string(config.Algorithm), config.Parameters)
	if err != nil {
		return result, nil, err
	}

	X, y := c.prepareTrainingData(trainData)
	err = c.manager.TrainModel(modelID, X, y)
	if err != nil {
		return result, nil, err
	}

	// 进行预测
	predictions, err := c.Predict(modelID, testFeatures)
	if err != nil {
		return result, nil, err
	}

	return result, predictions, nil
}

// GetSupportedAlgorithms 获取支持的算法列表
func (c *Client) GetSupportedAlgorithms() []AlgorithmType {
	return []AlgorithmType{
		OLS, Ridge, Lasso, Logistic, PLS,
		Polynomial, Exponential, Logarithmic, Power,
	}
}

// ValidateConfig 验证模型配置
func (c *Client) ValidateConfig(config *ModelConfig) error {
	if config == nil {
		return &Error{
			Code:    ErrInvalidParameters,
			Message: "model config cannot be nil",
		}
	}

	if !c.isValidAlgorithm(config.Algorithm) {
		return &Error{
			Code:    ErrInvalidAlgorithm,
			Message: fmt.Sprintf("unsupported algorithm: %s", config.Algorithm),
		}
	}

	// 验证算法特定参数
	return c.validateAlgorithmParameters(config.Algorithm, config.Parameters)
}

// 辅助方法

func (c *Client) isValidAlgorithm(algorithm AlgorithmType) bool {
	supportedAlgorithms := c.GetSupportedAlgorithms()
	for _, supported := range supportedAlgorithms {
		if algorithm == supported {
			return true
		}
	}
	return false
}

func (c *Client) validateData(data *TrainingData) error {
	if data.Features == nil || data.Target == nil {
		return &Error{
			Code:    ErrInvalidData,
			Message: "features and target cannot be nil",
		}
	}

	r, _ := data.Features.Dims()
	targetLen := data.Target.Len()

	if r != targetLen {
		return &Error{
			Code:    ErrInvalidData,
			Message: fmt.Sprintf("feature rows (%d) must match target length (%d)", r, targetLen),
		}
	}

	return nil
}

func (c *Client) prepareTrainingData(data *TrainingData) ([][]float64, []float64) {
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

func (c *Client) calculateMetrics(result *ModelResult, modelID string, X [][]float64, y []float64, lossFunc LossFunction) {
	// 获取预测值
	predictions, err := c.manager.PredictModel(modelID, X)
	if err != nil {
		return
	}

	// 计算各种指标
	switch lossFunc {
	case MSE:
		result.Metrics["mse"] = c.calculateMSE(y, predictions)
	case MAE:
		result.Metrics["mae"] = c.calculateMAE(y, predictions)
	case RMSE:
		result.Metrics["rmse"] = c.calculateRMSE(y, predictions)
	case R2:
		result.Metrics["r2"] = result.TrainingScore // R2 已经在TrainingScore中
	}

	// 总是计算R2和RMSE作为基本指标
	result.Metrics["r2"] = result.TrainingScore
	result.Metrics["rmse"] = c.calculateRMSE(y, predictions)
}

func (c *Client) performValidation(result *ModelResult, modelID string, data *TrainingData, config *ModelConfig) error {
	validation := config.Validation
	if validation == nil {
		return nil
	}

	switch validation.Method {
	case "holdout":
		return c.performHoldoutValidation(result, data, config, validation)
	case "kfold":
		return c.performKFoldValidation(result, data, config, validation)
	default:
		return &Error{
			Code:    ErrValidationFailed,
			Message: fmt.Sprintf("unsupported validation method: %s", validation.Method),
		}
	}
}

func (c *Client) performHoldoutValidation(result *ModelResult, data *TrainingData, config *ModelConfig, validation *ValidationConfig) error {
	// 分割数据
	X, y := c.prepareTrainingData(data)
	
	// 这里应该实现数据分割逻辑
	// 为简化，暂时使用全部数据作为验证集
	testScore := result.TrainingScore
	result.ValidationScore = &testScore
	
	return nil
}

func (c *Client) performKFoldValidation(result *ModelResult, data *TrainingData, config *ModelConfig, validation *ValidationConfig) error {
	// 实现K折交叉验证
	X, y := c.prepareTrainingData(data)
	
	// 转换为internal包需要的格式
	dataset := &types.Dataset{
		Features: X,
		Target:   y,
	}

	// 创建交叉验证器
	cv := evaluation.NewCrossValidator(validation.KFolds, validation.RandomSeed)
	
	// 执行交叉验证
	scores, err := cv.Validate(dataset, string(config.Algorithm), config.Parameters)
	if err != nil {
		return &Error{
			Code:    ErrValidationFailed,
			Message: "cross-validation failed",
			Details: err.Error(),
		}
	}

	// 计算统计信息
	meanScore, stdScore := c.calculateStats(scores)
	
	result.CrossValidation = &CVResult{
		Scores:    scores,
		MeanScore: meanScore,
		StdScore:  stdScore,
		FoldCount: validation.KFolds,
	}

	result.ValidationScore = &meanScore

	return nil
}

func (c *Client) validateAlgorithmParameters(algorithm AlgorithmType, params map[string]interface{}) error {
	// 根据不同算法验证参数
	switch algorithm {
	case Ridge, Lasso:
		if lambda, ok := params["lambda"]; ok {
			if lambdaVal, ok := lambda.(float64); ok {
				if lambdaVal < 0 {
					return &Error{
						Code:    ErrInvalidParameters,
						Message: "lambda must be non-negative",
					}
				}
			}
		}
	case Polynomial:
		if degree, ok := params["degree"]; ok {
			if degreeVal, ok := degree.(int); ok {
				if degreeVal < 1 || degreeVal > 10 {
					return &Error{
						Code:    ErrInvalidParameters,
						Message: "polynomial degree must be between 1 and 10",
					}
				}
			}
		}
	case PLS:
		if components, ok := params["components"]; ok {
			if compVal, ok := components.(int); ok {
				if compVal < 1 {
					return &Error{
						Code:    ErrInvalidParameters,
						Message: "PLS components must be positive",
					}
				}
			}
		}
	}

	return nil
}

// 统计计算辅助方法

func (c *Client) calculateMSE(actual, predicted []float64) float64 {
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

func (c *Client) calculateMAE(actual, predicted []float64) float64 {
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

func (c *Client) calculateRMSE(actual, predicted []float64) float64 {
	mse := c.calculateMSE(actual, predicted)
	return mat.Sqrt(mse)
}

func (c *Client) calculateStats(values []float64) (mean, std float64) {
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
	std = mat.Sqrt(sumSquares / float64(len(values)))

	return mean, std
}
