package gomodel

import (
	"gonum.org/v1/gonum/mat"
)

// AlgorithmType 定义支持的算法类型
type AlgorithmType string

const (
	// 线性模型
	OLS       AlgorithmType = "ols"
	Ridge     AlgorithmType = "ridge"
	Lasso     AlgorithmType = "lasso"
	Logistic  AlgorithmType = "logistic"
	PLS       AlgorithmType = "pls"
	
	// 非线性模型
	Polynomial  AlgorithmType = "polynomial"
	Exponential AlgorithmType = "exponential"
	Logarithmic AlgorithmType = "logarithmic"
	Power       AlgorithmType = "power"
)

// LossFunction 定义损失函数类型
type LossFunction string

const (
	MSE      LossFunction = "mse"      // 均方误差
	MAE      LossFunction = "mae"      // 平均绝对误差
	RMSE     LossFunction = "rmse"     // 均方根误差
	R2       LossFunction = "r2"       // 决定系数
	Accuracy LossFunction = "accuracy" // 准确率（分类）
	LogLoss  LossFunction = "logloss"  // 对数损失（分类）
)

// ModelConfig 模型配置结构
type ModelConfig struct {
	Algorithm    AlgorithmType            `json:"algorithm"`
	Parameters   map[string]interface{}   `json:"parameters"`
	LossFunction LossFunction             `json:"loss_function"`
	Validation   *ValidationConfig        `json:"validation,omitempty"`
}

// ValidationConfig 验证配置
type ValidationConfig struct {
	Method     string  `json:"method"`      // "holdout", "kfold", "none"
	TestSize   float64 `json:"test_size"`   // 测试集比例 (0-1)
	KFolds     int     `json:"k_folds"`     // K折交叉验证的K值
	RandomSeed int64   `json:"random_seed"` // 随机种子
}

// TrainingData 训练数据结构
type TrainingData struct {
	Features *mat.Dense `json:"-"`        // 特征矩阵
	Target   *mat.VecDense `json:"-"`     // 目标变量
	FeatureNames []string `json:"feature_names,omitempty"`
	TargetName   string   `json:"target_name,omitempty"`
}

// PredictionResult 预测结果
type PredictionResult struct {
	Predictions    []float64              `json:"predictions"`
	Probabilities  [][]float64            `json:"probabilities,omitempty"` // 分类概率
	Confidence     []float64              `json:"confidence,omitempty"`    // 置信度
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
}

// ModelResult 模型训练和评估结果
type ModelResult struct {
	Algorithm      AlgorithmType          `json:"algorithm"`
	Parameters     map[string]interface{} `json:"parameters"`
	TrainingScore  float64                `json:"training_score"`
	ValidationScore *float64              `json:"validation_score,omitempty"`
	TestScore      *float64               `json:"test_score,omitempty"`
	Metrics        map[string]float64     `json:"metrics"`
	ModelInfo      map[string]interface{} `json:"model_info"`
	CrossValidation *CVResult             `json:"cross_validation,omitempty"`
}

// CVResult 交叉验证结果
type CVResult struct {
	Scores     []float64 `json:"scores"`
	MeanScore  float64   `json:"mean_score"`
	StdScore   float64   `json:"std_score"`
	FoldCount  int       `json:"fold_count"`
}

// ModelSummary 模型摘要信息
type ModelSummary struct {
	Algorithm     AlgorithmType          `json:"algorithm"`
	Parameters    map[string]interface{} `json:"parameters"`
	TrainedAt     string                 `json:"trained_at"`
	DataShape     []int                  `json:"data_shape"` // [samples, features]
	Performance   map[string]float64     `json:"performance"`
	FeatureNames  []string               `json:"feature_names,omitempty"`
}

// DataPreprocessConfig 数据预处理配置
type DataPreprocessConfig struct {
	Normalize     bool    `json:"normalize"`      // 标准化
	Scale         bool    `json:"scale"`          // 缩放到[0,1]
	HandleMissing string  `json:"handle_missing"` // "drop", "mean", "median", "mode"
	OutlierMethod string  `json:"outlier_method"` // "iqr", "zscore", "none"
	OutlierThreshold float64 `json:"outlier_threshold"`
}

// Error 自定义错误类型
type Error struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Details string `json:"details,omitempty"`
}

func (e *Error) Error() string {
	if e.Details != "" {
		return e.Code + ": " + e.Message + " (" + e.Details + ")"
	}
	return e.Code + ": " + e.Message
}

// 常见错误代码
const (
	ErrInvalidAlgorithm   = "INVALID_ALGORITHM"
	ErrInvalidParameters  = "INVALID_PARAMETERS"
	ErrInvalidData        = "INVALID_DATA"
	ErrTrainingFailed     = "TRAINING_FAILED"
	ErrPredictionFailed   = "PREDICTION_FAILED"
	ErrValidationFailed   = "VALIDATION_FAILED"
	ErrModelNotTrained    = "MODEL_NOT_TRAINED"
)
