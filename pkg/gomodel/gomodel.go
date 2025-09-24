// Package gomodel provides a high-level API for machine learning algorithms
// 
// This package offers a unified interface for training, evaluating, and using
// various regression and classification models. It integrates data processing,
// model training, validation, and prediction capabilities.
//
// Example usage:
//
//	// Create a client
//	client := gomodel.NewClient(nil)
//	
//	// Prepare data
//	dataUtils := gomodel.NewDataUtils(0)
//	data, _ := dataUtils.CreateFromArrays(features, target, nil, "target")
//	
//	// Configure model
//	config := &gomodel.ModelConfig{
//		Algorithm: gomodel.OLS,
//		Parameters: map[string]interface{}{},
//		LossFunction: gomodel.R2,
//	}
//	
//	// Train model
//	result, _ := client.Train(data, config)
//	fmt.Printf("Training R²: %.4f\n", result.TrainingScore)
//
package gomodel

import (
	"fmt"
)

// Version returns the current version of the gomodel package
func Version() string {
	return "1.0.0"
}

// GetDefaultConfig returns a default model configuration for the specified algorithm
func GetDefaultConfig(algorithm AlgorithmType) *ModelConfig {
	config := &ModelConfig{
		Algorithm:    algorithm,
		Parameters:   make(map[string]interface{}),
		LossFunction: R2,
	}

	// Set algorithm-specific default parameters
	switch algorithm {
	case Ridge:
		config.Parameters["lambda"] = 1.0
	case Lasso:
		config.Parameters["lambda"] = 1.0
		config.Parameters["max_iterations"] = 1000
		config.Parameters["tolerance"] = 1e-6
	case Logistic:
		config.Parameters["learning_rate"] = 0.01
		config.Parameters["max_iterations"] = 1000
		config.Parameters["tolerance"] = 1e-6
		config.LossFunction = Accuracy
	case PLS:
		config.Parameters["components"] = 2
	case Polynomial:
		config.Parameters["degree"] = 2
	case Exponential:
		config.Parameters["max_iterations"] = 1000
		config.Parameters["tolerance"] = 1e-6
	case Logarithmic:
		config.Parameters["max_iterations"] = 1000
		config.Parameters["tolerance"] = 1e-6
	case Power:
		config.Parameters["max_iterations"] = 1000
		config.Parameters["tolerance"] = 1e-6
	}

	return config
}

// GetDefaultValidationConfig returns a default validation configuration
func GetDefaultValidationConfig() *ValidationConfig {
	return &ValidationConfig{
		Method:     "holdout",
		TestSize:   0.2,
		KFolds:     5,
		RandomSeed: 42,
	}
}

// QuickTrain provides a simplified interface for quick model training
// It uses default parameters and validation settings
func QuickTrain(features [][]float64, target []float64, algorithm AlgorithmType) (*ModelResult, error) {
	// Create client
	client := NewClient(nil)
	
	// Create data utils
	dataUtils := NewDataUtils(0)
	
	// Prepare data
	data, err := dataUtils.CreateFromArrays(features, target, nil, "target")
	if err != nil {
		return nil, err
	}
	
	// Get default config
	config := GetDefaultConfig(algorithm)
	config.Validation = GetDefaultValidationConfig()
	
	// Train model
	return client.Train(data, config)
}

// QuickPredict provides a simplified interface for quick prediction
// It trains a model and immediately makes predictions
func QuickPredict(trainFeatures [][]float64, trainTarget []float64, testFeatures [][]float64, algorithm AlgorithmType) ([]float64, error) {
	// Create client
	client := NewClient(nil)
	
	// Create data utils
	dataUtils := NewDataUtils(0)
	
	// Prepare training data
	trainData, err := dataUtils.CreateFromArrays(trainFeatures, trainTarget, nil, "target")
	if err != nil {
		return nil, err
	}
	
	// Prepare test features
	r := len(testFeatures)
	c := len(testFeatures[0])
	testMatrix := make([]float64, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			testMatrix[i*c+j] = testFeatures[i][j]
		}
	}
	
	// Convert to gonum matrix
	testFeaturesMatrix := NewDenseFromSlice(r, c, testMatrix)
	
	// Get default config
	config := GetDefaultConfig(algorithm)
	
	// Train and predict
	_, predictions, err := client.TrainAndPredict(trainData, testFeaturesMatrix, config)
	if err != nil {
		return nil, err
	}
	
	return predictions.Predictions, nil
}

// ValidateAlgorithm checks if an algorithm is supported
func ValidateAlgorithm(algorithm AlgorithmType) error {
	supportedAlgorithms := []AlgorithmType{
		OLS, Ridge, Lasso, Logistic, PLS,
		Polynomial, Exponential, Logarithmic, Power,
	}
	
	for _, supported := range supportedAlgorithms {
		if algorithm == supported {
			return nil
		}
	}
	
	return &Error{
		Code:    ErrInvalidAlgorithm,
		Message: fmt.Sprintf("unsupported algorithm: %s", algorithm),
	}
}

// GetAlgorithmInfo returns information about a specific algorithm
func GetAlgorithmInfo(algorithm AlgorithmType) map[string]interface{} {
	info := map[string]interface{}{
		"algorithm": algorithm,
		"type":      "unknown",
		"description": "No description available",
		"parameters": []string{},
	}
	
	switch algorithm {
	case OLS:
		info["type"] = "linear_regression"
		info["description"] = "Ordinary Least Squares regression"
		info["parameters"] = []string{}
		
	case Ridge:
		info["type"] = "linear_regression"
		info["description"] = "Ridge regression with L2 regularization"
		info["parameters"] = []string{"lambda"}
		
	case Lasso:
		info["type"] = "linear_regression"
		info["description"] = "Lasso regression with L1 regularization"
		info["parameters"] = []string{"lambda", "max_iterations", "tolerance"}
		
	case Logistic:
		info["type"] = "classification"
		info["description"] = "Logistic regression for binary classification"
		info["parameters"] = []string{"learning_rate", "max_iterations", "tolerance"}
		
	case PLS:
		info["type"] = "linear_regression"
		info["description"] = "Partial Least Squares regression"
		info["parameters"] = []string{"components"}
		
	case Polynomial:
		info["type"] = "nonlinear_regression"
		info["description"] = "Polynomial regression"
		info["parameters"] = []string{"degree"}
		
	case Exponential:
		info["type"] = "nonlinear_regression"
		info["description"] = "Exponential regression"
		info["parameters"] = []string{"max_iterations", "tolerance"}
		
	case Logarithmic:
		info["type"] = "nonlinear_regression"
		info["description"] = "Logarithmic regression"
		info["parameters"] = []string{"max_iterations", "tolerance"}
		
	case Power:
		info["type"] = "nonlinear_regression"
		info["description"] = "Power regression"
		info["parameters"] = []string{"max_iterations", "tolerance"}
	}
	
	return info
}

// GetAllAlgorithmsInfo returns information about all supported algorithms
func GetAllAlgorithmsInfo() map[AlgorithmType]map[string]interface{} {
	algorithms := []AlgorithmType{
		OLS, Ridge, Lasso, Logistic, PLS,
		Polynomial, Exponential, Logarithmic, Power,
	}
	
	info := make(map[AlgorithmType]map[string]interface{})
	for _, alg := range algorithms {
		info[alg] = GetAlgorithmInfo(alg)
	}
	
	return info
}
