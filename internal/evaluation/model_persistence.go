package evaluation

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"reflect"
	"time"

	"gonum.org/v1/gonum/mat"
)

// ModelSerializer 模型序列化接口
type ModelSerializer interface {
	// GetModelType 返回模型类型名称
	GetModelType() string
	// GetParameters 返回模型参数
	GetParameters() map[string]interface{}
	// SetParameters 从参数设置模型
	SetParameters(params map[string]interface{}) error
}

// ModelData 用于序列化的模型数据结构
type ModelData struct {
	ModelType    string                 `json:"model_type"`
	Parameters   map[string]interface{} `json:"parameters"`
	TrainingTime string                 `json:"training_time"`
	Metrics      map[string]float64     `json:"metrics,omitempty"`
}

// SaveModel 将模型保存到文件
func SaveModel(model ModelSerializer, filePath string, metrics map[string]float64) error {
	modelData := ModelData{
		ModelType:    model.GetModelType(),
		Parameters:   model.GetParameters(),
		TrainingTime: time.Now().Format(time.RFC3339),
		Metrics:      metrics,
	}

	// 转换为JSON
	jsonData, err := json.MarshalIndent(modelData, "", "  ")
	if err != nil {
		return fmt.Errorf("序列化模型失败: %w", err)
	}

	// 写入文件
	err = ioutil.WriteFile(filePath, jsonData, 0644)
	if err != nil {
		return fmt.Errorf("写入模型文件失败: %w", err)
	}

	return nil
}

// LoadModel 从文件加载模型到已创建的模型实例
func LoadModel(filePath string, model ModelSerializer) error {
	// 读取文件
	jsonData, err := ioutil.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("读取模型文件失败: %w", err)
	}

	// 解析JSON
	var modelData ModelData
	err = json.Unmarshal(jsonData, &modelData)
	if err != nil {
		return fmt.Errorf("解析模型数据失败: %w", err)
	}

	// 检查模型类型是否匹配
	if model.GetModelType() != modelData.ModelType {
		return errors.New("模型类型不匹配")
	}

	// 设置模型参数
	err = model.SetParameters(modelData.Parameters)
	if err != nil {
		return fmt.Errorf("设置模型参数失败: %w", err)
	}

	return nil
}

// 辅助函数：将mat.VecDense转换为[]float64
func VecDenseToSlice(vec *mat.VecDense) []float64 {
	n := vec.Len()
	slice := make([]float64, n)
	for i := 0; i < n; i++ {
		slice[i] = vec.At(i, 0)
	}
	return slice
}

// 辅助函数：将[]float64转换为mat.VecDense
func SliceToVecDense(slice []float64) *mat.VecDense {
	return mat.NewVecDense(len(slice), slice)
}

// 辅助函数：序列化map中的mat.VecDense
func SerializeParameters(params map[string]interface{}) map[string]interface{} {
	serialized := make(map[string]interface{})

	for key, value := range params {
		// 处理mat.VecDense类型
		if vec, ok := value.(*mat.VecDense); ok {
			serialized[key] = VecDenseToSlice(vec)
		} else {
			serialized[key] = value
		}
	}

	return serialized
}

// 辅助函数：反序列化参数中的向量
func DeserializeParameters(serialized map[string]interface{}, paramTypes map[string]reflect.Type) (map[string]interface{}, error) {
	params := make(map[string]interface{})

	for key, value := range serialized {
		// 检查是否需要特殊类型转换
		if paramType, exists := paramTypes[key]; exists {
			if paramType == reflect.TypeOf(&mat.VecDense{}) {
				// 转换为mat.VecDense
				slice, ok := value.([]interface{})
				if !ok {
					return nil, fmt.Errorf("参数 %s 不是有效的向量数据", key)
				}

				floatSlice := make([]float64, len(slice))
				for i, v := range slice {
					fv, ok := v.(float64)
					if !ok {
						return nil, fmt.Errorf("向量元素 %d 不是有效数字", i)
					}
					floatSlice[i] = fv
				}

				params[key] = SliceToVecDense(floatSlice)
			} else {
				// 对于其他类型，尝试直接转换
				params[key] = value
			}
		} else {
			// 未知类型，直接存储
			params[key] = value
		}
	}

	return params, nil
}
