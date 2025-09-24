package data

import (
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/feiyuluoye/Go-Model/internal/types"
	"io"
	"log"
	"os"
	"strconv"
)

// LoadCSV 从CSV文件加载数据
// filePath: CSV文件路径
// hasHeader: 是否包含表头
// targetColumn: 目标变量列名或索引
func LoadCSV(filePath string, hasHeader bool, targetColumn interface{}) (*types.Dataset, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("无法打开文件: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)

	// 读取所有记录
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("读取CSV文件失败: %w", err)
	}

	if len(records) == 0 {
		return nil, errors.New("CSV文件为空")
	}

	var featureNames []string
	var startRow int

	if hasHeader {
		// 使用第一行作为特征名
		featureNames = records[0]
		startRow = 1
	}

	// 确定目标列的索引
	targetIndex := -1
	switch v := targetColumn.(type) {
	case string:
		// 目标列是字符串名称
		if !hasHeader {
			return nil, errors.New("当目标列是名称时，文件必须包含表头")
		}
		for i, name := range featureNames {
			if name == v {
				targetIndex = i
				break
			}
		}
		if targetIndex == -1 {
			return nil, fmt.Errorf("未找到目标列: %s", v)
		}
	case int:
		// 目标列是索引
		if v < 0 || (hasHeader && v >= len(featureNames)) || (!hasHeader && v >= len(records[0])) {
			return nil, errors.New("目标列索引超出范围")
		}
		targetIndex = v
	default:
		return nil, errors.New("目标列参数类型必须是string或int")
	}

	// 准备数据集
	numSamples := len(records) - startRow
	numFeatures := len(records[startRow]) - 1

	features := make([][]float64, numSamples)
	target := make([]float64, numSamples)

	// 处理数据
	for i := 0; i < numSamples; i++ {
		row := records[startRow+i]
		features[i] = make([]float64, 0, numFeatures)

		for j := 0; j < len(row); j++ {
			if j == targetIndex {
				// 处理目标变量
				val, err := strconv.ParseFloat(row[j], 64)
				if err != nil {
					log.Printf("警告: 行 %d 的目标值 '%s' 不是有效数字，跳过此行", i, row[j])
					i-- // 回退索引
					numSamples--
					features = features[:numSamples]
					target = target[:numSamples]
					goto nextRow
				}
				target[i] = val
			} else {
				// 处理特征
				val, err := strconv.ParseFloat(row[j], 64)
				if err != nil {
					log.Printf("警告: 行 %d 列 %d 的值 '%s' 不是有效数字，使用0代替", i, j, row[j])
					val = 0.0
				}
				features[i] = append(features[i], val)
			}
		}
	nextRow:
	}

	// 如果有表头，需要移除目标列名
	if hasHeader {
		newFeatureNames := make([]string, 0, len(featureNames)-1)
		for i, name := range featureNames {
			if i != targetIndex {
				newFeatureNames = append(newFeatureNames, name)
			}
		}
		featureNames = newFeatureNames
	} else if len(featureNames) == 0 {
		// 如果没有表头，生成默认特征名
		featureNames = make([]string, numFeatures)
		for i := range featureNames {
			featureNames[i] = fmt.Sprintf("feature_%d", i)
		}
	}

	return types.NewDataset(features, target, featureNames), nil
}

// LoadJSON 从JSON文件加载数据
// filePath: JSON文件路径
// featureColumns: 特征列名称列表
// targetColumn: 目标变量列名称
func LoadJSON(filePath string, featureColumns []string, targetColumn string) (*types.Dataset, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("无法打开文件: %w", err)
	}
	defer file.Close()

	// 读取文件内容
	byteValue, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("读取JSON文件失败: %w", err)
	}

	// 解析JSON数据
	var data []map[string]interface{}
	err = json.Unmarshal(byteValue, &data)
	if err != nil {
		return nil, fmt.Errorf("解析JSON数据失败: %w", err)
	}

	if len(data) == 0 {
		return nil, errors.New("JSON数据为空")
	}

	// 准备数据集
	numSamples := len(data)
	numFeatures := len(featureColumns)

	features := make([][]float64, numSamples)
	target := make([]float64, numSamples)

	// 处理数据
	for i, record := range data {
		features[i] = make([]float64, numFeatures)

		// 处理目标变量
		targetVal, ok := record[targetColumn]
		if !ok {
			return nil, fmt.Errorf("记录 %d 缺少目标列: %s", i, targetColumn)
		}

		targetFloat, err := toFloat64(targetVal)
		if err != nil {
			return nil, fmt.Errorf("记录 %d 的目标值不是有效数字: %w", i, err)
		}
		target[i] = targetFloat

		// 处理特征
		for j, colName := range featureColumns {
			val, ok := record[colName]
			if !ok {
				log.Printf("警告: 记录 %d 缺少特征列: %s，使用0代替", i, colName)
				features[i][j] = 0.0
				continue
			}

			floatVal, err := toFloat64(val)
			if err != nil {
				log.Printf("警告: 记录 %d 的特征列 %s 值不是有效数字，使用0代替", i, colName)
				floatVal = 0.0
			}
			features[i][j] = floatVal
		}
	}

	return types.NewDataset(features, target, featureColumns), nil
}

// toFloat64 将interface{}转换为float64
func toFloat64(val interface{}) (float64, error) {
	switch v := val.(type) {
	case float64:
		return v, nil
	case float32:
		return float64(v), nil
	case int:
		return float64(v), nil
	case int64:
		return float64(v), nil
	case int32:
		return float64(v), nil
	case string:
		return strconv.ParseFloat(v, 64)
	default:
		return 0, fmt.Errorf("无法将类型 %T 转换为float64", val)
	}
}
