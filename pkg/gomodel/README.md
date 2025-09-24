# Go-Model Package

Go-Model是一个高级机器学习库，提供统一的API接口用于训练、评估和使用各种回归和分类模型。

## 特性

- **统一API**: 所有算法使用相同的接口
- **多种算法**: 支持线性和非线性回归、分类算法
- **数据处理**: 内置数据加载、预处理和分割功能
- **模型验证**: 支持交叉验证和holdout验证
- **性能评估**: 多种评估指标和模型比较
- **易于使用**: 简化的快速训练和预测接口

## 支持的算法

### 线性模型
- **OLS**: 普通最小二乘法回归
- **Ridge**: 岭回归（L2正则化）
- **Lasso**: Lasso回归（L1正则化）
- **Logistic**: 逻辑回归（二分类）
- **PLS**: 偏最小二乘回归

### 非线性模型
- **Polynomial**: 多项式回归
- **Exponential**: 指数回归
- **Logarithmic**: 对数回归
- **Power**: 幂回归

## 安装

```bash
go get github.com/feiyuluoye/Go-Model/pkg/gomodel
```

## 快速开始

### 基本使用

```go
package main

import (
    "fmt"
    "github.com/feiyuluoye/Go-Model/pkg/gomodel"
)

func main() {
    // 准备数据
    features := [][]float64{
        {1.0, 2.0},
        {2.0, 3.0},
        {3.0, 4.0},
        {4.0, 5.0},
    }
    target := []float64{3.0, 5.0, 7.0, 9.0}
    
    // 快速训练
    result, err := gomodel.QuickTrain(features, target, gomodel.OLS)
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("训练R²: %.4f\n", result.TrainingScore)
}
```

### 完整示例

```go
package main

import (
    "fmt"
    "github.com/feiyuluoye/Go-Model/pkg/gomodel"
)

func main() {
    // 1. 创建客户端
    client := gomodel.NewClient(nil)
    
    // 2. 创建数据工具
    dataUtils := gomodel.NewDataUtils(42)
    
    // 3. 准备训练数据
    features := [][]float64{
        {1.0, 2.0, 3.0},
        {2.0, 3.0, 4.0},
        {3.0, 4.0, 5.0},
        {4.0, 5.0, 6.0},
        {5.0, 6.0, 7.0},
    }
    target := []float64{6.0, 9.0, 12.0, 15.0, 18.0}
    
    trainData, err := dataUtils.CreateFromArrays(features, target, 
        []string{"x1", "x2", "x3"}, "y")
    if err != nil {
        panic(err)
    }
    
    // 4. 配置模型
    config := &gomodel.ModelConfig{
        Algorithm:    gomodel.Ridge,
        Parameters:   map[string]interface{}{"lambda": 1.0},
        LossFunction: gomodel.R2,
        Validation: &gomodel.ValidationConfig{
            Method:     "kfold",
            KFolds:     5,
            RandomSeed: 42,
        },
    }
    
    // 5. 训练模型
    result, err := client.Train(trainData, config)
    if err != nil {
        panic(err)
    }
    
    // 6. 输出结果
    fmt.Printf("算法: %s\n", result.Algorithm)
    fmt.Printf("训练R²: %.4f\n", result.TrainingScore)
    if result.ValidationScore != nil {
        fmt.Printf("验证R²: %.4f\n", *result.ValidationScore)
    }
    
    // 7. 交叉验证结果
    if result.CrossValidation != nil {
        cv := result.CrossValidation
        fmt.Printf("交叉验证: %.4f ± %.4f\n", cv.MeanScore, cv.StdScore)
    }
}
```

## API 参考

### 核心类型

#### Client
主要的客户端接口，提供模型训练和预测功能。

```go
client := gomodel.NewClient(&gomodel.ClientConfig{
    DefaultValidation: &gomodel.ValidationConfig{
        Method:   "holdout",
        TestSize: 0.2,
    },
    RandomSeed: 42,
    Verbose:    true,
})
```

#### ModelConfig
模型配置结构，定义算法类型、参数和验证方法。

```go
config := &gomodel.ModelConfig{
    Algorithm:    gomodel.Lasso,
    Parameters:   map[string]interface{}{
        "lambda": 0.1,
        "max_iterations": 1000,
    },
    LossFunction: gomodel.R2,
    Validation: &gomodel.ValidationConfig{
        Method: "kfold",
        KFolds: 10,
    },
}
```

#### TrainingData
训练数据结构，包含特征矩阵和目标变量。

```go
data := &gomodel.TrainingData{
    Features:     featureMatrix,  // *mat.Dense
    Target:       targetVector,   // *mat.VecDense
    FeatureNames: []string{"x1", "x2"},
    TargetName:   "y",
}
```

### 主要方法

#### 训练模型
```go
result, err := client.Train(data, config)
```

#### 预测
```go
predictions, err := client.Predict(modelID, testFeatures)
```

#### 训练并预测
```go
result, predictions, err := client.TrainAndPredict(trainData, testFeatures, config)
```

### 数据工具

#### 创建数据
```go
dataUtils := gomodel.NewDataUtils(42)

// 从数组创建
data, err := dataUtils.CreateFromArrays(features, target, featureNames, targetName)

// 从CSV加载
data, err := dataUtils.LoadFromCSV("data.csv", "target_column", true)

// 从JSON加载
data, err := dataUtils.LoadFromJSON("data.json")
```

#### 数据预处理
```go
// 标准化
normalizedData, err := dataUtils.Normalize(data)

// 缩放到[0,1]
scaledData, err := dataUtils.Scale(data)

// 分割训练测试集
trainData, testData, err := dataUtils.SplitTrainTest(data, 0.2, true)
```

#### 生成合成数据
```go
// 线性数据
data, err := dataUtils.GenerateSyntheticData(100, 3, 0.1, "linear")

// 多项式数据
data, err := dataUtils.GenerateSyntheticData(100, 2, 0.05, "polynomial")

// 分类数据
data, err := dataUtils.GenerateSyntheticData(200, 4, 0.1, "classification")
```

### 模型管理

```go
manager := gomodel.NewModelManager()

// 训练模型
trainedModel, err := manager.TrainModel(config, data)

// 预测
predictions, err := manager.PredictWithModel(modelID, features)

// 获取模型列表
models := manager.GetModelList()

// 比较模型
comparison, err := manager.CompareModels([]string{id1, id2}, "r2_score")

// 交叉验证
cvResult, err := manager.CrossValidateModel(config, data, 5)
```

## 算法参数

### Ridge回归
```go
Parameters: map[string]interface{}{
    "lambda": 1.0,  // 正则化强度
}
```

### Lasso回归
```go
Parameters: map[string]interface{}{
    "lambda":         0.1,   // 正则化强度
    "max_iterations": 1000,  // 最大迭代次数
    "tolerance":      1e-6,  // 收敛容差
}
```

### 逻辑回归
```go
Parameters: map[string]interface{}{
    "learning_rate":  0.01,  // 学习率
    "max_iterations": 1000,  // 最大迭代次数
    "tolerance":      1e-6,  // 收敛容差
}
```

### PLS回归
```go
Parameters: map[string]interface{}{
    "components": 2,  // 主成分数量
}
```

### 多项式回归
```go
Parameters: map[string]interface{}{
    "degree": 3,  // 多项式度数
}
```

## 验证方法

### Holdout验证
```go
Validation: &gomodel.ValidationConfig{
    Method:     "holdout",
    TestSize:   0.2,        // 测试集比例
    RandomSeed: 42,
}
```

### K折交叉验证
```go
Validation: &gomodel.ValidationConfig{
    Method:     "kfold",
    KFolds:     5,          // 折数
    RandomSeed: 42,
}
```

## 评估指标

- **R2**: 决定系数（回归）
- **MSE**: 均方误差
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **Accuracy**: 准确率（分类）
- **LogLoss**: 对数损失（分类）

## 错误处理

```go
result, err := client.Train(data, config)
if err != nil {
    if gomodelErr, ok := err.(*gomodel.Error); ok {
        fmt.Printf("错误代码: %s\n", gomodelErr.Code)
        fmt.Printf("错误信息: %s\n", gomodelErr.Message)
        if gomodelErr.Details != "" {
            fmt.Printf("详细信息: %s\n", gomodelErr.Details)
        }
    }
}
```

## 最佳实践

1. **数据预处理**: 在训练前对数据进行标准化或缩放
2. **参数调优**: 使用交叉验证选择最佳参数
3. **模型验证**: 始终在独立的测试集上评估模型
4. **错误处理**: 检查所有可能的错误情况
5. **性能监控**: 使用多种指标评估模型性能

## 完整示例

参见 `examples/pkg_usage/` 目录中的完整示例代码。
