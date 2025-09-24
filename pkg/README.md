# Go-Model PKG Package

这是Go-Model项目的公共API包，为外部库提供统一、简洁的机器学习算法调用接口。

## 📁 包结构

```
pkg/
└── gomodel/
    ├── types.go           # 核心类型定义
    ├── client.go          # 主要客户端接口
    ├── data_utils.go      # 数据处理工具
    ├── model_manager.go   # 模型管理器
    ├── gomodel.go         # 包入口和便捷函数
    ├── utils.go           # 实用工具函数
    └── README.md          # 详细使用文档
```

## 🚀 核心功能

### 1. 统一的算法接口
- **支持9种算法**: OLS、Ridge、Lasso、Logistic、PLS、Polynomial、Exponential、Logarithmic、Power
- **一致的API**: 所有算法使用相同的训练、预测接口
- **灵活配置**: 支持算法参数、损失函数、验证方法的自定义配置

### 2. 完整的数据处理流水线
- **数据加载**: 支持CSV、JSON文件加载和数组创建
- **数据预处理**: 标准化、缩放、异常值处理
- **数据分割**: 训练测试集分割、交叉验证
- **合成数据**: 生成线性、多项式、分类测试数据

### 3. 高级模型管理
- **模型生命周期**: 创建、训练、预测、评估、删除
- **性能比较**: 多模型性能对比和排序
- **交叉验证**: K折交叉验证和holdout验证
- **批量操作**: 批量预测和模型评估

### 4. 丰富的评估指标
- **回归指标**: R²、MSE、MAE、RMSE
- **分类指标**: Accuracy、LogLoss
- **验证方法**: 交叉验证、holdout验证
- **统计信息**: 均值、标准差、置信区间

## 🎯 设计特点

### 外部友好的API设计
```go
// 简单快速的使用方式
result, err := gomodel.QuickTrain(features, target, gomodel.OLS)

// 完整功能的使用方式
client := gomodel.NewClient(config)
result, err := client.Train(data, modelConfig)
```

### 类型安全和错误处理
```go
type Error struct {
    Code    string `json:"code"`
    Message string `json:"message"`
    Details string `json:"details,omitempty"`
}
```

### 灵活的配置系统
```go
type ModelConfig struct {
    Algorithm    AlgorithmType            `json:"algorithm"`
    Parameters   map[string]interface{}   `json:"parameters"`
    LossFunction LossFunction             `json:"loss_function"`
    Validation   *ValidationConfig        `json:"validation,omitempty"`
}
```

### 完整的结果信息
```go
type ModelResult struct {
    Algorithm       AlgorithmType          `json:"algorithm"`
    TrainingScore   float64                `json:"training_score"`
    ValidationScore *float64               `json:"validation_score,omitempty"`
    Metrics         map[string]float64     `json:"metrics"`
    CrossValidation *CVResult              `json:"cross_validation,omitempty"`
}
```

## 🔧 集成的Internal功能

### 数据处理集成
- 使用 `internal/data` 进行CSV/JSON数据加载
- 集成 `internal/types` 的数据结构定义
- 支持数据预处理和特征工程

### 模型训练集成
- 调用 `internal/models` 的统一模型接口
- 支持所有线性和非线性算法
- 自动参数验证和错误处理

### 评估功能集成
- 使用 `internal/evaluation` 进行交叉验证
- 集成多种评估指标计算
- 支持模型性能比较和排序

## 📊 使用场景

### 1. 快速原型开发
```go
// 一行代码完成训练和评估
result, _ := gomodel.QuickTrain(X, y, gomodel.Ridge)
```

### 2. 生产环境部署
```go
// 完整的配置和验证流程
client := gomodel.NewClient(config)
result, _ := client.Train(data, modelConfig)
```

### 3. 模型研究和比较
```go
// 批量模型训练和性能比较
manager := gomodel.NewModelManager()
comparison, _ := manager.CompareModels(modelIDs, "r2_score")
```

### 4. 数据科学工作流
```go
// 完整的数据处理到模型部署流程
dataUtils := gomodel.NewDataUtils(42)
data, _ := dataUtils.LoadFromCSV("data.csv", "target", true)
normalizedData, _ := dataUtils.Normalize(data)
trainData, testData, _ := dataUtils.SplitTrainTest(normalizedData, 0.2, true)
```

## 🎨 API设计原则

1. **简单易用**: 提供QuickTrain/QuickPredict等便捷接口
2. **功能完整**: 支持完整的机器学习工作流程
3. **类型安全**: 使用强类型定义，减少运行时错误
4. **错误友好**: 详细的错误信息和错误代码
5. **扩展性强**: 易于添加新算法和功能
6. **性能优化**: 使用gonum进行高效数值计算

## 📈 性能特点

- **内存效率**: 使用gonum矩阵进行高效内存管理
- **并发安全**: 模型管理器支持并发访问
- **计算优化**: 集成高性能数值计算库
- **缓存机制**: 训练好的模型可重复使用

## 🔗 与Internal包的关系

```
pkg/gomodel (外部API)
    ↓ 调用
internal/models (模型实现)
internal/data (数据处理)
internal/evaluation (模型评估)
internal/types (类型定义)
```

这种设计确保了：
- 外部用户只需要了解pkg接口
- 内部实现可以独立演进
- 代码复用和模块化
- 清晰的职责分离

## 📚 文档和示例

- **详细文档**: `pkg/gomodel/README.md`
- **基础示例**: `examples/pkg_usage/basic_example.go`
- **高级示例**: `examples/pkg_usage/advanced_example.go`
- **API参考**: 完整的类型定义和方法说明

通过这个pkg包，外部开发者可以轻松集成Go-Model的机器学习功能，而无需了解内部实现细节。
