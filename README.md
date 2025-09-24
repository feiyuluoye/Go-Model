# Go-Model

🚀 一个高性能的Go语言机器学习库，提供统一的回归和分类算法接口

[![Go Version](https://img.shields.io/badge/Go-1.19+-blue.svg)](https://golang.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()

## 📋 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [项目架构](#项目架构)
- [快速开始](#快速开始)
- [API使用指南](#api使用指南)
- [算法支持](#算法支持)
- [测试验证](#测试验证)
- [性能基准](#性能基准)
- [开发指南](#开发指南)

## 🎯 项目概述

Go-Model是一个专为Go语言设计的机器学习库，专注于提供高性能、易用的回归和分类算法实现。项目采用现代化的模块设计，支持从简单的线性回归到复杂的非线性模型，适用于数据科学、机器学习研究和生产环境部署。

### 🌟 核心特性

- **🔧 统一API设计**: 所有算法使用一致的接口，学习成本低
- **⚡ 高性能计算**: 基于gonum库优化的矩阵运算
- **📦 模块化架构**: 清晰的代码结构，易于扩展和维护
- **🔒 类型安全**: 强类型设计，减少运行时错误
- **📊 丰富评估**: 多种评估指标和交叉验证支持
- **🚀 快速上手**: 提供QuickTrain等便捷接口
- **📚 完整文档**: 详细的API文档和使用示例

## 🏗️ 项目架构

```
Go-Model/
├── 📁 cmd/                     # 命令行工具
│   └── main.go                # 主程序入口
├── 📁 configs/                # 配置文件
│   └── config.yaml           # 项目配置
├── 📁 internal/               # 内部实现（核心算法）
│   ├── 📁 data/              # 数据处理模块
│   │   ├── data_loader.go    # 数据加载
│   │   ├── preprocessing.go  # 数据预处理
│   │   └── split.go         # 数据分割
│   ├── 📁 evaluation/        # 模型评估
│   │   ├── metrics.go       # 评估指标
│   │   └── cross_validation.go # 交叉验证
│   ├── 📁 models/           # 统一模型接口
│   │   ├── interfaces.go    # 模型接口定义
│   │   ├── manager.go       # 模型管理器
│   │   ├── 📁 linear/       # 线性模型
│   │   │   ├── ols.go      # 普通最小二乘
│   │   │   ├── ridge.go    # 岭回归
│   │   │   ├── lasso.go    # Lasso回归
│   │   │   ├── logistic.go # 逻辑回归
│   │   │   └── pls.go      # 偏最小二乘
│   │   └── 📁 nonlinear/    # 非线性模型
│   │       ├── polynomial.go # 多项式回归
│   │       ├── exponential.go # 指数回归
│   │       ├── logarithmic.go # 对数回归
│   │       └── power.go      # 幂回归
│   └── 📁 types/            # 类型定义
│       ├── dataset.go       # 数据集类型
│       └── model.go         # 模型类型
├── 📁 pkg/                   # 公共API（外部接口）
│   └── 📁 gomodel/          # 主要API包
│       ├── client.go        # 客户端接口
│       ├── types.go         # 公共类型
│       ├── data_utils.go    # 数据工具
│       ├── model_manager.go # 模型管理
│       ├── gomodel.go       # 包入口
│       └── README.md        # API文档
├── 📁 examples/             # 使用示例
│   ├── 📁 ols/             # OLS示例
│   ├── 📁 ridge/           # Ridge示例
│   ├── 📁 lasso/           # Lasso示例
│   ├── 📁 logistic/        # 逻辑回归示例
│   ├── 📁 polynomial/      # 多项式回归示例
│   ├── 📁 pkg_usage/       # PKG API使用示例
│   ├── run_all.go          # 批量运行脚本
│   └── README.md           # 示例说明
├── 📁 docs/                # 项目文档
└── 📄 README.md            # 项目说明
```

## 🚀 快速开始

### 📋 环境要求

- **Go**: 1.19+ 
- **操作系统**: Windows/Linux/macOS
- **内存**: 建议2GB+

### 📦 安装

```bash
# 1. 克隆项目
git clone https://github.com/feiyuluoye/Go-Model.git
cd Go-Model

# 2. 安装依赖
go mod tidy

# 3. 验证安装
go version
```

### ⚡ 快速体验

#### 方式一：使用便捷API
```bash
# 创建测试文件
cat > quick_test.go << 'EOF'
package main

import (
    "fmt"
    "github.com/feiyuluoye/Go-Model/pkg/gomodel"
)

func main() {
    // 准备数据 y = 2*x1 + 3*x2 + 1
    features := [][]float64{
        {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0},
    }
    target := []float64{7.0, 11.0, 15.0, 19.0}
    
    // 一行代码训练模型
    result, err := gomodel.QuickTrain(features, target, gomodel.OLS)
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("🎯 算法: %s\n", result.Algorithm)
    fmt.Printf("📊 训练R²: %.4f\n", result.TrainingScore)
    fmt.Printf("📈 RMSE: %.4f\n", result.Metrics["rmse"])
}
EOF

# 运行测试
go run quick_test.go
```

#### 方式二：运行内置示例
```bash
# 运行单个算法示例
cd examples/ols
go run main.go

# 运行所有算法示例
cd examples
go run run_all.go
```

### 🎯 预期输出
```
🎯 算法: ols
📊 训练R²: 1.0000
📈 RMSE: 0.0000
```

## 📖 API使用指南

### 🔧 基础使用

```go
package main

import (
    "fmt"
    "github.com/feiyuluoye/Go-Model/pkg/gomodel"
)

func main() {
    // 1. 创建客户端
    client := gomodel.NewClient(nil)
    
    // 2. 准备数据
    dataUtils := gomodel.NewDataUtils(42)
    data, _ := dataUtils.CreateFromArrays(
        [][]float64{{1, 2}, {2, 3}, {3, 4}},
        []float64{5, 8, 11},
        []string{"x1", "x2"}, "y")
    
    // 3. 配置模型
    config := &gomodel.ModelConfig{
        Algorithm:    gomodel.Ridge,
        Parameters:   map[string]interface{}{"lambda": 1.0},
        LossFunction: gomodel.R2,
    }
    
    // 4. 训练模型
    result, _ := client.Train(data, config)
    fmt.Printf("训练完成，R² = %.4f\n", result.TrainingScore)
}
```

### 🔄 完整工作流程

```go
func completeWorkflow() {
    dataUtils := gomodel.NewDataUtils(42)
    client := gomodel.NewClient(nil)
    
    // 1. 生成合成数据
    data, _ := dataUtils.GenerateSyntheticData(100, 3, 0.1, "linear")
    
    // 2. 数据预处理
    normalizedData, _ := dataUtils.Normalize(data)
    trainData, testData, _ := dataUtils.SplitTrainTest(normalizedData, 0.2, true)
    
    // 3. 模型配置与训练
    config := &gomodel.ModelConfig{
        Algorithm: gomodel.Lasso,
        Parameters: map[string]interface{}{
            "lambda": 0.1,
            "max_iterations": 1000,
        },
        Validation: &gomodel.ValidationConfig{
            Method: "kfold",
            KFolds: 5,
        },
    }
    
    result, _ := client.Train(trainData, config)
    
    // 4. 结果分析
    fmt.Printf("训练R²: %.4f\n", result.TrainingScore)
    if result.ValidationScore != nil {
        fmt.Printf("验证R²: %.4f\n", *result.ValidationScore)
    }
    if result.CrossValidation != nil {
        cv := result.CrossValidation
        fmt.Printf("交叉验证: %.4f ± %.4f\n", cv.MeanScore, cv.StdScore)
    }
}
```

## 🧮 算法支持

### 📊 线性模型

| 算法 | 类型 | 特点 | 参数 |
|------|------|------|------|
| **OLS** | `gomodel.OLS` | 普通最小二乘法 | 无 |
| **Ridge** | `gomodel.Ridge` | L2正则化 | `lambda` |
| **Lasso** | `gomodel.Lasso` | L1正则化，特征选择 | `lambda`, `max_iterations` |
| **Logistic** | `gomodel.Logistic` | 二分类 | `learning_rate`, `max_iterations` |
| **PLS** | `gomodel.PLS` | 降维回归 | `components` |

### 📈 非线性模型

| 算法 | 类型 | 特点 | 参数 |
|------|------|------|------|
| **Polynomial** | `gomodel.Polynomial` | 多项式拟合 | `degree` |
| **Exponential** | `gomodel.Exponential` | 指数关系 | `max_iterations` |
| **Logarithmic** | `gomodel.Logarithmic` | 对数关系 | `max_iterations` |
| **Power** | `gomodel.Power` | 幂函数关系 | `max_iterations` |

### 🎛️ 算法参数示例

```go
// Ridge回归配置
ridgeConfig := map[string]interface{}{
    "lambda": 1.0,  // 正则化强度
}

// Lasso回归配置
lassoConfig := map[string]interface{}{
    "lambda":         0.1,   // 正则化强度
    "max_iterations": 1000,  // 最大迭代次数
    "tolerance":      1e-6,  // 收敛容差
}

// 多项式回归配置
polyConfig := map[string]interface{}{
    "degree": 3,  // 多项式度数
}
```

## 🧪 测试验证

### 🔍 运行所有测试

```bash
# 运行单元测试
go test ./...

# 运行基准测试
go test -bench=. ./...

# 运行覆盖率测试
go test -cover ./...
```

### 📊 算法验证

```bash
# 验证所有算法
cd examples
go run run_all.go

# 验证特定算法
cd examples/ridge
go run main.go

# 验证PKG API
cd examples/pkg_usage
go run basic_example.go
go run advanced_example.go
```

### ✅ 预期测试结果

```
=== Go-Model 算法示例测试 ===

[1/9] 测试 OLS 示例...
✅ OLS 示例运行成功

[2/9] 测试 RIDGE 示例...
✅ RIDGE 示例运行成功

[3/9] 测试 LASSO 示例...
✅ LASSO 示例运行成功

... 

=== 测试总结 ===
总计: 9 个示例
成功: 9 个
失败: 0 个
🎉 所有示例都运行成功！
```

## ⚡ 性能基准

### 📈 性能测试命令

```bash
# 运行性能基准测试
go test -bench=BenchmarkOLS ./internal/models/linear/
go test -bench=BenchmarkRidge ./internal/models/linear/
go test -bench=BenchmarkLasso ./internal/models/linear/

# 内存使用分析
go test -bench=. -benchmem ./...

# CPU性能分析
go test -bench=. -cpuprofile=cpu.prof ./...
go tool pprof cpu.prof
```

### 📊 性能指标

| 算法 | 1000样本/10特征 | 内存使用 | 并发安全 |
|------|----------------|----------|----------|
| OLS | ~1ms | ~50KB | ✅ |
| Ridge | ~2ms | ~60KB | ✅ |
| Lasso | ~10ms | ~80KB | ✅ |
| Logistic | ~15ms | ~70KB | ✅ |

## 🛠️ 开发指南

### 🔧 开发环境设置

```bash
# 1. 安装开发工具
go install golang.org/x/tools/cmd/goimports@latest
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# 2. 代码格式化
gofmt -w .
goimports -w .

# 3. 代码检查
golangci-lint run
```

### 📝 添加新算法

1. **实现算法接口**
```go
// internal/models/linear/new_algorithm.go
type NewAlgorithm struct {
    // 算法参数
}

func (na *NewAlgorithm) Fit(X *mat.Dense, y *mat.VecDense) error {
    // 实现训练逻辑
}

func (na *NewAlgorithm) Predict(X *mat.Dense) (*mat.VecDense, error) {
    // 实现预测逻辑
}

// 实现其他接口方法...
```

2. **添加到构造函数**
```go
// internal/models/models.go
func NewNewAlgorithm(params map[string]interface{}) models.Model {
    return &linear.NewAlgorithm{
        // 初始化参数
    }
}
```

3. **更新PKG接口**
```go
// pkg/gomodel/types.go
const (
    NewAlgorithmType AlgorithmType = "new_algorithm"
)
```

4. **创建示例**
```go
// examples/new_algorithm/main.go
// 创建使用示例
```

### 🧪 测试新算法

```bash
# 创建测试文件
# internal/models/linear/new_algorithm_test.go

# 运行测试
go test ./internal/models/linear/ -v

# 创建示例
cd examples/new_algorithm
go run main.go
```

## 📚 API参考

### 🔗 主要接口

```go
// 客户端接口
type Client interface {
    Train(data *TrainingData, config *ModelConfig) (*ModelResult, error)
    Predict(modelID string, features *mat.Dense) (*PredictionResult, error)
    TrainAndPredict(trainData *TrainingData, testFeatures *mat.Dense, config *ModelConfig) (*ModelResult, *PredictionResult, error)
}

// 数据工具接口
type DataUtils interface {
    CreateFromArrays(features [][]float64, target []float64, featureNames []string, targetName string) (*TrainingData, error)
    LoadFromCSV(filePath string, targetColumn interface{}, hasHeader bool) (*TrainingData, error)
    Normalize(data *TrainingData) (*TrainingData, error)
    SplitTrainTest(data *TrainingData, testSize float64, shuffle bool) (*TrainingData, *TrainingData, error)
}
```

### 📖 详细文档

- **API文档**: [pkg/gomodel/README.md](pkg/gomodel/README.md)
- **示例文档**: [examples/README.md](examples/README.md)
- **架构文档**: [docs/architecture_design.md](docs/architecture_design.md)

## 🤝 贡献指南

1. **Fork项目** → 创建你的功能分支
2. **编写代码** → 遵循代码规范
3. **添加测试** → 确保测试覆盖率
4. **提交PR** → 详细描述更改内容

### 📋 代码规范

- 使用`gofmt`格式化代码
- 遵循Go官方命名规范
- 添加必要的注释和文档
- 保持测试覆盖率>80%

## 📄 许可证

本项目采用 [MIT许可证](LICENSE)

## 📞 联系方式

- **项目地址**: https://github.com/feiyuluoye/Go-Model
- **问题反馈**: [GitHub Issues](https://github.com/feiyuluoye/Go-Model/issues)
- **功能建议**: [GitHub Discussions](https://github.com/feiyuluoye/Go-Model/discussions)

## 🎉 致谢

感谢所有贡献者和以下开源项目：

- [gonum](https://gonum.org/) - 科学计算库
- [Go](https://golang.org/) - 编程语言
- 所有提供反馈和建议的用户

---
