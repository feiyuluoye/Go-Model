# Go-Model 统一模型架构

本目录包含了重构后的统一模型架构，整合了原来分散在 `internal/core` 和 `internal/regression` 中的重复功能。

## 目录结构

```
internal/models/
├── interfaces.go          # 统一的模型接口定义
├── manager.go             # 模型管理器
├── models.go              # 统一的模型构造函数导出
├── linear/                # 线性回归模型
│   ├── ols.go            # 普通最小二乘法
│   ├── ridge.go          # 岭回归
│   ├── lasso.go          # Lasso回归
│   ├── logistic.go       # 逻辑回归
│   └── pls.go            # 偏最小二乘回归
└── nonlinear/            # 非线性回归模型
    ├── polynomial.go     # 多项式回归
    ├── exponential.go    # 指数回归
    ├── logarithmic.go    # 对数回归
    └── power.go          # 幂回归
```

## 主要改进

### 1. 统一接口
所有模型都实现了统一的 `Model` 接口：
- `Fit(X *mat.Dense, y *mat.VecDense) error` - 训练模型
- `Predict(X *mat.Dense) *mat.VecDense` - 预测
- `Score(X *mat.Dense, y *mat.VecDense) float64` - 计算R²分数
- `GetParameters() map[string]interface{}` - 获取模型参数
- `GetModelType() string` - 获取模型类型

### 2. 使用gonum库
- 采用了更数学严谨的gonum库实现
- 提供了更好的数值稳定性和性能
- 统一了矩阵和向量操作

### 3. 模型管理器
- 提供了统一的模型创建、训练、预测和评估接口
- 支持模型的生命周期管理
- 线程安全的模型存储和访问

### 4. 消除重复代码
- 删除了 `internal/core` 和 `internal/regression` 中的重复实现
- 保留了最佳的算法实现
- 统一了代码风格和错误处理

## 支持的模型

### 线性模型
- **OLS**: 普通最小二乘法回归
- **Ridge**: 岭回归（L2正则化）
- **Lasso**: Lasso回归（L1正则化）
- **Logistic**: 逻辑回归（分类）
- **PLS**: 偏最小二乘回归

### 非线性模型
- **Polynomial**: 多项式回归
- **Exponential**: 指数回归 (y = a * exp(b * x))
- **Logarithmic**: 对数回归 (y = a * ln(x) + b)
- **Power**: 幂回归 (y = a * x^b)

## 使用示例

```go
import (
    "github.com/feiyuluoye/Go-Model/internal/models"
    "gonum.org/v1/gonum/mat"
)

// 创建模型管理器
manager := models.NewModelManager()

// 配置模型
config := &models.ModelConfig{
    ModelType: "ridge",
    Parameters: map[string]interface{}{
        "alpha": 1.0,
    },
}

// 训练模型
result, err := manager.TrainModel(config, X, y)
if err != nil {
    log.Fatal(err)
}

// 进行预测
predictions, err := manager.Predict(result.ModelID, XTest)
if err != nil {
    log.Fatal(err)
}
```

## 迁移指南

如果你之前使用了 `internal/core` 或 `internal/regression` 中的模型，请按以下方式迁移：

### 旧代码
```go
// 旧的core包
import "github.com/feiyuluoye/Go-Model/internal/core"
model := core.NewOLSModel(true)

// 旧的regression包
import "github.com/feiyuluoye/Go-Model/internal/regression/linear"
model := linear.NewOLS(true)
```

### 新代码
```go
// 新的统一接口
import "github.com/feiyuluoye/Go-Model/internal/models"
model := models.NewOLS()
```

## 注意事项

1. 新架构使用 `gonum.org/v1/gonum/mat` 进行矩阵操作
2. 所有模型的输入格式统一为 `*mat.Dense` 和 `*mat.VecDense`
3. 错误处理更加统一和详细
4. 模型参数通过 `GetParameters()` 方法获取，便于序列化和持久化
