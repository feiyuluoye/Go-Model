# Go-Model 架构重新设计

## 需求分析
- 提供两种独立的输出方式：gRPC接口输出和pkg包外部调用
- 两种方式互不干扰，拒绝输出方式页面之间互相调用
- 共享核心业务逻辑但保持接口独立

## 新架构设计

### 1. 目录结构重构
```
go-model/
├── cmd/                    # 命令行入口
│   ├── grpc-server/       # gRPC服务器入口
│   └── pkg-example/       # pkg包使用示例
├── internal/              # 内部包（不对外暴露）
│   ├── core/              # 核心业务逻辑
│   ├── grpc/              # gRPC相关实现
│   └── service/           # 服务层（可选）
├── pkg/                   # 可导出的包
│   ├── regression/        # 回归模型包（对外暴露）
│   ├── config/            # 配置包
│   └── proto/             # proto定义
└── examples/              # 使用示例
```

### 2. 核心设计原则

#### 2.1 接口分离原则
- **gRPC接口层**: 只处理gRPC协议相关逻辑
- **pkg包接口层**: 提供直接的Go API调用
- **核心业务层**: 共享的业务逻辑，被两种接口层调用

#### 2.2 依赖方向
```
gRPC接口层 → 核心业务层 ← pkg包接口层
     ↓                    ↓
   gRPC协议             Go API调用
```

#### 2.3 禁止交叉调用
- gRPC接口层不能直接调用pkg包接口层
- pkg包接口层不能直接调用gRPC接口层
- 所有调用必须通过核心业务层

### 3. 具体实现方案

#### 3.1 核心业务层 (internal/core/)
```go
// 定义统一的模型接口
type Model interface {
    Fit(X [][]float64, y []float64) error
    Predict(X [][]float64) ([]float64, error)
    Score(X [][]float64, y []float64) (float64, error)
    GetModelInfo() *ModelInfo
}

// 模型工厂
type ModelFactory struct {
    // 创建各种模型实例
}

// 模型管理器
type ModelManager struct {
    // 管理模型生命周期
}
```

#### 3.2 gRPC接口层 (internal/grpc/)
```go
// gRPC服务实现
type RegressionGRPCServer struct {
    coreManager *core.ModelManager
    // 只包含gRPC相关逻辑
}

// 实现proto定义的所有服务方法
func (s *RegressionGRPCServer) TrainModel(ctx context.Context, req *proto.TrainRequest) (*proto.TrainResponse, error) {
    // 调用核心业务层
    result := s.coreManager.TrainModel(req)
    // 转换为gRPC响应格式
    return convertToGRPCResponse(result)
}
```

#### 3.3 pkg包接口层 (pkg/regression/)
```go
// 对外暴露的API
type RegressionClient struct {
    coreManager *core.ModelManager
}

func NewRegressionClient() *RegressionClient {
    return &RegressionClient{
        coreManager: core.NewModelManager(),
    }
}

func (c *RegressionClient) TrainModel(config *TrainConfig) (*TrainingResult, error) {
    // 调用核心业务层
    return c.coreManager.TrainModel(config)
}
```

### 4. 数据流设计

#### 4.1 gRPC数据流
```
gRPC客户端 → gRPC服务端 → 核心业务层 → 模型实现
     ↑          ↑              ↑
gRPC响应 ← gRPC转换层 ← 业务结果 ← 模型结果
```

#### 4.2 pkg包数据流
```
外部调用 → pkg包接口 → 核心业务层 → 模型实现
    ↑          ↑           ↑
直接返回 ← 结果转换 ← 业务结果 ← 模型结果
```

### 5. 配置管理

#### 5.1 统一配置
```go
type Config struct {
    GRPC     GRPCConfig     `yaml:"grpc"`
    Model    ModelConfig    `yaml:"model"`
    Logging  LoggingConfig  `yaml:"logging"`
}

type GRPCConfig struct {
    Address string `yaml:"address"`
    Port    int    `yaml:"port"`
}
```

### 6. 错误处理

#### 6.1 统一错误类型
```go
type ErrorCode int

const (
    ErrorCodeInvalidInput ErrorCode = iota + 1
    ErrorCodeModelNotFound
    ErrorCodeTrainingFailed
    // ...
)

type ModelError struct {
    Code    ErrorCode
    Message string
    Details map[string]interface{}
}
```

#### 6.2 错误转换
- gRPC层：将ModelError转换为gRPC错误格式
- pkg包层：直接返回ModelError

### 7. 测试策略

#### 7.1 单元测试
- 核心业务层：测试模型逻辑
- gRPC层：测试协议转换
- pkg包层：测试API接口

#### 7.2 集成测试
- gRPC客户端-服务端测试
- pkg包使用示例测试

### 8. 部署方案

#### 8.1 gRPC服务部署
```bash
# 启动gRPC服务器
go run cmd/grpc-server/main.go -config config.yaml
```

#### 8.2 pkg包使用
```go
import "github.com/your-org/go-model/pkg/regression"

client := regression.NewRegressionClient()
result, err := client.TrainModel(config)
```

### 9. 优势分析

1. **清晰分离**: gRPC和pkg包完全独立
2. **代码复用**: 核心业务逻辑被共享
3. **易于维护**: 各层职责明确
4. **可扩展性**: 新增输出方式只需添加新的接口层
5. **测试友好**: 各层可以独立测试

### 10. 迁移计划

1. 创建新的目录结构
2. 重构核心业务逻辑
3. 实现gRPC接口层
4. 实现pkg包接口层
5. 更新配置和文档
6. 测试验证

这个架构设计确保了两种输出方式的完全独立，同时最大化代码复用，符合现代软件架构的最佳实践。
