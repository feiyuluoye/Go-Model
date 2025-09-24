package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

func main() {
	fmt.Println("🚀 Go-Model 完整功能测试")
	fmt.Println("=" + string(make([]byte, 50)) + "=")

	startTime := time.Now()
	
	// 1. 测试依赖安装
	fmt.Println("\n📦 1. 检查依赖...")
	if err := runCommand("go", "mod", "tidy"); err != nil {
		log.Printf("❌ 依赖安装失败: %v", err)
		return
	}
	fmt.Println("✅ 依赖检查完成")

	// 2. 测试编译
	fmt.Println("\n🔨 2. 测试项目编译...")
	if err := runCommand("go", "build", "./..."); err != nil {
		log.Printf("❌ 编译失败: %v", err)
		return
	}
	fmt.Println("✅ 编译成功")

	// 3. 运行单元测试
	fmt.Println("\n🧪 3. 运行单元测试...")
	if err := runCommand("go", "test", "./...", "-v"); err != nil {
		log.Printf("⚠️  部分测试失败: %v", err)
	} else {
		fmt.Println("✅ 单元测试通过")
	}

	// 4. 测试示例运行
	fmt.Println("\n📊 4. 测试算法示例...")
	testExamples()

	// 5. 测试PKG API
	fmt.Println("\n🔌 5. 测试PKG API...")
	testPkgAPI()

	// 6. 性能基准测试
	fmt.Println("\n⚡ 6. 运行性能基准...")
	if err := runCommand("go", "test", "-bench=.", "-benchtime=1s", "./internal/models/..."); err != nil {
		log.Printf("⚠️  基准测试失败: %v", err)
	} else {
		fmt.Println("✅ 基准测试完成")
	}

	duration := time.Since(startTime)
	fmt.Printf("\n🎉 测试完成！总耗时: %v\n", duration)
	
	// 输出使用指南
	printUsageGuide()
}

func testExamples() {
	examples := []string{"ols", "ridge", "lasso", "logistic", "polynomial"}
	
	for _, example := range examples {
		examplePath := filepath.Join("examples", example)
		if _, err := os.Stat(examplePath); os.IsNotExist(err) {
			fmt.Printf("⚠️  示例目录不存在: %s\n", example)
			continue
		}
		
		fmt.Printf("  测试 %s 示例...", example)
		
		// 切换到示例目录并运行
		originalDir, _ := os.Getwd()
		os.Chdir(examplePath)
		
		if err := runCommandQuiet("go", "run", "main.go"); err != nil {
			fmt.Printf(" ❌ 失败\n")
		} else {
			fmt.Printf(" ✅ 成功\n")
		}
		
		os.Chdir(originalDir)
	}
}

func testPkgAPI() {
	// 创建临时测试文件
	testCode := `package main

import (
	"fmt"
	"github.com/feiyuluoye/Go-Model/pkg/gomodel"
)

func main() {
	// 测试快速训练
	features := [][]float64{{1, 2}, {2, 3}, {3, 4}}
	target := []float64{5, 8, 11}
	
	result, err := gomodel.QuickTrain(features, target, gomodel.OLS)
	if err != nil {
		panic(err)
	}
	
	fmt.Printf("PKG API测试成功 - R²: %.4f\n", result.TrainingScore)
}`

	// 写入临时文件
	tmpFile := "pkg_test_temp.go"
	if err := os.WriteFile(tmpFile, []byte(testCode), 0644); err != nil {
		fmt.Printf("❌ 创建测试文件失败: %v\n", err)
		return
	}
	defer os.Remove(tmpFile)

	// 运行测试
	if err := runCommandQuiet("go", "run", tmpFile); err != nil {
		fmt.Printf("❌ PKG API测试失败: %v\n", err)
	} else {
		fmt.Println("✅ PKG API测试成功")
	}
}

func runCommand(name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func runCommandQuiet(name string, args ...string) error {
	cmd := exec.Command(name, args...)
	return cmd.Run()
}

func printUsageGuide() {
	fmt.Println("\n📚 使用指南:")
	fmt.Println("=" + string(make([]byte, 50)) + "=")
	
	fmt.Println("\n🚀 快速开始:")
	fmt.Println("  # 运行OLS示例")
	fmt.Println("  cd examples/ols && go run main.go")
	fmt.Println()
	fmt.Println("  # 运行所有示例")
	fmt.Println("  cd examples && go run run_all.go")
	fmt.Println()
	fmt.Println("  # 测试PKG API")
	fmt.Println("  cd examples/pkg_usage && go run basic_example.go")

	fmt.Println("\n🔧 开发命令:")
	fmt.Println("  # 安装依赖")
	fmt.Println("  go mod tidy")
	fmt.Println()
	fmt.Println("  # 运行测试")
	fmt.Println("  go test ./...")
	fmt.Println()
	fmt.Println("  # 性能基准")
	fmt.Println("  go test -bench=. ./...")
	fmt.Println()
	fmt.Println("  # 代码格式化")
	fmt.Println("  gofmt -w .")

	fmt.Println("\n📖 API使用:")
	fmt.Println("  import \"github.com/feiyuluoye/Go-Model/pkg/gomodel\"")
	fmt.Println("  result, _ := gomodel.QuickTrain(features, target, gomodel.OLS)")

	fmt.Println("\n🔗 更多信息:")
	fmt.Println("  - API文档: pkg/gomodel/README.md")
	fmt.Println("  - 示例代码: examples/")
	fmt.Println("  - 项目文档: docs/")
}
