package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

func main() {
	fmt.Println("=== Go-Model 算法示例测试 ===\n")

	// 定义所有示例目录
	examples := []string{
		"ols",
		"ridge", 
		"lasso",
		"logistic",
		"pls",
		"polynomial",
		"exponential",
		"logarithmic",
		"power",
	}

	successCount := 0
	totalCount := len(examples)

	for i, example := range examples {
		fmt.Printf("[%d/%d] 测试 %s 示例...\n", i+1, totalCount, strings.ToUpper(example))
		
		if runExample(example) {
			fmt.Printf("✅ %s 示例运行成功\n\n", strings.ToUpper(example))
			successCount++
		} else {
			fmt.Printf("❌ %s 示例运行失败\n\n", strings.ToUpper(example))
		}
	}

	// 输出总结
	fmt.Println("=== 测试总结 ===")
	fmt.Printf("总计: %d 个示例\n", totalCount)
	fmt.Printf("成功: %d 个\n", successCount)
	fmt.Printf("失败: %d 个\n", totalCount-successCount)
	
	if successCount == totalCount {
		fmt.Println("🎉 所有示例都运行成功！")
	} else {
		fmt.Printf("⚠️  有 %d 个示例需要修复\n", totalCount-successCount)
	}
}

func runExample(exampleName string) bool {
	// 获取当前工作目录
	currentDir, err := os.Getwd()
	if err != nil {
		fmt.Printf("错误: 无法获取当前目录: %v\n", err)
		return false
	}

	// 构建示例目录路径
	exampleDir := filepath.Join(currentDir, exampleName)
	
	// 检查目录是否存在
	if _, err := os.Stat(exampleDir); os.IsNotExist(err) {
		fmt.Printf("错误: 示例目录不存在: %s\n", exampleDir)
		return false
	}

	// 切换到示例目录
	originalDir := currentDir
	err = os.Chdir(exampleDir)
	if err != nil {
		fmt.Printf("错误: 无法切换到目录 %s: %v\n", exampleDir, err)
		return false
	}

	// 确保在函数结束时切换回原目录
	defer func() {
		os.Chdir(originalDir)
	}()

	// 运行 go run main.go
	cmd := exec.Command("go", "run", "main.go")
	
	// 设置超时
	done := make(chan error, 1)
	go func() {
		done <- cmd.Run()
	}()

	select {
	case err := <-done:
		if err != nil {
			fmt.Printf("错误: 运行失败: %v\n", err)
			return false
		}
		return true
	case <-time.After(30 * time.Second):
		fmt.Printf("错误: 运行超时 (30秒)\n")
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
		return false
	}
}
