package config

import (
	"os"

	"gopkg.in/yaml.v3"
)

// Config 应用配置
type Config struct {
	GRPC     GRPCConfig     `yaml:"grpc"`
	Database DatabaseConfig `yaml:"database"`
	Logging  LoggingConfig  `yaml:"logging"`
}

// GRPCConfig gRPC配置
type GRPCConfig struct {
	Address string `yaml:"address"`
	Port    int    `yaml:"port"`
	Timeout int    `yaml:"timeout"`
}

// DatabaseConfig 数据库配置
type DatabaseConfig struct {
	Host     string `yaml:"host"`
	Port     int    `yaml:"port"`
	Name     string `yaml:"name"`
	User     string `yaml:"user"`
	Password string `yaml:"password"`
}

// LoggingConfig 日志配置
type LoggingConfig struct {
	Level string `yaml:"level"`
	File  string `yaml:"file"`
}

// DefaultConfig 返回默认配置
func DefaultConfig() *Config {
	return &Config{
		GRPC: GRPCConfig{
			Address: "localhost",
			Port:    50051,
			Timeout: 30,
		},
		Database: DatabaseConfig{
			Host: "localhost",
			Port: 5432,
			Name: "regression_db",
		},
		Logging: LoggingConfig{
			Level: "info",
			File:  "logs/app.log",
		},
	}
}

// Load 从文件加载配置
func Load(filename string) (*Config, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var cfg Config
	err = yaml.Unmarshal(data, &cfg)
	if err != nil {
		return nil, err
	}

	return &cfg, nil
}

// Save 保存配置到文件
func (c *Config) Save(filename string) error {
	data, err := yaml.Marshal(c)
	if err != nil {
		return err
	}

	return os.WriteFile(filename, data, 0644)
}
