package models

import (
	"github.com/feiyuluoye/Go-Model/internal/models/linear"
	"github.com/feiyuluoye/Go-Model/internal/models/nonlinear"
)

// 导出所有模型构造函数，提供统一的访问接口

// Linear models
func NewOLS() Model {
	return linear.NewOLS()
}

func NewRidge(lambda float64) Model {
	return linear.NewRidge(lambda)
}

func NewLasso(lambda float64) Model {
	return linear.NewLasso(lambda)
}

func NewLogistic() Model {
	return linear.NewLogistic()
}

func NewPLS(numComponents int) Model {
	return linear.NewPLS(numComponents)
}

// Nonlinear models
func NewPolynomial(degree int) Model {
	return nonlinear.NewPolynomial(degree)
}

func NewExponential() Model {
	return nonlinear.NewExponential()
}

func NewLogarithmic() Model {
	return nonlinear.NewLogarithmic()
}

func NewPower() Model {
	return nonlinear.NewPower()
}
