package linear

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

// PLS 偏最小二乘回归模型实现
type PLS struct {
	XWeights      *mat.Dense // W
	YWeights      *mat.Dense // C
	XLoadings     *mat.Dense // P
	YLoadings     *mat.Dense // Q
	XScores       *mat.Dense // T
	YScores       *mat.Dense // U
	NumComponents int
	isTrained     bool
}

// NewPLS 创建新的PLS回归模型
func NewPLS(numComponents int) *PLS {
	return &PLS{
		NumComponents: numComponents,
		isTrained:     false,
	}
}

// Fit 训练PLS模型使用NIPALS算法
func (p *PLS) Fit(X *mat.Dense, y *mat.VecDense) error {
	n, pDim := X.Dims()

	// 转换y为矩阵
	yMatrix := mat.NewDense(n, 1, nil)
	for i := 0; i < n; i++ {
		yMatrix.Set(i, 0, y.At(i, 0))
	}

	// 工作矩阵（我们将修改副本）
	XResidual := mat.DenseCopyOf(X)
	YResidual := mat.DenseCopyOf(yMatrix)

	// 初始化存储矩阵
	p.XScores = mat.NewDense(n, p.NumComponents, nil)
	p.YScores = mat.NewDense(n, p.NumComponents, nil)
	p.XWeights = mat.NewDense(pDim, p.NumComponents, nil)
	p.YWeights = mat.NewDense(1, p.NumComponents, nil)
	p.XLoadings = mat.NewDense(pDim, p.NumComponents, nil)
	p.YLoadings = mat.NewDense(1, p.NumComponents, nil)

	// NIPALS算法
	for k := 0; k < p.NumComponents; k++ {
		// 步骤1：取u为YResidual的第一列
		u := mat.Col(nil, 0, YResidual)

		var wVec, cVec *mat.VecDense

		// 迭代收敛第k个成分
		for innerIter := 0; innerIter < 100; innerIter++ {
			uOld := make([]float64, len(u))
			copy(uOld, u)

			// 步骤2：w = X' * u / (u' * u)
			wVec = mat.NewVecDense(pDim, nil)
			wVec.MulVec(XResidual.T(), mat.NewVecDense(n, u))
			uNorm := mat.Dot(mat.NewVecDense(n, u), mat.NewVecDense(n, u))
			if uNorm > 0 {
				wVec.ScaleVec(1.0/uNorm, wVec)
			}

			// 标准化w
			wNorm := mat.Norm(wVec, 2)
			if wNorm > 0 {
				wVec.ScaleVec(1.0/wNorm, wVec)
			}

			// 步骤3：t = X * w
			tVec := mat.NewVecDense(n, nil)
			tVec.MulVec(XResidual, wVec)

			// 存储得分
			p.XScores.SetCol(k, tVec.RawVector().Data)

			// 步骤4：c = Y' * t / (t' * t)
			cVec = mat.NewVecDense(1, nil)
			cVec.MulVec(YResidual.T(), tVec)
			tNorm := mat.Dot(tVec, tVec)
			if tNorm > 0 {
				cVec.ScaleVec(1.0/tNorm, cVec)
			}

			// 步骤5：u = Y * c / (c' * c)
			uVec := mat.NewVecDense(n, nil)
			uVec.MulVec(YResidual, cVec)
			cNorm := mat.Dot(cVec, cVec)
			if cNorm > 0 {
				uVec.ScaleVec(1.0/cNorm, uVec)
			}
			u = uVec.RawVector().Data

			// 检查收敛性
			var diff float64
			for i := 0; i < n; i++ {
				diff += (u[i] - uOld[i]) * (u[i] - uOld[i])
			}
			diff = math.Sqrt(diff)
			if diff < 1e-6 {
				break
			}
		}

		// 步骤6：p = X' * t / (t' * t)
		pVec := mat.NewVecDense(pDim, nil)
		pVec.MulVec(XResidual.T(), p.XScores.ColView(k))
		tNorm := mat.Dot(p.XScores.ColView(k), p.XScores.ColView(k))
		if tNorm > 0 {
			pVec.ScaleVec(1.0/tNorm, pVec)
		}
		p.XLoadings.SetCol(k, pVec.RawVector().Data)

		// 步骤7：q = Y' * t / (t' * t)
		qVec := mat.NewVecDense(1, nil)
		qVec.MulVec(YResidual.T(), p.XScores.ColView(k))
		if tNorm > 0 {
			qVec.ScaleVec(1.0/tNorm, qVec)
		}
		p.YLoadings.SetCol(k, qVec.RawVector().Data)

		// 存储权重
		wWeight := mat.VecDenseCopyOf(wVec)
		cWeight := mat.VecDenseCopyOf(cVec)
		p.XWeights.SetCol(k, wWeight.RawVector().Data)
		p.YWeights.SetCol(k, cWeight.RawVector().Data)

		// 收缩X和Y
		tt := mat.NewDense(n, n, nil)
		tt.Outer(1.0, p.XScores.ColView(k), p.XScores.ColView(k))
		xNorm := mat.Dot(p.XScores.ColView(k), p.XScores.ColView(k))
		if xNorm > 0 {
			tt.Scale(1.0/xNorm, tt)
		}
		var newX mat.Dense
		newX.Mul(tt, XResidual)
		XResidual.Sub(XResidual, &newX)

		var newY mat.Dense
		newY.Mul(tt, YResidual)
		YResidual.Sub(YResidual, &newY)
	}

	p.isTrained = true
	return nil
}

// Predict 使用训练好的PLS模型进行预测
func (p *PLS) Predict(X *mat.Dense) *mat.VecDense {
	n, _ := X.Dims()
	predictions := mat.NewVecDense(n, nil)

	// 计算 X_scores = X * W
	xScores := mat.NewDense(n, p.NumComponents, nil)
	xScores.Mul(X, p.XWeights)

	// 计算 Y_hat = X_scores * Q^T
	yHat := mat.NewDense(n, 1, nil)
	yHat.Mul(xScores, p.YLoadings.T())

	// 提取预测值
	for i := 0; i < n; i++ {
		predictions.SetVec(i, yHat.At(i, 0))
	}

	return predictions
}

// Score 计算模型评分 (R²)
func (p *PLS) Score(X *mat.Dense, y *mat.VecDense) float64 {
	yPred := p.Predict(X)
	var ssTotal, ssRes float64
	ymean := 0.0

	n, _ := y.Dims()
	for i := 0; i < n; i++ {
		ymean += y.At(i, 0)
	}
	ymean /= float64(n)

	for i := 0; i < n; i++ {
		diff := y.At(i, 0) - ymean
		ssTotal += diff * diff
		diff = y.At(i, 0) - yPred.At(i, 0)
		ssRes += diff * diff
	}

	if ssTotal == 0 {
		return 1.0
	}
	return 1 - ssRes/ssTotal
}

// GetParameters 返回模型参数
func (p *PLS) GetParameters() map[string]interface{} {
	params := make(map[string]interface{})
	params["num_components"] = p.NumComponents

	// 转换矩阵为切片用于序列化
	if p.XWeights != nil {
		params["x_weights"] = denseToSlice2D(p.XWeights)
	}
	if p.YLoadings != nil {
		params["y_loadings"] = denseToSlice2D(p.YLoadings)
	}

	return params
}

// GetModelType 返回模型类型名称
func (p *PLS) GetModelType() string {
	return "PLS"
}

// 辅助函数：将*mat.Dense转换为[][]float64
func denseToSlice2D(m *mat.Dense) [][]float64 {
	rows, cols := m.Dims()
	slice := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		slice[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			slice[i][j] = m.At(i, j)
		}
	}
	return slice
}
