package types

// Dataset 表示回归数据集
type Dataset struct {
	Features     [][]float64
	Target       []float64
	FeatureNames []string
}

// NewDataset 创建新的数据集
func NewDataset(features [][]float64, target []float64, featureNames []string) *Dataset {
	return &Dataset{
		Features:     features,
		Target:       target,
		FeatureNames: featureNames,
	}
}

// NumSamples 返回样本数量
func (d *Dataset) NumSamples() int {
	if len(d.Features) == 0 {
		return 0
	}
	return len(d.Features)
}

// NumFeatures 返回特征数量
func (d *Dataset) NumFeatures() int {
	if len(d.Features) == 0 {
		return 0
	}
	return len(d.Features[0])
}

// IsValid 检查数据集是否有效
func (d *Dataset) IsValid() bool {
	if d.NumSamples() == 0 || d.NumFeatures() == 0 {
		return false
	}
	if len(d.Target) != d.NumSamples() {
		return false
	}
	return true
}
