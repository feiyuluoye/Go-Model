package gomodel

import (
	"gonum.org/v1/gonum/mat"
)

// NewDenseFromSlice creates a new Dense matrix from a slice
func NewDenseFromSlice(rows, cols int, data []float64) *mat.Dense {
	return mat.NewDense(rows, cols, data)
}

// NewDenseFromArrays creates a new Dense matrix from 2D array
func NewDenseFromArrays(data [][]float64) *mat.Dense {
	if len(data) == 0 || len(data[0]) == 0 {
		return nil
	}
	
	rows := len(data)
	cols := len(data[0])
	
	// Flatten the 2D array
	flatData := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			flatData[i*cols+j] = data[i][j]
		}
	}
	
	return mat.NewDense(rows, cols, flatData)
}

// NewVecDenseFromSlice creates a new VecDense from a slice
func NewVecDenseFromSlice(data []float64) *mat.VecDense {
	return mat.NewVecDense(len(data), data)
}

// MatrixToArrays converts a Dense matrix to 2D array
func MatrixToArrays(matrix *mat.Dense) [][]float64 {
	rows, cols := matrix.Dims()
	result := make([][]float64, rows)
	
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			result[i][j] = matrix.At(i, j)
		}
	}
	
	return result
}

// VectorToSlice converts a VecDense to slice
func VectorToSlice(vector *mat.VecDense) []float64 {
	n := vector.Len()
	result := make([]float64, n)
	
	for i := 0; i < n; i++ {
		result[i] = vector.AtVec(i)
	}
	
	return result
}
