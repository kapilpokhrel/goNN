package test

import (
	"math"
	"testing"

	"github.com/kapilpokhrel/goNN/pkg/layer"
	"gonum.org/v1/gonum/mat"
)

func TestDenseLayerForwardPropgation(t *testing.T) {
	var layer layer.DenseLayer

	layer.Weights = mat.NewDense(2, 3, []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6})
	layer.Biases = mat.NewDense(1, 3, []float64{0.1, 0.2, 0.3})

	input := mat.NewDense(1, 2, []float64{1, 2})
	expected_output := mat.NewDense(1, 3, []float64{1, 1.4, 1.8})
	result, err := layer.Forward(input)

	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}

	// Check if the ouput matches the expected result
	if !mat.EqualApprox(expected_output, result, 1e-14) {
		t.Fatalf(
			"Output didn't match\nExpected = %v\nGot = %v\n",
			mat.Formatted(expected_output, mat.Prefix("  "), mat.Squeeze()),
			mat.Formatted(result, mat.Prefix("  "), mat.Squeeze()),
		)
	}

	// Check whether the input is stored in the layer for backward propagation
	if !mat.Equal(input, layer.Input) {
		t.Fatalf("Input not correctly stored in layer")
	}

	// Checking with wrong dimension input
	input = mat.NewDense(1, 3, nil)
	_, err = layer.Forward(input)
	if err == nil {
		t.Fatalf("expected dimension error, got none")
	}
}

func TestTanhLayerForwardPropagation(t *testing.T) {
	var layer layer.TanhLayer

	inputs := [][]float64{
		{1, 1.4, 1.8},
		{0.1, 2, 3, 4, 1.1},
		{-1, 0, -3},
	}

	outputs := [][]float64{
		{math.Tanh(1), math.Tanh(1.4), math.Tanh(1.8)},
		{math.Tanh(0.1), math.Tanh(2), math.Tanh(3), math.Tanh(4), math.Tanh(1.1)},
		{math.Tanh(-1), math.Tanh(0), math.Tanh(-3)},
	}

	for i, input := range inputs {
		input_matrix := mat.NewDense(1, len(input), input)
		expected_output := mat.NewDense(1, len(input), outputs[i])

		result, err := layer.Forward(input_matrix)

		if err != nil {
			t.Errorf("expected no error, got %v", err)
		}

		if !mat.EqualApprox(expected_output, result, 1e-14) {
			t.Fatalf(
				"Output didn't match\nExpected = %v\nGot = %v\n",
				mat.Formatted(expected_output, mat.Prefix("  "), mat.Squeeze()),
				mat.Formatted(result, mat.Prefix("  "), mat.Squeeze()),
			)
		}
	}
}
