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

func TestDenseLayerBackwardPropgation(t *testing.T) {
	var layer layer.DenseLayer

	layer.Weights = mat.NewDense(2, 3, []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6})
	layer.Biases = mat.NewDense(1, 3, []float64{0.1, 0.2, 0.3})
	layer.Input = mat.NewDense(1, 2, []float64{1, 2})

	out_grad := mat.NewDense(1, 3, []float64{0.2, 0.4, 0.6})
	result_inputgrad := layer.Backward(out_grad, 0.1)

	expected_weights := mat.NewDense(2, 3, []float64{0.08, 0.16, 0.24, 0.36, 0.42, 0.48})
	expected_biases := mat.NewDense(1, 3, []float64{0.08, 0.16, 0.24})

	expected_inputgrad := mat.NewDense(1, 2, []float64{0.28, 0.64})

	if !mat.EqualApprox(expected_weights, layer.Weights, 1e-14) {
		t.Fatalf(
			"Weights didn't update correctly\nExpected = %v\nGot = %v\n",
			mat.Formatted(expected_weights, mat.Prefix("  "), mat.Squeeze()),
			mat.Formatted(layer.Weights, mat.Prefix("  "), mat.Squeeze()),
		)
	}

	if !mat.EqualApprox(expected_biases, layer.Biases, 1e-14) {
		t.Fatalf(
			"Biases didn't update correctly\nExpected = %v\nGot = %v\n",
			mat.Formatted(expected_biases, mat.Prefix("  "), mat.Squeeze()),
			mat.Formatted(layer.Biases, mat.Prefix("  "), mat.Squeeze()),
		)
	}

	if !mat.EqualApprox(expected_inputgrad, result_inputgrad, 1e-14) {
		t.Fatalf(
			"Input gradient didn't match\nExpected = %v\nGot = %v\n",
			mat.Formatted(expected_inputgrad, mat.Prefix("  "), mat.Squeeze()),
			mat.Formatted(result_inputgrad, mat.Prefix("  "), mat.Squeeze()),
		)
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

func TestTanhLayerBackwardPropagation(t *testing.T) {
	var layer layer.TanhLayer

	inputs := [][]float64{
		{1, 1.4, 1.8},
		{0.1, 2, 3, 4, 1.1},
		{-1, 0, -3},
	}

	grads := [][]float64{
		{1, 0.2, -0.8},
		{-0.1, 0.02, 0.3, 0.5, 0.11},
		{0, 0, -0.25},
	}

	outputs := make([][]float64, 3)
	for i := 0; i < 3; i++ {
		output_i := make([]float64, len(inputs[i]))
		for j, value := range inputs[i] {
			output_i[j] = grads[i][j] * (1 - math.Pow(math.Tanh(value), 2))
		}
		outputs[i] = output_i
	}

	for i, input := range inputs {
		input_matrix := mat.NewDense(1, len(input), input)
		grad_matrix := mat.NewDense(1, len(input), grads[i])
		expected_output := mat.NewDense(1, len(input), outputs[i])

		layer.Input = input_matrix
		result := layer.Backward(grad_matrix, 0.1)

		if !mat.EqualApprox(expected_output, result, 1e-14) {
			t.Fatalf(
				"Output didn't match\nExpected = %v\nGot = %v\n",
				mat.Formatted(expected_output, mat.Prefix("  "), mat.Squeeze()),
				mat.Formatted(result, mat.Prefix("  "), mat.Squeeze()),
			)
		}
	}
}
