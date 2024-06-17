package test

import (
	"math"
	"testing"

	"github.com/kapilpokhrel/goNN/pkg/loss"
	"gonum.org/v1/gonum/mat"
)

const float64EqualityThreshold = 1e-8

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) <= float64EqualityThreshold
}

func TestMSELoss(t *testing.T) {
	predicted_values := mat.NewDense(1, 3, []float64{1, 1.4, 1.8})
	true_values := mat.NewDense(1, 3, []float64{2, 2, 2})

	output := float64((1 + 0.36 + 0.04) / 3.0)
	result := loss.MSE(true_values, predicted_values)

	if !almostEqual(output, result) {
		t.Fatalf("Expected = %f, Got = %f", output, result)
	}
}

func TestMSEPrime(t *testing.T) {
	predicted_values := mat.NewDense(1, 3, []float64{1, 1.4, 1.8})
	true_values := mat.NewDense(1, 3, []float64{2, 2, 2})

	expected_output := mat.NewDense(1, 3, []float64{-2.0 / 3, -6.0 / 15, -2.0 / 15})
	result := loss.MSE_Prime(true_values, predicted_values)

	if !mat.EqualApprox(expected_output, result, float64EqualityThreshold) {
		t.Fatalf(
			"Output didn't match\nExpected = %v\nGot = %v\n",
			mat.Formatted(expected_output, mat.Prefix("  "), mat.Squeeze()),
			mat.Formatted(result, mat.Prefix("  "), mat.Squeeze()),
		)
	}
}
