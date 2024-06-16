package layer

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type TanhLayer struct {
	Input *mat.Dense
}

func Tanh(insize int) *TanhLayer {
	var layer TanhLayer
	layer.Input = mat.NewDense(1, insize, nil)

	return &layer
}

func (layer *TanhLayer) Forward(input *mat.Dense) *mat.Dense {
	layer.Input = input

	var result mat.Dense

	// result = tanh(input) ; for element in input
	result.Apply(func(i, j int, v float64) float64 { return math.Tanh(v) }, input)
	return &result
}

func (layer *TanhLayer) Backward(output_grad *mat.Dense, rate float64) *mat.Dense {
	/*
		dL/dinput = dL/dy * dy/dinput

		y = tanh(input)
		so, dy/dinput (y') = sech^2(input) = 1 - tanh^2(input)

		dL/dinput = dL/dy * (1 - tanh^2(input))
	*/

	var y_prime mat.Dense
	y_prime.Apply(func(i, j int, v float64) float64 { return 1.0 - math.Tanh(v)*math.Tanh(v) }, layer.Input)

	var result mat.Dense
	result.MulElem(output_grad, &y_prime)

	return &result
}
