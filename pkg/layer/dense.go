package layer

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type DenseLayer struct {
	Weights *mat.Dense
	Biases  *mat.Dense
	Input   *mat.Dense
}

func Dense(insize, outsize int) *DenseLayer {
	var layer DenseLayer
	randdata := make([]float64, outsize)
	for i := range randdata {
		randdata[i] = rand.NormFloat64()
	}
	layer.Biases = mat.NewDense(1, outsize, randdata)

	randdata = make([]float64, insize*outsize)
	for i := range randdata {
		randdata[i] = rand.NormFloat64()
	}
	layer.Weights = mat.NewDense(insize, outsize, randdata)
	layer.Input = mat.NewDense(1, insize, nil)

	return &layer
}

func (layer *DenseLayer) Forward(input *mat.Dense) *mat.Dense {
	/*
		This is the method to handle forward progation through the layer.
		This applies the corresponding weights and biases to its inputs and
		returns the gegnerated output
	*/

	/*
		In a general 2 input (x1, x2), 3 output (y1, y2, y3) layer, calculations are as

		y1 = x1 * w1 + x2 * w4 + b1
		y2 = x1 * w2 + x2 * w5 + b2
		y3 = x1 * w3 + x2 * w6 + b3

		The following matrix calculation gives the same result

		[y1 y2 y3] = [x1 x2]*[w1 w2 w3] + [b1 b2 b3]
							 [w4 w5 w6]

	*/

	layer.Input = input

	var output mat.Dense
	output.Mul(layer.Input, layer.Weights)
	output.Add(&output, layer.Biases)
	return &output
}

func (layer *DenseLayer) Backward(output_grad *mat.Dense, rate float64) *mat.Dense {
	/*
		This is the method to handle backward propagation through the layer.
		This receives output gradient (gradient of Loss with respect to the output of layer)
		and learning rate.
		Using chain rule, gradient respect to each weights and biases is calcualted
		and weights and biases are updated.
		Finally, this returns the gradient with respece to the inputs of the layer,
		which acts as the output gradient for the previous layer
		(output of previous layer is input for this layer)
	*/

	/*
		We have [dL/dy1 dL/dy2 dL/dy3] as output gradient
		For weights gradient, we want dL/dw1 ...
		Using chain rule,

		dL/dw1 = dL/dy1 * dy1/dw1
		As we have seen earlier, y1 = x1 * w1 + x2 * w4 + b1

		so,
		dL/dw1 = dL/dy1 * d(x1 * w1 + x2 * w4 + b1)/dw1 = dL/dy1 * x1
		dL/dw2 = dL/dy2 * x1
		dL/dw3 = dL/dy3 * x1
		dL/dw4 = dL/dy1 * x2
		dL/dw5 = dL/dy2 * x2
		dL/dw5 = dL/dy3 * x2

		The following matrix calculation gives the same result as previous

		[dL/dw1 dL/dw2 dL/dw3] = [x1] * [dL/dy1 dL/dy2 dL/dy3]
		[dL/dw4 dL/dw5 dL/dw6]	 [x2]

		Same deduction can be done to find dL/db and dL/dx

	*/

	var weights_grad mat.Dense
	weights_grad.Mul(layer.Input.T(), output_grad)

	var input_grad mat.Dense
	input_grad.Mul(output_grad, layer.Weights.T())

	// weights -= rate * weights_grad
	layer.Weights.Apply(func(i, j int, v float64) float64 {
		return v - rate*weights_grad.At(i, j)
	}, layer.Weights)

	// biases -= rate * output_grad
	layer.Biases.Apply(func(i, j int, v float64) float64 {
		return v - rate*output_grad.At(i, j)
	}, layer.Biases)

	return &input_grad
}
