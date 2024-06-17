package network

import (
	"fmt"

	"github.com/kapilpokhrel/goNN/pkg/layer"
	"gonum.org/v1/gonum/mat"
)

type Network struct {
	Layers    []layer.Layer
	Loss      func(*mat.Dense, *mat.Dense) float64
	LossPrime func(*mat.Dense, *mat.Dense) *mat.Dense
}

func (network *Network) Predict(input *mat.Dense) *mat.Dense {
	result := input
	for _, layer := range network.Layers {
		var err error
		result, err = layer.Forward(result)
		if err != nil {
			fmt.Println(err)
			return nil
		}
	}
	return result
}

func (network *Network) BackProp(out_grad *mat.Dense, rate float64) {
	in_grad := out_grad
	for i := len(network.Layers) - 1; i >= 0; i-- {
		layer := network.Layers[i]
		in_grad = layer.Backward(in_grad, rate)
	}
}

func (network *Network) Train(inputs []*mat.Dense, outputs []*mat.Dense, epoch int, rate float64) {
	for i := 0; i < epoch; i++ {
		err := float64(0)
		for j, input := range inputs {
			result := network.Predict(input)
			err += network.Loss(outputs[j], result)

			out_grad := network.LossPrime(outputs[j], result)
			network.BackProp(out_grad, rate)
		}
		fmt.Printf("Epoch = (%d/%d), error = %f\n", i+1, epoch, err/float64(len(inputs)))
	}
}
