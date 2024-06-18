package dsr

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"

	"github.com/kapilpokhrel/goNN/pkg/layer"
	"github.com/kapilpokhrel/goNN/pkg/loss"
	"github.com/kapilpokhrel/goNN/pkg/network"
	"gonum.org/v1/gonum/mat"
)

func GenSequenceInandOut() ([]float64, []float64) {
	// Target Set
	// Blue: 0.1
	// Green: 0.2
	TargetSet := []float64{0.1, 0.2}

	// Yellow: 0.3
	// Red: 0.4
	DistractorSet := []float64{0.3, 0.4}

	// Grey: 0.5
	// Black: 0.6
	PromptSet := []float64{0.5, 0.6}

	target := generateRandomFixedLengthSlice(TargetSet, 2)
	distractor := generateRandomFixedLengthSlice(DistractorSet, 6)
	sequence := append(target, distractor...)

	rand.Shuffle(len(sequence), func(i, j int) {
		sequence[i], sequence[j] = sequence[j], sequence[i]
	})

	sequence = append(sequence, PromptSet...)

	// Generating output
	output := make([]float64, len(sequence))
	obtained_target := make([]float64, 2)
	index := 0
	for _, value := range sequence {
		if value == 0.1 || value == 0.2 {
			obtained_target[index] = value
			index++
		}
	}
	output[len(output)-2] = obtained_target[0]
	output[len(output)-1] = obtained_target[1]

	return sequence, output
}

func generateRandomFixedLengthSlice(src []float64, length int) []float64 {
	result := make([]float64, length)

	for i := 0; i < length; i++ {
		result[i] = src[rand.Intn(len(src))]
	}

	return result
}

func roundFloat(val float64, precision uint) float64 {
	ratio := math.Pow(10, float64(precision))
	return math.Round(val*ratio) / ratio
}

func DSR_net() {

	// make training inputs & outputs
	inputs := make([]*mat.Dense, 2000)
	outputs := make([]*mat.Dense, 2000)
	for i := 0; i < 2000; i++ {
		in, out := GenSequenceInandOut()
		inputs[i] = mat.NewDense(1, 10, in)
		outputs[i] = mat.NewDense(1, 10, out)
	}

	var dsr_network network.Network
	if _, err := os.Stat("examples/dsr/dsr_trained.json"); errors.Is(err, os.ErrNotExist) {
		layers := []layer.Layer{
			layer.Dense(10, 30),
			layer.Tanh(30),
			layer.Dense(30, 30),
			layer.Tanh(30),
			layer.Dense(30, 10),
			layer.Tanh(10),
		}

		dsr_network = network.Network{
			Layers:    layers,
			Loss:      loss.MSE,
			LossPrime: loss.MSE_Prime,
		}

		dsr_network.Train(inputs, outputs, 20000, 0.5)

		dsr_network.Save("examples/dsr/dsr_trained.json")
		fmt.Println("Training Finished!!")
	} else {
		dsr_network.Load("examples/dsr/dsr_trained.json")
	}

	for i := 0; i < 15; i++ {
		in, _ := GenSequenceInandOut()
		predicted_output := dsr_network.Predict(mat.NewDense(1, 10, in))

		var rounded_out mat.Dense
		rounded_out.Apply(func(i, j int, v float64) float64 { return roundFloat(v, 1) }, predicted_output)

		fmt.Printf("%v, %v\n", in, mat.Formatted(&rounded_out, mat.Prefix("  "), mat.Squeeze()))
	}
}
