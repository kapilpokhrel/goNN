package xor

import (
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"

	"github.com/kapilpokhrel/goNN/pkg/layer"
	"github.com/kapilpokhrel/goNN/pkg/loss"
	"github.com/kapilpokhrel/goNN/pkg/network"
	"gonum.org/v1/gonum/mat"
)

func arange(start, stop, step float64) []float64 {
	N := int(math.Ceil((stop - start) / step))
	rnge := make([]float64, N)
	for x := range rnge {
		rnge[x] = start + step*float64(x)
	}
	return rnge
}

func XOR_net() {
	// make training inputs & outputs
	inputs := make([]*mat.Dense, 4)
	inputs[0] = mat.NewDense(1, 2, []float64{0, 0})
	inputs[1] = mat.NewDense(1, 2, []float64{0, 1})
	inputs[2] = mat.NewDense(1, 2, []float64{1, 0})
	inputs[3] = mat.NewDense(1, 2, []float64{1, 1})

	outputs := make([]*mat.Dense, 4)
	outputs[0] = mat.NewDense(1, 1, []float64{0})
	outputs[1] = mat.NewDense(1, 1, []float64{1})
	outputs[2] = mat.NewDense(1, 1, []float64{1})
	outputs[3] = mat.NewDense(1, 1, []float64{0})

	var xor_network network.Network
	if _, err := os.Stat("examples/xor/xor_trained.json"); errors.Is(err, os.ErrNotExist) {
		layers := []layer.Layer{
			layer.Dense(2, 3),
			layer.Tanh(3),
			layer.Dense(3, 1),
			layer.Tanh(1),
		}

		xor_network = network.Network{
			Layers:    layers,
			Loss:      loss.MSE,
			LossPrime: loss.MSE_Prime,
		}

		xor_network.Train(inputs, outputs, 1000, 0.01)

		xor_network.Save("examples/xor/xor_trained.json")
		fmt.Println("Training Finished!!")
	} else {
		xor_network.Load("examples/xor/xor_trained.json")
	}

	f, err := os.Create("./examples/xor/xor-boundry.csv")
	if err != nil {
		panic(err)
	}
	f.WriteString("x,y,z\n")
	for _, x := range arange(0., 1., 0.02) {
		for _, y := range arange(0., 1., 0.02) {
			input := mat.NewDense(1, 2, []float64{x, y})
			output := xor_network.Predict(input)
			f.WriteString(fmt.Sprintf("%f,%f,%f\n", x, y, output.At(0, 0)))
		}
	}

	ch := make(chan []byte)
	go func() {
		cmd := exec.Command("python", "examples/xor/plot.py")
		out, _ := cmd.CombinedOutput()
		ch <- out
	}()
	fmt.Println(<-ch)

}
