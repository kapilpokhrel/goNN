package network

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"runtime"

	"github.com/kapilpokhrel/goNN/pkg/layer"
	"github.com/kapilpokhrel/goNN/pkg/loss"
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

// From: https://stackoverflow.com/a/7053871
func GetFunctionName(i interface{}) string {
	return runtime.FuncForPC(reflect.ValueOf(i).Pointer()).Name()
}

type JSONLayer map[string]string //weights and biases are stored in base64 after gonum's MarshalBinary
type JSONNewtork struct {
	Layers []JSONLayer
	Loss   string
}

func (network *Network) Save(fpath string) {
	json_layers := make([]JSONLayer, len(network.Layers))
	var json_network JSONNewtork

	for i, current_layer := range network.Layers {
		json_layer := make(JSONLayer)
		switch reflect.TypeOf(current_layer).String() {
		case "*layer.DenseLayer":
			json_layer["type"] = "Dense"
			original_layer := current_layer.(*layer.DenseLayer)
			weights_bin, _ := original_layer.Weights.MarshalBinary()
			biases_bin, _ := original_layer.Biases.MarshalBinary()

			weights_base64 := make([]byte, base64.StdEncoding.EncodedLen(len(weights_bin)))
			base64.StdEncoding.Encode(weights_base64, weights_bin)
			json_layer["weights"] = string(weights_base64)

			biases_base64 := make([]byte, base64.StdEncoding.EncodedLen(len(biases_bin)))
			base64.StdEncoding.Encode(biases_base64, biases_bin)
			json_layer["biases"] = string(biases_base64)

		case "*layer.TanhLayer":
			json_layer["type"] = "Tanh"
		default:
		}

		json_layers[i] = json_layer
	}
	json_network.Layers = json_layers
	switch GetFunctionName(network.Loss) {
	case "github.com/kapilpokhrel/goNN/pkg/loss.MSE":
		json_network.Loss = "MSE"
	default:
	}

	f, err := os.Create(fpath)
	if err != nil {
		panic(err)
	}
	json_str, _ := json.MarshalIndent(json_network, "", "    ")
	f.WriteString(string(json_str))
}

func (network *Network) Load(fpath string) {
	var json_network JSONNewtork

	data, err := os.ReadFile(fpath)
	if err != nil {
		panic(err)
	}
	json.Unmarshal(data, &json_network)

	layers := make([]layer.Layer, len(json_network.Layers))
	for i, current_layer := range json_network.Layers {
		switch current_layer["type"] {
		case "Dense":
			var denselayer layer.DenseLayer
			weights_bin, _ := base64.StdEncoding.DecodeString(current_layer["weights"])
			biases_bin, _ := base64.StdEncoding.DecodeString(current_layer["biases"])

			var weights mat.Dense
			weights.UnmarshalBinary(weights_bin)
			var biases mat.Dense
			biases.UnmarshalBinary(biases_bin)

			denselayer.Weights = &weights
			denselayer.Biases = &biases
			layers[i] = &denselayer

		case "Tanh":
			var tanhlayer layer.TanhLayer
			layers[i] = &tanhlayer
		default:
		}

	}

	network.Layers = layers
	switch json_network.Loss {
	case "MSE":
		network.Loss = loss.MSE
		network.LossPrime = loss.MSE_Prime
	default:
	}
}
