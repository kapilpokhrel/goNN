package network

import (
	"github.com/kapilpokhrel/goNN/pkg/layer"
	"gonum.org/v1/gonum/mat"
)

type Network struct {
	Layers    []layer.Layer
	Loss      func(*mat.Dense, *mat.Dense) float64
	LossPrime func(*mat.Dense, *mat.Dense) *mat.Dense
}

func (netowrk *Network) Predict(input *mat.Dense) {

}
func (network *Network) Train(input []*mat.Dense, output []*mat.Dense, epoch int, rate float64) {

}
