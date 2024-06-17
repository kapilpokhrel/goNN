package layer

import (
	"gonum.org/v1/gonum/mat"
)

type Layer interface {
	Forward(input *mat.Dense) (*mat.Dense, error)
	Backward(out_grad *mat.Dense, rate float64) *mat.Dense
}
