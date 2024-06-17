package loss

import (
	"gonum.org/v1/gonum/mat"
)

func MSE(true_val *mat.Dense, pred_val *mat.Dense) float64 {
	var result mat.Dense
	result.Sub(true_val, pred_val)

	result.Apply(func(i, j int, v float64) float64 { return v * v }, &result)

	r, c := result.Dims()
	size := float64(r * c)

	// size can't be zero, as 0 lenght matrix isn't allowed in gonum
	return mat.Sum(&result) / size
}
func MSE_Prime(true_val *mat.Dense, pred_val *mat.Dense) *mat.Dense {
	/*
		Loss(L) = Sigma(i=0 to N)[(truei - predi)^2] / N
		dL/dpredi = -2(truei - predi) / N
	*/

	var result mat.Dense
	result.Sub(pred_val, true_val)

	r, c := result.Dims()
	size := float64(r * c)
	result.Scale(2/size, &result)
	return &result
}
