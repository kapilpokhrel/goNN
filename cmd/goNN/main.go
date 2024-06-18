package main

import (
	"fmt"

	"github.com/kapilpokhrel/goNN/examples/dsr"
	"github.com/kapilpokhrel/goNN/examples/xor"
)

func main() {
	fmt.Println("Choose an example to run:")
	fmt.Println("1. Learning XOR")
	fmt.Println("2. Distracted Sequence Recall task")

	var inp int
	fmt.Scan(&inp)

	switch inp {
	case 1:
		fmt.Println("Running XOR learning task")
		xor.XOR_net()
	case 2:
		fmt.Println("Running Distracted Sequence Recall task")
		dsr.DSR_net()
	default:
	}
}
