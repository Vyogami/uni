package main

import (
	"fmt"
)

type Stack []string

func (st *Stack) IsEmpty() bool {
	return len(*st) == 0
}

func (st *Stack) Push(str string) {
	*st = append(*st, str)
}

func (st *Stack) Pop() bool {
	if st.IsEmpty() {
		return false
	} else {
		index := len(*st) - 1
		*st = (*st)[:index]
		return true
	}
}

func main() {
	fmt.Println("Let's play!")
}
