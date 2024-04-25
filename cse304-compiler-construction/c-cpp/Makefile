1: clean
	@gcc ./infix-to-postfix.c -o ./infix-to-postfix && ./infix-to-postfix

2: clean
	@gcc ./token-count.c -o ./token-count && ./token-count

3: clean
	@g++ ./keyword-count.cpp -o ./keyword-count && ./keyword-count

4: clean
	@g++ ./tokenizer.cpp -o ./tokenizer && ./tokenizer

5: clean
	@g++ ./remove-left-recursions.cpp -o ./remove-left-recursions && ./remove-left-recursions

6: clean
	@g++ ./remove-left-factoring.cpp -o ./remove-left-factoring && ./remove-left-factoring

7: clean
	@g++ ./calculate-first-follow.cpp -o ./calculate-first-follow && ./calculate-first-follow

clean:
	@rm -rf ./infix-to-postfix
	@rm -rf ./token-count 
	@rm -rf ./keyword-count ./output.txt
	@rm -rf ./tokenizer
	@rm -rf ./remove-left-recursions
	@rm -rf ./remove-left-factoring
	@rm -rf ./calculate-first-follow


.DEFAULT_GOAL := clean
.PHONY: 1, 2, 3, 4, 5, 6, 7, clean
