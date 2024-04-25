#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#define MAX_SIZE 100

typedef struct {
    char stack[MAX_SIZE];
    int top;
} Stack;

void initialize(Stack* s) {
    s->top = -1;
}

void push(Stack* s, char item) {
    if (s->top < MAX_SIZE - 1) {
        s->stack[++(s->top)] = item;
    }
}

char pop(Stack* s) {
    if (s->top >= 0) {
        return s->stack[(s->top)--];
    }
    return '\0';  // Empty stack
}

int isOperator(char ch) {
    return (ch == '+' || ch == '-' || ch == '*' || ch == '/');
}

int precedence(char ch) {
    if (ch == '+' || ch == '-')
        return 1;
    if (ch == '*' || ch == '/')
        return 2;
    return 0;
}

void infixToPostfix(const char* infix, char* postfix) {
    Stack operatorStack;
    initialize(&operatorStack);

    int i = 0, j = 0;
    char ch;

    while ((ch = infix[i++]) != '\0') {
        if (isalnum(ch)) {
            postfix[j++] = ch;
        } else if (isOperator(ch)) {
            while (precedence(ch) <= precedence(operatorStack.stack[operatorStack.top]) && operatorStack.top != -1) {
                postfix[j++] = pop(&operatorStack);
            }
            push(&operatorStack, ch);
        }
    }

    while (operatorStack.top != -1) {
        postfix[j++] = pop(&operatorStack);
    }

    postfix[j] = '\0';  // Null-terminate the postfix expression
}

int main() {
    const char* infixExpression = "a+b*c-d/e";
    char postfixExpression[MAX_SIZE];

    infixToPostfix(infixExpression, postfixExpression);

    printf("Infix Expression: %s\n", infixExpression);
    printf("Postfix Expression: %s\n", postfixExpression);

    return 0;
}
