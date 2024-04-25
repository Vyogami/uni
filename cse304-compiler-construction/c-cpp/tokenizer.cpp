#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>

using namespace std;

bool isFunction(const string& word) {
    return word.find('(') != string::npos;
}

bool isMultComment(const string& word)
{
   return word.find("/*")!=string::npos;
}
bool isComment(const string& word)
{
    return word.find("//")!=string::npos;
}

bool isLiteral(const string& word) {
    return isdigit(word[0]);
     if (word.size() >= 3 && word[0] == '\'' && word[2] == '\'') {
        return true;
    }
}


int main() {
    int func_count = 0;
    int keyword_count = 0;
    int comment_count=0;
    int lit_count=0;
    bool multi_comment=0;
     unordered_set<string> keywords = {
        "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor",
        "bool", "break", "case", "catch", "char", "char16_t", "char32_t", "class",
        "compl", "const", "constexpr", "const_cast", "continue", "decltype", "default",
        "delete", "do", "double", "dynamic_cast", "else", "enum", "explicit", "export",
        "extern", "false", "float", "for", "friend", "goto", "if", "inline", "int",
        "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq", "nullptr",
        "operator", "or", "or_eq", "private", "protected", "public", "register",
        "reinterpret_cast", "return", "short", "signed", "sizeof", "static", "static_assert",
        "static_cast", "struct", "switch", "template", "this", "thread_local", "throw",
        "true", "try", "typedef", "typeid", "typename", "union", "unsigned", "using",
        "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq"
    };

    ifstream inputFile("./tokenizer.cpp");
    ofstream outputFile("output.txt");
    int wordCount = 0;

    if (inputFile.is_open() && outputFile.is_open()) {
        string line;
        while (getline(inputFile, line, ';')) {
            istringstream iss(line);
            string word;
            while (iss >> word) {
                cout << word << " ";
                if (keywords.find(word) != keywords.end()) {
                    keyword_count++;
                    cout << " (C++ keyword)" << endl;
                } else if (isFunction(word)) {
                    func_count++;
                    cout << " (C++ function)" << endl;
                }
                else if(isComment(word))
                {
                    comment_count++;
                    cout<< " (C++ Comment)"<<endl;
                }
                else if(isMultComment(word))
                {
                    cout<<"(C++ Multi-line comment)"<<endl;
                    multi_comment=true;
                }
                if(multi_comment==true)
                {
                    cout<<"Inside Multiple Comments"<<endl;
                    if(word.find("*/")!=string::npos)
                    {
                        cout<<"C++ MULTI-COMMENT CLOSED"<<endl;
                        multi_comment=false;
                        continue;
                    }
                }
                if(!isComment(word) && !isMultComment(word) && !multi_comment)
                {   if(isLiteral(word))
                    {
                        lit_count++;
                        cout<<" (C++ Literal)"<<endl;
                    }      
                        outputFile<<word<<"\n";
                }
                wordCount++;
            }
        }

        inputFile.close();
        outputFile.close();

        cout << "\nFile copied successfully." << endl;
        cout << "Total count: " << wordCount << endl;
        cout<< "Total functions: " << func_count << endl;
        cout << "Total keywords: " << keyword_count << endl;
        cout << "Total comments: " << comment_count << endl;
        cout<<"Total Literals: "<<lit_count<<endl;
    } else {
        cout << "Failed to open the files." << endl;
    }

    return 0;
}
