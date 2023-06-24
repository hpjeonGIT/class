## ANTLR Programming Masterclass with Python
- Instructor: Lucas Bazilio

## Section 1: Course Introduction

1. Introduction to the Course

2. Course Agenda
- Basics of Compiler Design
- Lexical, Syntactic and Semantic Analysis
- Master ANTLR to build interpreters, programming languages, ...
- Concept of visitor and listener
- Programming language from Zero

## Section 2: Compiler Fundamentals

3. Introduction to Compilation
- We will create language processors:
  - Vocabulary Definition
  - Grammar Definition
  - Generawtion of the abstract syntax tree
  - Interpretation through the path of the tree

4. Compiler Example
- test.c
```c
 int func(int a, int b) {
   a = a + b;
   return a;
}
```
- gcc -S test.c 
- test.s
```asm
func:
.LFB0:
  .cfi_startproc
  endbr64
  pushq  %rbp
  .cfi_def_cfa_offset 16
  .cfi_offset 6, -16
  movq  %rsp, %rbp
  .cfi_def_cfa_register 6
  movl  %edi, -4(%rbp)
  movl  %esi, -8(%rbp)
  movl  -8(%rbp), %eax
  addl  %eax, -4(%rbp)
  movl  -4(%rbp), %eax
  popq  %rbp
  .cfi_def_cfa 7, 8
  ret
  .cfi_endproc
```

5. Interpreter Concept

6. Syntax of a Programming Language
- The syntax of a programming language is the set of rules that define the combinations of symbols that are considered correctly structured constructs
- Syntax is often specified using a context free grammar (CFG)
- The most basic elements (words) are specified through regular expressions

## Section 3: ANTLR4 and Python3 Installation

7. Install ANTLR4 and Python3
- sudo apt install antlr4
  - dpkg -L antlr4
  - 4.7.2 is installed as of June 2023 at Ubuntu20
  - NOT compatible with python3-runtime which is 4.13
- Download https://www.antlr.org/download/antlr-4.13.0-complete.jar
- `alias antlr4='java -jar ~/sw_local/antlr-4.13.0-complete.jar'`
- `alias grun='java org.antlr.v4.runtime.misc.TestRig'`
- `pip3 install antlr4-python3-runtime`

## Section 4: Introduction to ANTLR

8. First Program in ANTLR
- Expr.g
```g4
grammar Expr;
root : expr EOF ;         # rule 1
expr : expr PLUS expr     # rule 2
    | NUM
    ;
NUM : [0-9]+ ;            # token definition 1
PLUS : '+' ;              # token definition 2
WS : [ \n]+ -> skip ;     # token definition 3
```
- The name of *.g file (Expr) must match the name of grammar (Expr)
- expr: definition of the grammar for the sum of natural numbers
- skip: tell the scanner that the WS token should not reach the parser
- root: to process the end of file
- Compilation in python3
  - `antlr4 -Dlanguage=Python3 -no-listener Expr.g`
  - Generates: 
    - ExprLexer.py and ExprLexer.tokens
    - ExprParser.py and Expr.tokens
- Test script
```py3
from antlr4     import *
from ExprLexer  import ExprLexer
from ExprParser import ExprParser
input_stream = InputStream(input('? '))
lexer = ExprLexer(input_stream)
token_stream = CommonTokenStream(lexer)
parser = ExprParser(token_stream)
tree = parser.root()
print(tree.toStringTree(recog=parser))
```
- Testing:
```bash
$ python3 test.py
? 2 + 3
(root (expr (expr 2) + (expr 3)) <EOF>)
$ python3 test.py
? 1 +  4+ 2 + 3
(root (expr (expr (expr (expr 1) + (expr 4)) + (expr 2)) + (expr 3)) <EOF>)
```

9. Notes on the Entry
- One single line: `input_stream = InputStream(input('? '))`
- Stdin: `input_stream = StdinStream()`
- A file passed as a parameter: `input_stream = FileStream(sys.argv[1])`
- Files with accents: `input_stream = FileStream(sys.arvg[1], encoding='utf-8')`

10. Visitors
- Visitors are tree walkers, a mechanism to traverse the AST
- `antlr4 -Dlanguage=Python3 -no-listener -visitor Expr.g` will compile the grammar and generate the visitor template (ExprVisitor.py)
```py
# Generated from Expr.g by ANTLR 4.13.0
from antlr4 import *
if "." in __name__:
    from .ExprParser import ExprParser
else:
    from ExprParser import ExprParser
# This class defines a complete generic visitor for a parse tree produced by ExprParser.
class ExprVisitor(ParseTreeVisitor):
    # Visit a parse tree produced by ExprParser#root.
    def visitRoot(self, ctx:ExprParser.RootContext):
        return self.visitChildren(ctx)
    # Visit a parse tree produced by ExprParser#expr.
    def visitExpr(self, ctx:ExprParser.ExprContext):
        return self.visitChildren(ctx)
del ExprParser
```
- visitExpr is the callback associated with the rule `Expr` to visit it

11. Evaluator Visitor
- EvalVisitor.py
```py
if __name__ is not None and "." in __name__:
  from .ExprParser import ExprParser
  from .ExprVisitor import ExprVisitor
else:
  from ExprParser import ExprParser
  from ExprVisitor import ExprVisitor
class EvalVisitor(ExprVisitor):
  def visitRoot(self,ctx):   # for the rule 1 from grammar. Inherits from ExprVisitor
    l = list(ctx.getChildren())
    print("Root Visited")
    print("ANS=",self.visit(l[0])) # there is only list item as the rule 1 has one one expr => 'root : expr EOF ;'
  def visitExpr(self,ctx):   # for the rule 2 from grammar. Inherits from ExprVisitor
    l = list(ctx.getChildren())
    if len(l) == 1:  # NUM
      print("NUM found")
      return int(l[0].getText())
    else: # len(l) == 3, expr, PLUS, expr from 'expr : expr PLUS expr'
      print("EXPR found")
      return self.visit(l[0]) + self.visit(l[2])
```
- Visitor information
  - With the children: `ctx.getChildren()`
  - Node text: `n.getText()`
  - Node token in text format: `ExprParser.symbolicNames[n.getSymbol().type]`
  - Internal index of the PLUS token for the parser: `ExprParser.PLUS`
- test.py
```py
from antlr4     import *
from ExprLexer  import ExprLexer
from ExprParser import ExprParser
from EvalVisitor import EvalVisitor
input_stream = InputStream(input('? '))
lexer = ExprLexer(input_stream)
token_stream = CommonTokenStream(lexer)
parser = ExprParser(token_stream)
tree = parser.root()
visitor = EvalVisitor()
visitor.visit(tree)
```
```bash
$ python3 test.py 
? 1+2+3+4
Root Visited
EXPR found
EXPR found
EXPR found
NUM found
NUM found
NUM found
NUM found
ANS= 10
```

## Section 5: Elementary Interpreters

12. Interpreter for Addition and Subtraction
- Expr.g
```
grammar Expr;
root : expr EOF ;
expr : expr PLUS expr
    | expr SUB expr
    | NUM
    ;
NUM : [0-9]+ ;
PLUS : '+' ;
SUB : '-' ;
WS : [ \n]+ -> skip ;
```
- `antlr4 -Dlanguage=Python3 -no-listener -visitor Expr.g`
  - Will produce ExprVisitor.py. We use this class and make a subclass 
- visitor.py
```py
if __name__ is not None and "." in __name__:
  from .ExprParser import ExprParser
  from .ExprVisitor import ExprVisitor
else:
  from ExprParser import ExprParser
  from ExprVisitor import ExprVisitor
class ElementalVisitor(ExprVisitor):
  def visitRoot(self,ctx):
    print("Root Visited")
    l = list(ctx.getChildren())
    print("ANS=",self.visit(l[0]))
  def visitExpr(self,ctx):
    l = list(ctx.getChildren())
    if len(l) == 1:
      print("NUM found")
      return int(l[0].getText())
    else: # len(l) == 3
      if (l[1].getText() == '+'):
        print("PLUS found")
        return self.visit(l[0]) + self.visit(l[2])
      else:
        print("SUB found")
        return self.visit(l[0]) - self.visit(l[2])
```
- test.py
```py
from antlr4     import *
from ExprLexer  import ExprLexer
from ExprParser import ExprParser
from visitor    import ElementalVisitor
input_stream = InputStream(input('? '))
lexer        = ExprLexer(input_stream)
token_stream = CommonTokenStream(lexer)
parser       = ExprParser(token_stream)
tree         = parser.root()
visitor      = ElementalVisitor()
visitor.visit(tree)
```
- Testing:
```bash
$ python3 test.py 
? 1+2+3-4
Root Visited
SUB found
PLUS found
PLUS found
NUM found
NUM found
NUM found
NUM found
ANS= 2
```
13. Introduction to Labels
- Labels are mechanisms that help us to clarify the code
- Must be the first character as Capital and the others are small
- Must be consistent through visitor function names
```
expr : expr PLUS expr
    | NUM
```
  - Which case is for PLUS or NUM ?
  - Instead of conditional statements in the VisitExpr(self,ctx), each token case can be accessed directly, using Labels
- When labes are used, all labels must be given for alternatives
- Label as a commentd in the right column
      
14. Use of Labels
- Expr.g
```
grammar Expr;
root: expr EOF ;
expr: expr PLUS expr # Sum
    | expr SUB expr  # Sub
    | NUM            # Value
    ;
NUM : [0-9]+ ;
PLUS: '+' ;
SUB : '-' ;
WS  : [ \n]+ -> skip ;
```
- visitor.py
```py
if __name__ is not None and "." in __name__:
  from .ExprParser import ExprParser
  from .ExprVisitor import ExprVisitor
else:
  from ExprParser import ExprParser
  from ExprVisitor import ExprVisitor
class ElementalVisitor(ExprVisitor):
  def visitRoot(self,ctx):
    print("Root Visited")
    l = list(ctx.getChildren())
    print("ANS=",self.visit(l[0]))
  def visitValue(self,ctx):
    print("NUM found")
    l = list(ctx.getChildren())
    return int(l[0].getText())
    l = list(ctx.getChildren())
  def visitSum(self,ctx):
    print("PLUS found")
    l = list(ctx.getChildren())
    return self.visit(l[0]) + self.visit(l[2])
  def visitSub(self,ctx):
    print("SUB found")
    l = list(ctx.getChildren())
    return self.visit(l[0]) - self.visit(l[2])
```
- test.py is same above
- Testing:
```bash
python3 test.py 
? 1+2+3-4
Root Visited
SUB found
PLUS found
PLUS found
NUM found
NUM found
NUM found
NUM found
ANS= 2
```

15. Interpreter for the Product
- Order of mathematics
  - */% are done earlier than +/-  
  - The order in a rule indicates the priority of the order
- Expr.g
```
grammar Expr;
root: expr EOF ;
expr: expr MUL expr  # Mul
    | expr PLUS expr # Sum
    | expr SUB expr  # Sub
    | NUM            # Value
    ;
NUM : [0-9]+ ;
PLUS: '+' ;
SUB : '-' ;
MUL : '*' ;
WS  : [ \n]+ -> skip ;
```
- visitory.py
```py
if __name__ is not None and "." in __name__:
  from .ExprParser import ExprParser
  from .ExprVisitor import ExprVisitor
else:
  from ExprParser import ExprParser
  from ExprVisitor import ExprVisitor
class ElementalVisitor(ExprVisitor):
  def visitRoot(self,ctx):
    print("Root Visited")
    l = list(ctx.getChildren())
    print("ANS=",self.visit(l[0]))
  def visitValue(self,ctx):
    print("NUM found")
    l = list(ctx.getChildren())
    return int(l[0].getText())
    l = list(ctx.getChildren())
  def visitMul(self,ctx):
    print("MULT found")
    l = list(ctx.getChildren())
    return self.visit(l[0]) * self.visit(l[2])
  def visitSum(self,ctx):
    print("PLUS found")
    l = list(ctx.getChildren())
    return self.visit(l[0]) + self.visit(l[2])
  def visitSub(self,ctx):
    print("SUB found")
    l = list(ctx.getChildren())
    return self.visit(l[0]) - self.visit(l[2])
```
- test.py
```py
from antlr4     import *
from ExprLexer  import ExprLexer
from ExprParser import ExprParser
from visitor    import ElementalVisitor
input_stream = InputStream(input('? '))
lexer        = ExprLexer(input_stream)
token_stream = CommonTokenStream(lexer)
parser       = ExprParser(token_stream)
tree         = parser.root()
visitor      = ElementalVisitor()
visitor.visit(tree)
```
- Testing:
```bash
$ python3 test.py
? 1+ 2*3-4
Root Visited
SUB found
PLUS found
NUM found
MULT found
NUM found
NUM found
NUM found
ANS= 3
```

16. Interpreter for the Division
- MUL and DIV have higher priorties than PLUS and SUB
  - Make sure the order in the rule 2
- Expr.g
```g4
grammar Expr;
root: expr EOF ;
expr: expr MUL expr  # Mul
    | expr DIV expr  # Div
    | expr PLUS expr # Sum
    | expr SUB expr  # Sub
    | NUM            # Value
    ;
NUM : [0-9]+ ;
PLUS: '+' ;
SUB : '-' ;
MUL : '*' ;
DIV : '/' ;
WS  : [ \n]+ -> skip ;
```
- visitor.py
```py
if __name__ is not None and "." in __name__:
  from .ExprParser import ExprParser
  from .ExprVisitor import ExprVisitor
else:
  from ExprParser import ExprParser
  from ExprVisitor import ExprVisitor
class ElementalVisitor(ExprVisitor):
  def visitRoot(self,ctx):
    print("Root Visited")
    l = list(ctx.getChildren())
    print("ANS=",self.visit(l[0]))
  def visitValue(self,ctx):
    print("NUM found")
    l = list(ctx.getChildren())
    return int(l[0].getText())
    l = list(ctx.getChildren())
  def visitMul(self,ctx):
    print("MUL found")
    l = list(ctx.getChildren())
    return self.visit(l[0]) * self.visit(l[2])
  def visitDiv(self,ctx):
    print("DIV found")
    l = list(ctx.getChildren())
    return self.visit(l[0]) / self.visit(l[2])
  def visitSum(self,ctx):
    print("PLUS found")
    l = list(ctx.getChildren())
    return self.visit(l[0]) + self.visit(l[2])
  def visitSub(self,ctx):
    print("SUB found")
    l = list(ctx.getChildren())
    return self.visit(l[0]) - self.visit(l[2])
```
- test.py is same above
- Testing:
```bash
$ python3 test.py 
? 1 + 2*6/3 - 2
Root Visited
SUB found
PLUS found
NUM found
DIV found
MUL found
NUM found
NUM found
NUM found
NUM found
ANS= 3.0
```

## Section 6: Interpreter with Variables

17. Initial Interpreter - Grammar
- We want a folling evaluation/execution mechanism
```
x := 3 + 5
write x
y := 6 - 4
write y
```
  - Assignment or print command
  - Use a dictionary as a symbol table 
```py
import opertor
ops = {'+': operator.add, '-': operator.sub, \
       '*': operator.mul, \
       '/': operator.truediv, '^': operator.pow }
```
  - [a-z]+ : positive Kleene's Closure. 1 or more
  - [a-z]* : 0 or more
- Expr.g
```g4
grammar Expr;
root: action+ EOF;
action: NAME ':=' expr
      | 'write' NAME
      ;
expr: <assoc=right> expr '^' expr
    | expr ('*'|'/') expr
    | expr ('+'|'-') expr  
    | NUM
    ;
NUM:  [0-9]+;
NAME: [a-z]+;
WS  : [ \n]+ -> skip ;
```
- `antlr4 -Dlanguage=Python3 -no-listener -visitor Expr.g`

18. Initial Interpreter - Vistor
- Define a visitor for the evaluation/execution mechanism above
- VarVisitor.py
```py
import operator
ops = {'+': operator.add, '-': operator.sub,'*': operator.mul, '/': operator.truediv, '^': operator.pow }
if __name__ is not None and "." in __name__:
  from .ExprParser import ExprParser
  from .ExprVisitor import ExprVisitor
else:
  from ExprParser import ExprParser
  from ExprVisitor import ExprVisitor
class VarVisitor(ExprVisitor):
  def __init__(self):
    self.myvars = {}
  def visitRoot(self,ctx):
    print("Root Visited")
    l = list(ctx.getChildren())
    for a in l:
      print(self.visit(a))
  def visitExpr(self,ctx):
    print("Expr found")
    l = list(ctx.getChildren())
    if len(l) == 1: # when NUM
      return int(l[0].getText())
    else:
      return (ops[l[1].getText()](self.visit(l[0]),self.visit(l[2])))
  def visitAction(self,ctx):
    l = list(ctx.getChildren())
    if len(l) == 3: # assignment =>  Name := expr
      if (l[1].getText() == ':='):
        self.myvars[l[0].getText()] = self.visit(l[2])
        return 'assignment to ' + l[0].getText()
      else:
        return 'ERROR'
    else: # when len(l) == 2, PRINT
      if (l[0].getText() == 'write'):
        return str(self.myvars[l[1].getText()])
      else:
        return 'ERROR'
```

19. Initial Interpreter - Python script
- A file passed as a paramter: 
  - `input_stream = FileStream(sys.argv[1])`
- script.py
```py
from antlr4     import *
from ExprLexer  import ExprLexer
from ExprParser import ExprParser
from VarVisitor import VarVisitor
import sys
input_stream = FileStream(sys.argv[1])
lexer = ExprLexer(input_stream)
token_stream = CommonTokenStream(lexer)
parser = ExprParser(token_stream)
tree = parser.root()
visitor = VarVisitor()
visitor.visit(tree)
```
- input.txt
```
a := 3 + 8
write a
b := 4*7
write b
```
- Testing:
```bash
$ python3 script.py input.txt 
Root Visited
Expr found
Expr found
Expr found
assignment to a
11
Expr found
Expr found
Expr found
assignment to b
28
None
```

## Section 7: Interpreters with Conditional Recognition

20. Interpreter 1 - Grammar
- Define the grammar of an interpreter to evaluate the conditional structure if-else with the less strict (<) and greater strict (>) operators
- Also the interpreter must be able to print
```
if 2 < 3
print 8
else print 5
```
- Condition.g
```g4
grammar Condition;
root: action+ EOF;
action: 'if' expr action ('else' action)? # Condition
     | 'print' expr                       # Print
     ;
expr: expr GT expr  # Gt
    | expr LT expr  # Lt
    | NUM           # Value
    ;
GT: '>' ;
LT: '<' ;
NUM: [0-9]+ ;
WS: [ \t\r\n]+ -> skip;
```

21. Interpreter 1 - Visitor
- `antlr4 -Dlanguage=Python3 -no-listener -visitor Condition.g`
- Visitor.py
```py
if __name__ is not None and "." in __name__:
  from .ConditionParser import ConditionParser
  from .ConditionVisitor import ConditionVisitor
else:
  from ConditionParser import ConditionParser
  from ConditionVisitor import ConditionVisitor
class Visitor(ConditionVisitor):
  def visitRoot(self,ctx):
    print("Root Visited")
    l = list(ctx.getChildren())
    for a in l:
      self.visit(a)
  def visitCondition(self,ctx):
    l = list(ctx.getChildren())
    if self.visit(l[1]) == 1: # if true
      self.visit(ctx.action(0))
    elif len(l) > 3: # having else condition
      if ctx.getChild(3).getText() == 'else':
        self.visit(ctx.action(1))
  def visitPrint(self,ctx):
    l = list(ctx.getChildren())
    print(self.visit(l[1])) # Need to visit expr
  def visitGt(self,ctx):
    l = list(ctx.getChildren())
    return int(self.visit(l[0]) > self.visit(l[2]))
  def visitLt(self,ctx):
    l = list(ctx.getChildren())
    return int(self.visit(l[0]) < self.visit(l[2]))
  def visitValue(self,ctx):
    print("Value = ", ctx.NUM().getText())
    return int(ctx.NUM().getText())
```
- script.py
```py
import sys
from antlr4     import *
from ConditionLexer  import ConditionLexer
from ConditionParser import ConditionParser
from Visitor         import Visitor
input_stream = FileStream(sys.argv[1])
lexer = ConditionLexer(input_stream)
token_stream = CommonTokenStream(lexer)
parser = ConditionParser(token_stream)
tree = parser.root()
visitor = Visitor()
visitor.visit(tree)
```
- input.txt
```
if 2 < 3
print 8
else print 5
if 7 > 9
print 7
else print 9
```
- Testing:
```bash
$ python3 script.py input.txt 
Root Visited
Value =  2
Value =  3
Value =  8
8
Value =  7
Value =  9
Value =  9
9
```

22. Interpreter 2 - Grammar
- Define the grammar of an interpreter to evalulate the conditional structure if-else with the operators less strict (<), greater strict (>), (==) equality and non equality (!=)
- There are two existing instructions: write and next
```
iff 2==3
write 8
otherwise
write 5
```
- write prints a given expression
- next prints the next number after evaluating the expression

23. Interpreter 2 - Visitor

## Section 8: Interpreters with While

24. Loop Recognition - Grammar

25. Loop Recognition - Visitor

## Section 9: Final Programming Language

26. Introduction to the Project

27. Bazilio Program Example

28. Presentation of the Bazilio Programming Language

29. Specification of Bazilio Programming Language
