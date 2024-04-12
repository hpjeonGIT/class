## COBOL Complete Reference Course!
- Instructor: Topictrick Education

## Section 1: Welcome.

### 1. Welcome

### 2. Prerequisite for the Course

### 3. How to use the exercise and practice file

## Section 2: Getting Started

### 4. Introduction to IBM mainframe
- IBM Mainframe z16
- Banking, finance, retail, ...

### 5. What is JCL?
- Job Control Language
  - Script language
  - Essential for batch processing
- Microfocus COBOL: free option is available

### 6. How to setup COBOL Environment - Write, Compile, and Execute

## Section 3: Introduction to COBOL

### 7. What is COBOL?
- Common Business-Orientedl Language
- Structured Programming Capability
- Scalable and reliable

### 8. Features of COBOL

### 9. History of COBOL
- Grace Hopper

### 10. Why COBOL dominates enterprise computing?

### 11. Future of COBOL
- 95% of ATM transaction

## Section 4: COBOL Fundamentals

### 12. Program vs OS programs vs application programs

### 13. Characters in COBOL

### 14. Components of COBOL
- Characters
- Reserved Keywords
- User defined words
- Variables, Literals, Structures,
- Optional words
- Constants
- Intrinsic Functions

### 15. Reserved Words in COBOL
- Keywords
  - ADD, DELETE, SEARCH, READ, WRITE, CALL, ...
- Optional words: Included in the format of a clause, an entry, or a statement to improve readability. No effect on execution
  - GIVING, ROUNDOFF, AFTER, ...
- Figurative constants: One value constant
  - ZERO, ZEROES, SPACE, SPACES, NULL, ALL, HIGH-VALUE, LOW-VALUE
  - Might be used as initializers
- Special object identifiers
- Special registers
- Special character words

### 16. COBOL Coding Rules and Guidelines
- 01-80 columns
- 1-6: Sequence number
- 7: comment, continuation
- 8-11: DIVISION, SECTION, paragraph name and items of the DATA division
- 12-72: COBOL entries including PROCEDURE DIVISION
- 73-80: Identification
- Common block 
  - Identification division
  - Environment division
  - Data divsion
  - Procedure division

### 17. Data Types in COBOL
- Alphabetic: DAVID
- Numeric: 30, 1.1
- Alphanmeric: 1000$
- Edited Numeric: $1,00
- Edited Alphanumeric: date

### 18. COBOL Program Structure Overview
- COBOL program division
  - Identification division: for documentation purpose
  - Environment division: file related
  - Data division: input/output format
  - Procedure division: Business logic

### 19. Identification Division in COBOL
- Mandatory but no effect on the execution
  - PROGRAM-ID paragraph
  - AUTHOR
  - INSTALLATION
  - DATA-WRITTEN
  - DATA-COMPILED
  - SECURITY
- Documentation purpose

### 20. Environment Division in COBOL
- Optional division
  - Configuration section: SOURCE-COMPUTER, OBJECT-COMPUTER, ...
  - Input-output section: FILE-CONTROL, I-O CONTROL

### 21. Data Division in COBOL
- Declares the data items
- FILE SECTION: FD, SD entries
- WORKING-STORAGE SECTION: declaration of temporary variables and record structures
- LOCAL-STORAGE SECTION: Storage invocation
- LINKAGE SECTION: field info
- COMMUNICATION SECTION: 
- REPORT SECTION: RD entries
- SCREEN SECTION: formatted input/output

### 22. Procedure Division in COBOL
- Includes statements and sentences for reading input file data, processing it and writing the data to the output file

### 23. Hyphens in Paragraphs, and Variable Names in COBOL
- Paragraph names in COBOL are limited to A-Z, 0-9, and hyphen(-)
  - Hyphen cannot be the first character

### 24. Define a Data-name/Variable in COBOL
- Data name or identifier
- Requirement
  - Level number: specifies the hierarchy of data within a record and identify special-purpose data entries
    - 01: For record description
    - 02-49: For fields with records
    - 01/77: For independent items. Must begin in Area A
    - 66: For RENAMES clause
    - 88: For condition names
  - PICTURE clause: specifies the data type and the amount of storage required
    - A: Alphabetic
    - X: Alphanumeric. Any character. Unused positions to the right are set to spaces
    - 9: Digits. Unused positions to the left are set to zeros
    - S: Sign
    - Z: Zero subpress digit
    - ,: inserted comma
    - .: inserted decimal point
    - -: minus sign if negative    
  - Value clause: specifies the initial value. Can be a numeric literal, non-numeric literal or a figurative constant. Values can be changed in the PROCEDURE DIVISION

### 25. COBOL Group item
- Consists of one or more elementary items. Level number, data name and value clause are used to describe a group item
- The level number must be b/w 01 and 49. Typically 01, 05, 10, 15, ...
- Level 1 items must begin in the A margin. Other level numbers can begin in A or B margin
- No PICTURE clause for a group item
- A group item is treated as an alphanumeric item

### 26. COBOL Initialize Statement
- Applying INITIALIZE statement to the group item, we can reset all of sub items

### 27. COBOL Like statement
- Allows the attributes of a data item to be defined by copying them from a pre-defiend data item

### 28. COBOL Level 88 Clause (SWITCH)
- Condition name must be unique

### 29. COBOL Redefines Statement
- Defines the same field storage into two or more different ways

### 30. COBOL Renames Statement
- Defines an alternative name or alias for data elements or a group data itmes
- Special level number 66

### 31. COBOL Accept Statement
- Reads user input

### 32. COBOL Display Statement
- To sysout device or monitor/terminal

### 33. Let's write a Hello World COBOL Program
- At Ubuntu20, `sudo apt install open-cobol`
- ch33.cbl:
```cobol
IDENTIFICATION DIVISION.
PROGRAM-ID. HELL001.
AUTHOR.     AAA.
DATE-WRITTEN. MAR 31, 2022
*> DESC
PROCEDURE DIVISION.
A001-MAIN.
  DISPLAY "HELLO WORLD"
STOP RUN.  
```
- Demo:
```bash
$ cobc -free -x -o ch33.exe ch33.cbl 
$ ./ch33.exe 
HELLO WORLD
```

### 34. 1. Demo class: Introduction to COBOL Programming Overview and Basics

## Section 5: Understanding Loop Constructs in COBOL

### 35. Logical Control Structures in COBOL
