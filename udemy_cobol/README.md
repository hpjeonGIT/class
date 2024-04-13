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
```cobol
000100 IDENTIFICATION DIVISION.
000200  PROGRAM-ID. DEMOCL04.
000300  AUTHOR. TOPICTRICK.
000400  DATE-WRITTEN.  01-JAN-2023.
000500  DATE-COMPILED. 01-JAN-2023.
000600*
000700* -----------------------------------------------------------*
000800*    PROJ: Demo Class 4. 
000900*        DESC: This program demonstrate the following COBOL    
001000*              statements.
001100*              1. COBOL Program Divisions. 
001200*              2. COBOL Variables and Group Variables. 
001300*              3. COBOL Initilialize Statement. 
001400*              4. COBOL level 88 condition clause. 
001500*              5. COBOL Redefine and Rename clause. 
001600*              6. COBOL Accept and Display clause. 
001700*
001800*        Files: 
001900*              Input  - Employee Detail File (Sequential). 
002000*              Output - Monthly Report       (Sequential).
002100*
002200*        Note: This program does not have any exception handling
002300*              error routine or file-status condition codes. 
002400* -----------------------------------------------------------*            
002500 ENVIRONMENT DIVISION.
002600  INPUT-OUTPUT SECTION.
002700  FILE-CONTROL.
002800     SELECT EMP-FILE ASSIGN TO  EMPCUR
002900            ORGANIZATION IS SEQUENTIAL.
003000*
003100     SELECT REP-FILE ASSIGN TO EMPRPT
003200            ORGANIZATION IS SEQUENTIAL.
003300*
003400 DATA DIVISION.
003500  FILE SECTION.
003600  FD EMP-FILE.
003700  COPY EMPREC REPLACING ==(EP)==  BY ==IN==.
003800*
003900  FD REP-FILE.
004000  01 REP-FILE-REC               PIC X(150).
004100*
004200 WORKING-STORAGE SECTION.
004300*
004400  01 WS-SWITCH.
004500     05 END-OF-FILE-SWITCH      PIC X(01)  VALUE 'N'.
004600        88 END-OF-FILE                     VALUE 'Y'.
004700        88 NOT-END-OF-FILE                 VALUE 'N'.
004800*
004900  01 WS-COUNTERS.
005000     05 WS-INP-REC              PIC 9(05).
005100     05 WS-OUT-REC              PIC 9(05).
005200*
005300  01 WS-TEMP-DATE.
005400     05 WS-HDTE                 PIC 9(08).
005500     05 THIS REDEFINES WS-HDTE.
005600        10 WS-HDTE-DD           PIC X(02).
005700        10 WS-HDTE-MM           PIC X(02).
005800        10 WS-HDTE-YYYY         PIC X(04).
005900     05 WS-TDY-DTE              PIC 9(08).
006000     05 THIS REDEFINES WS-TDY-DTE.
006100        10 WS-TDYDTE-YYYY       PIC 9(04).
006200        10 WS-TDYDTE-MM         PIC 9(02).
006300        10 WS-TDYDTE-DD         PIC 9(02).
006400
006500*
006600  01 HEAD1.
006700     05 FILLER            PIC X(60) VALUE SPACES.
006800     05 FILLER            PIC X(10) VALUE ' EMPLOYEE '.
006900     05 FILLER            PIC X(25) VALUE 'MANAGEMENT SYSTEM.'.
007000     05 FILLER            PIC X(41) VALUE SPACES.
007100     05 HD-DTE.
007200        10 HD-DTE-DD      PIC X(02) VALUE SPACES.
007300        10 FILLER         PIC X(01) VALUE '/'.
007400        10 HD-DTE-MM      PIC X(02) VALUE SPACES.
007500        10 FILLER         PIC X(01) VALUE '/'.
007600        10 HD-DTE-YYYY    PIC X(04) VALUE SPACES.
007700        10 HD-DOT         PIC X(01) VALUE '.'.
007800        10 HD-FILLER      PIC X(03) VALUE SPACES.
007900     66 WS-PGM-SUMY-DTE RENAMES HD-DTE-DD THRU HD-FILLER.
008000*
008100  01 HEAD2.
008200     05 FILLER            PIC X(70) VALUE SPACES.
008300     05 FILLER            PIC X(11) VALUE 'DUBLIN, IRL'.
008400     05 FILLER            PIC X(69) VALUE SPACES.
008500*
008600  01 COLHEAD3.
008700     05 FILLER            PIC X(04) VALUE SPACES.
008800     05 FILLER            PIC X(05) VALUE 'EMPNO'.
008900     05 FILLER            PIC X(02) VALUE SPACES.
009000     05 FILLER            PIC X(06) VALUE 'F NAME'.
009100     05 FILLER            PIC X(08) VALUE SPACES.
009200     05 FILLER            PIC X(07) VALUE 'M INIIT'.
009300     05 FILLER            PIC X(01) VALUE SPACES.
009400     05 FILLER            PIC X(06) VALUE 'L NAME'.
009500     05 FILLER            PIC X(11) VALUE SPACES.
009600     05 FILLER            PIC X(06) VALUE 'W DEPT'.
009700     05 FILLER            PIC X(03) VALUE SPACES.
009800     05 FILLER            PIC X(05) VALUE 'PH NO'.
009900     05 FILLER            PIC X(03) VALUE SPACES.
010000     05 FILLER            PIC X(06) VALUE 'H DATE'.
010100     05 FILLER            PIC X(05) VALUE SPACES.
010200     05 FILLER            PIC X(06) VALUE 'JOB TY'.
010300     05 FILLER            PIC X(04) VALUE SPACES.
010400     05 FILLER            PIC X(08) VALUE 'ED LEVEL'.
010500     05 FILLER            PIC X(02) VALUE SPACES.
010600     05 FILLER            PIC X(03) VALUE 'SEX'.
010700     05 FILLER            PIC X(02) VALUE SPACES.
010800     05 FILLER            PIC X(10) VALUE 'BIRTH DATE'.
010900     05 FILLER            PIC X(05) VALUE SPACES.
011000     05 FILLER            PIC X(06) VALUE 'SALARY'.
011100     05 FILLER            PIC X(05) VALUE SPACES.
011200     05 FILLER            PIC X(05) VALUE 'BONUS'.
011300     05 FILLER            PIC X(05) VALUE SPACES.
011400     05 FILLER            PIC X(04) VALUE 'COMM'.
011500     05 FILLER            PIC X(07) VALUE SPACES.
011600
011700* Report file detail line header.
011800
011900  01 DTL-LINE.
012000     05 FILLER               PIC X(03) VALUE SPACES.
012100     05 DTL-EMPNO            PIC Z(06).
012200     05 FILLER               PIC X(02) VALUE SPACES.
012300     05 DTL-FIRSTNME         PIC X(12).
012400     05 FILLER               PIC X(04) VALUE SPACES.
012500     05 DTL-MIDINIT          PIC X(01).
012600     05 FILLER               PIC X(05) VALUE SPACES.
012700     05 DTL-LASTNAME         PIC X(17).
012800     05 FILLER               PIC X(02) VALUE SPACES.
012900     05 DTL-WORKDEPT         PIC X(03).
013000     05 FILLER               PIC X(05) VALUE SPACES.
013100     05 DTL-PHONENO          PIC Z(04).
013200     05 FILLER               PIC X(02) VALUE SPACES.
013300     05 DTL-HIREDATE.
013400        10 DTL-HIREDATE-DD   PIC 9(02).
013500        10 FILLER            PIC X(01) VALUE '/'.
013600        10 DTL-HIREDATE-MM   PIC 9(02).
013700        10 FILLER            PIC X(01) VALUE '/'.
013800        10 DTL-HIREDATE-YYYY PIC 9(04).
013900     05 FILLER               PIC X(03) VALUE SPACES.
014000     05 DTL-JOB              PIC X(08).
014100     05 FILLER               PIC X(04) VALUE SPACES.
014200     05 DTL-EDLEVEL          PIC Z(02).
014300     05 FILLER               PIC X(06) VALUE SPACES.
014400     05 DTL-SEX              PIC X(01).
014500     05 FILLER               PIC X(03) VALUE SPACES.
014600     05 DTL-BIRTHDATE.
014700        10 DTL-BRTHDATE-DD   PIC 9(02).
014800        10 FILLER            PIC X(01) VALUE '/'.
014900        10 DTL-BRTHDATE-MM   PIC 9(02).
015000        10 FILLER            PIC X(01) VALUE '/'.
015100        10 DTL-BRTHDATE-YYYY PIC 9(04).
015200     05 FILLER               PIC X(03) VALUE SPACES.
015300     05 DTL-SALARY           PIC ZZZZZZ9.99.
015400     05 FILLER               PIC X(03) VALUE SPACES.
015500     05 DTL-BONUS            PIC ZZZ9.99.
015600     05 FILLER               PIC X(03) VALUE SPACES.
015700     05 DTL-COMM             PIC ZZZ9.
015800     05 FILLER               PIC X(07) VALUE SPACES.
015900*
016000 01 TRL-LINE.
016100     05 FILLER               PIC X(04)  VALUE SPACES.
016200     05 TRL-LINE-MSG         PIC X(27)  VALUE SPACES.
016300     05 TRL-COUNT            PIC 9(05)  VALUE ZEROES.
016400     05 FILLER               PIC X(114) VALUE SPACES.
016500*
016600 01 RPT-BLK-LNE.
016700    05 RPT-BLK-AST           PIC X(01)  VALUE '*'.
016800    05 RPT-BLK-SPC           PIC X(149) VALUE SPACES.
016900*
017000 PROCEDURE DIVISION.
017100 0000-CORE-BUSINESS-LOGIC.
017200     PERFORM A000-INIT-VALS
017300     PERFORM B000-OPEN-FILE
017400     PERFORM C000-PRNT-HDRS
017500     PERFORM D000-PROC-RECD
017600     PERFORM X000-CLSE-FILE
017700     STOP RUN.
017800*
017900 A000-INIT-VALS SECTION.
018000 A010-INIT-TMP-VALS.
018100     INITIALIZE WS-COUNTERS, DTL-LINE, TRL-LINE,
018200                WS-TEMP-DATE.
018300*
018400 A099-EXIT.
018500      EXIT.
018600*
018700 B000-OPEN-FILE SECTION.
018800 B010-OPEN-FILE.
018900      OPEN INPUT  EMP-FILE
019000           OUTPUT REP-FILE.
019100 B099-EXIT.
019200      EXIT.
019300*
019400 C000-PRNT-HDRS SECTION.
019500 C010-MOVE-TDY-DATE.
019600      ACCEPT WS-TDY-DTE    FROM DATE YYYYMMDD
019700      MOVE WS-TDYDTE-DD    TO HD-DTE-DD
019800      MOVE WS-TDYDTE-MM    TO HD-DTE-MM
019900      MOVE WS-TDYDTE-YYYY  TO HD-DTE-YYYY.
020000
020100 C020-PRNT-RPT-HDRS.
020200      WRITE REP-FILE-REC FROM HEAD1.
020300      WRITE REP-FILE-REC FROM HEAD2.
020400      WRITE REP-FILE-REC FROM RPT-BLK-LNE.
020500      WRITE REP-FILE-REC FROM COLHEAD3.
020600      WRITE REP-FILE-REC FROM RPT-BLK-LNE.
020700*
020800 C099-EXIT.
020900      EXIT.
021000*
021100 D000-PROC-RECD SECTION.
021200 D010-READ-FILE-REC.
021300     PERFORM UNTIL END-OF-FILE
021400         READ EMP-FILE
021500              AT END     SET END-OF-FILE TO TRUE
021600              NOT AT END PERFORM E000-PRNT-REPT
021700         END-READ
021800     END-PERFORM.
021900*
022000 D099-EXIT.
022100      EXIT.
022200*
022300 E000-PRNT-REPT SECTION.
022400 E010-MOVE-DTL-REC.
022500     MOVE  IN-EMPNO        TO DTL-EMPNO
022600     MOVE  IN-FIRSTNME     TO DTL-FIRSTNME
022700     MOVE  IN-MIDINIT      TO DTL-MIDINIT
022800     MOVE  IN-LASTNAME     TO DTL-LASTNAME
022900     MOVE  IN-WORKDEPT     TO DTL-WORKDEPT
023000     MOVE  IN-PHONENO      TO DTL-PHONENO
023100     MOVE  ZEROES          TO WS-HDTE
023200     MOVE  IN-HIREDATE     TO WS-HDTE
023300     MOVE  WS-HDTE-DD      TO DTL-HIREDATE-DD
023400     MOVE  WS-HDTE-MM      TO DTL-HIREDATE-MM
023500     MOVE  WS-HDTE-YYYY    TO DTL-HIREDATE-YYYY
023600     MOVE  IN-JOBTY        TO DTL-JOB
023700     MOVE  IN-EDLEVEL      TO DTL-EDLEVEL
023800     MOVE  IN-SEX          TO DTL-SEX
023900     MOVE  ZEROES          TO WS-HDTE
024000     MOVE  IN-BIRTHDATE    TO WS-HDTE
024100     MOVE  WS-HDTE-DD      TO DTL-BRTHDATE-DD
024200     MOVE  WS-HDTE-MM      TO DTL-BRTHDATE-MM
024300     MOVE  WS-HDTE-YYYY    TO DTL-BRTHDATE-YYYY
024400     MOVE  IN-SALARY       TO DTL-SALARY
024500     MOVE  IN-BONUS        TO DTL-BONUS
024600     MOVE  IN-COMM         TO DTL-COMM
024700     ADD +1                TO WS-INP-REC.
024800*
024900 E020-WRITE-DTL-REC.
025000     WRITE REP-FILE-REC FROM DTL-LINE.
025100     INITIALIZE DTL-LINE
025200     ADD +1   TO WS-OUT-REC.
025300*
025400 E099-EXIT.
025500      EXIT.
025600*
025700 X000-CLSE-FILE SECTION.
025800 X010-PRNT-TRL-REC.
025900     WRITE REP-FILE-REC FROM RPT-BLK-LNE
026000*
026100     MOVE 'TOTAL NO OF RECORD READ  :' TO TRL-LINE-MSG
026200     MOVE  WS-INP-REC    TO TRL-COUNT
026300     WRITE REP-FILE-REC FROM TRL-LINE
026400*
026500     MOVE  SPACES        TO TRL-LINE-MSG
026600     MOVE  ZEROES        TO TRL-COUNT
026700*
026800     MOVE 'TOTAL NO OF RECORD PRINT :' TO TRL-LINE-MSG
026900     MOVE  WS-OUT-REC    TO TRL-COUNT
027000*
027100     WRITE REP-FILE-REC FROM TRL-LINE.
027200
027300 X020-CLOSE-FILE.
027400     CLOSE EMP-FILE, REP-FILE.
027500*
027600 X020-PRINT-TOTALS.
027700     DISPLAY '****** PROGRAM SUMMARY ****************'
027800     DISPLAY 'PGM EXECUTION DATE       :', WS-PGM-SUMY-DTE
027900     DISPLAY 'TOTAL NO OF RECORD READ  :', WS-INP-REC
028000     DISPLAY 'TOTAL NO OF RECORD PRINT :', WS-OUT-REC
028100     DISPLAY '****************************************'.
028200*
028300 X099-EXIT.
028400      EXIT.
```

## Section 5: Understanding Loop Constructs in COBOL

### 35. Logical Control Structures in COBOL
- Four logical statements
  - SEQUENCE: step by step process
  - SELECTION: IF-THEN-ELSE
  - ITERATION: loop using UNTIL
  - CASE STRUCTURE: case switch using WHEN

### 36. COBOL Perform Statement
- COBOL Logical statements
- Inline Perform
```cobol
PERFROM UNTIL NO-MORE-REC
        READ EMP-MAST
             AT END
        ....
        END-READ
END-PERFORM
```
- Out-of-line Perform
```cobol
A000-MAIN-PARA
  PERFORM B010-CAL-TOT-SAL
  DISPLAY 'EMP SAL': WS-SAL.
B010-CAL-TOT-SAL.
  COMPUTE WS-SAL = WS-MNTH-SAL + WS-MNTH-BONUS
...
```
- PERFORM Statement formats
  - Paragraph name
  - Times phase: repeats as many as requested
  - Varying phrase: repeats as the condition is met
  - Until phrase: repeats until the condition is met
- PERFORM is used for looping while PERFORM THRU is used for branching

### 37. COBOL IF THEN ELSE END-IF Statements
```cobol
IF TRN-TYPE = 'DR' THEN
  ADD +1 TO WS-DR-CNT
  DISPLAY 'TOTAL DEBIT TRAN CNT', WS-DR-CNT
ELSE
  ADD +1 TO WS-CR-CNT
  DISPLAY 'TOTAL DEBIT TRAN CNT', WS-CR-CNT
END-IF.
```
- Not more than 3-nested IF statements are not recommended

### 38. COBOL Evaluate Statements
- Introduced with COBOL-85
- Replaces nested IF's
```cobol
EVALUATE identifier/expression
  WHEN condition-1
    imperative statement 1
  WHEN condition-2
    imperative statement 2
  WHEN OTHER
    imperatve statement 3
END-EVALUATE
```
- With TRUE
- With ALSO
  - WHEN clause will have ALSO as well, providing 2ndary conditions
- With conditions
- With THRU

### 39. COBOL GOTO Statement
- Unconditional: transfers control to the first statement in the specified paragraph/selection
- Conditional: transfers control to one of a series of procedures, depending on the value of the identifier
```cobol
GO TO procedure name
GO TO procedure name-1 DEPENDING ON identifier
```

### 40. COBOL CONTINUE vs NEXT SENTENCE 
- NEXT SENTENCE: equivalent to GO TO
  - Will ignore END-IF
  - Control will passes to the statement after the closest following **period (.)**
  - Not used in modern programming often
- CONTINUE: no effect on execution
  - For better readability

### 41. 2. Demo Class: Mastering Control Structures in COBOL

## Section 6: String Manipulation Operations in COBOL

### 42. COBOL Text Manipulation
- STRING: concatenates two or more sending fields into one receiving field
- UNSTRING: decouple a field
- INSPECT: counts characters
- REFERENCE: Refers to a specific location within a field

### 43. COBOL String Statement
- STRING literal-1 DELIMITED BY literal-2 SIZE ... INTO literal-3 END-STRING
- Options
  - WITH POINTER
  - ON OVERFLOW: 
```cobol
STRING FIRST-NAME DELIMITED BY ' '
  ' ' DELIMITED BY SIZE
  LAST-NAME DELIMITED BY ' '
    INTO EMPLOYEE-NAME
END-STRING.
```
- `' ' DELIMITED BY SIZE` will add a space b/w first-name and last name when concatenated

### 44. COBOL Unstring Statement
```cobol
UNSTRING EMPLOYEE-NAME
  DELIMITED BY SIZE ','
  INTO FIRST-NAME LAST-NAME
END-STRING.  
```

### 45. COBOL Inspect Statement
- Replaces a specific character in a field with another character
- May be used to validate formats
- Statement has different formats by use of Tallying, Replacing, and Converting clauses
```cobol
PROCEDURE DIVISION.
A000-MAIN-SECTION.
  INSPECT WS-DTE REPLACING ALL "-" BY "/".
  INSPECT WS-NME TALLYING WS-CNT FOR ALL SPACES.
  INPSECT WS-NME REPLACING ALL "D" BY "D" AFTER INITIAL "D".
  INPSECT WS-NME CONVERTING "ABCDEFGHIJKLMNOPQRSTUVWXYZ" TO "abcdefghijklmnopqrstuvwxyz".
```

### 46. COBOL Reference Modification Statement
- Allows to refer to a portion of a field
```cobol
MOVE EMPNAME(1:5) TO FNAME.
```

### 47. 3. Demo Class: Mastering COBOL String Manipulation

## Section 7: Advanced COBOL Programming Techniques

### 48. Call Statement in COBOL
- Main-program
- Sub-program
```cobol
CALL "EMPTAX01" 
   USING EMP-SAL, EMP-TAX.
```
- CALL ... BY REFERENCE: main and sub can access the address of the data
- CALL ... BY CONTENT: can change the data from the main but will not be stored in main (?)
- CALL ... BY VALUE: no address of data
- Static vs dynamic call
- Do not use `STOP` in the sub program
- Sub program CANNOT CALL main program

### 49. Copy Statement in COBOL
- Copy the portion of COBOL code
- Runs at compilation time
- If you change the code in copy book, recompilation is required

### 50. Intrinsic Functions in COBOL
- Calendar function
  - CURRENT-DATE
  - WHEN-COMPILED
  - INTEGER-OF-DATE
  - DAY-OF-INTEGER
  - INTEGER-OF-DAY
  - DATE-OF-INTEGER
- Statistical and numerical analysis
  - INTEGER
  - MAX/MIN
  - MEAN/NUMVAL
  - MEDIAN
  - SUM/RANDOM
  - RANGE/SQRT
- Trigonomertic and financial functions
  - SIN/COS/TAN
  - ASIN/ACOS/ATAN
  - ANNUITY
  - PRESENT-VALUE
- Character and string functions
  - LOWER-CASE
  - UPPER-CASE
  - LENGTH
  - ORD
  - CHAR
  - REVERSE

### 51. 4. Demo class: Master COBOL Call statements

### 52. 5. Demo class: Mastering intrinsic functions in COBOL

## Section 8: COBOL Table Handling: Mastering Data Arrays and Indexing

### 53. Introduction to COBOL Tables
- An Array or Table in COBOL is a linear data structure that is used to store, access, and process data
- Elements can be used in arithmetic and logical operations
- Elements are stored contiguously
- An array/table can have dimension up to 7 but mostly 1,2,3D arrays are used

### 54. Occurs Clause in COBOL
- OCCURS clause is used to define an array/table
```cobol
WORKING-STORAGE SECTION.
01 WS-HR-TEMP-REC.
  05 WS-HR-TEMP OCCURS 24 TIMES PIC S9(3).
```  
  - Only level numbers 02-49 are allowed for OCCURS clause
- Array with initialized value
```cobol
WORKING-STORAGE SECTION.
01 WS-YR-MNTHS VALUE "JANFEBMARAPRJUNJULAUGSEPOCTNOVDEC".
  05 WS-MNTH OCCURS 12 TIMES PIC X(3).
```
- 2d array sample
```cobol
WORKING-STORAGE SECTION.
01 WS-TEMP-REC.
  05 WS-DAYS OCCURS 07 TIMES.
    10 WS-HOURS OCCURS 24 TIMES.
      15 WS-TEMP  PIC S9(3).
```
- Group items
```cobol
WORKING-STORAGE SECTION.
01 WS-BOOKS.
  05 WS-BK-DTLS OCCURS 100 TIMES.
    10 WS-AUTHOR.
      15 WS-FIRST-NAME PIC A(15).
      15 WS-MID-NAME   PIC A(10).
      15 WS-LAST-NAME  PIC A(10).
    10 WS-TITLE        PIC A(60).
```

### 55. Variable Length Table in COBOL
- Not dynamically adjusted table
- Min/max of entries that it can contain

### 56. SET Statement in COBOL
- Sets an index, an index data item or an integer field
```cobol
PROCEDURE DIVISION.
A001-ACCESS-DATA
...
SET X1 TO 1
...
SET X1 UP BY 1
...
SET X1 DOWN BY 1
```

### 57. COBOL Refering to an item in a table Subscript/index
- The subscript/index must be enclosed with a pair of parenthesis
- Subscripts are easy to use but not as efficient as indexes
- Access data using subscript
  - A subscript is a field containing an occurrence number
  - Must be an integer
```cobol
WORKING-STORAGE SECTION.
01 WS-HR-TEMP-REC.
  05 WS-HR-TEMP OCCURS 10 TIMES PIC S9(03)
  01 TBL-SUB PIC S9(02) COMP.
PROCEDURE DIVISION.
A001-ACCESS-DATA.
...
DISPLAY "1. TEMPERATURE:", WS-HR-TEMP(01).
MOVE 02 TO TBL-SUB
DISPLAY "2. TEMPERATURE:", WS-HR-TEMP(TBL-SUB).
DISPLAY "3. TEMPERATURE:", WS-HR-TEMP(TBL-SUB+1).
```
- Access data using an index
  - INDEXED BY clause
```cobol
WORKING-STORAGE SECTION.
01 WS-HR-TEMP-REC.
  05 HW-HR-TEMP OCCURS 10 TIMES PC S9(03)
                INDEXED BY TX1.
PROCEDURE DIVISION
A001-ACCESS-DATA.
....
SET TX1 TO 1.
DISPLAY "1. TEMPERATURE:", WS-HR-TEMP(TX1).
SET TXT1 UP BY 1.
DISPLAY "2. TEMPERATURE:", WS-HR-TEMP(TX1).
DISPLAY "3. TEMPERATURE:", WS-HR-TEMP(03).
```

### 58. What is the difference b/w subscript and index
- Subscript
  - Defined in a separate WORKING-STORAGE entry
  - Use PERFORM ... VARYING or MOVE/ADD/SUBTRACT
- Index
  - Defined along with the OCCURS for the array or table
  - Use PERFORM ... VARYING or SET. Cannot use MOVE/ADD/SUBTRACT

### 59. COBOL Table - How to load data into a table?
- INITIALIZE statement
- VALUE clause
```cobol
WORKING-STORAGE SECTION.
01 WS-YR-MNTHS VALUE "JANFEBMARAPRJUNJULAUGSEPOCTNOVDEC".
  05 WS-MNTH OCCURS 12 TIMES PIC X(3).
```  
- Load table dynamically
  - Use the PERFORM statement and either subscripting or indexing
```cobol
WORKING-STORAGE-SECTION.
01 PRICE-TABLE.
  05 PRICE-GROUP OCCURS 16 TIMES
                 INDEXED BY IX1.
  10 ITEM-NUMBER PIC 9(3).
  10 ITEM-PRICE  PIC S99V99.
01 WS-SWITCH     PIC X(01)  VALUE 'N'
  88 END-OF-FILE  VALUE 'Y'
PROCEDURE DIVISION.
A001-INIT-TBL.
PERFORM VARYING IX1 FROM 1 BY 1 UNTIL IX1 >=16
  MOVE ZEROES TO ITEM-NUMBER(IX1)
  MOVE ZEROES TO ITEM-PRICE (IX1)
  PERFORM A002-READ-PROD-FLE
  IF NOT EOF-OF-FILE THEN
    MOVE IT-NUMBER TO ITEM-NUMBER(IX1)
    MOVE IT-PRICE  TO ITEM-PRICE (IX1)
  END-IF
END-PERFORM.
```

### 60. Search Statement in COBOL
- Sequential/serial search
  - Searches the table entries starting with the current index value
  - Cannot find multiple matches. Only the first matching element
```cobol
PROCEDURE DIVISION.
A001-SEARCH-TBL.
SET IX1 TO 1.
SEARCH PRICE-GROUP
  AT END
    MOVE "N" TO ITEM-FOUND-SWITCH
  WHEN ITEM-NUMBER(IX1) = TR-ITEM-NO
    MOVE "Y" TO ITEM-FOUND-SWITCH
END-SEARCH.
```

### 61. Search All Statement in COBOL
- Binary search
  - Table must be sorted beforehand
```cobol
WORKING-STORAGE SECTION.
01 PRICE-TABLE.
05 PRICE-GROUP OCCURS 16 TIMES
    ASCENDING KEY IS ITEM-NUMBER
    INDEXED BY IX1.
  10 ITEM-NUMBER PIC 9(03).
  10 ITEM-PRICE  PIC S9(02)V9(02).
...
PROCEDURE DIVISION.
A001-SEARCH-TBL.
  SEARCH ALL PRICE-GROUP
    AT END
      MOVE "N" TO ITEM-FOUND-SWITCH
    WHEN ITEM-NUMBER(IX1) = TR-ITEM-NO
      MOVE "Y" TO ITEM-FOUND-SWITCH
  END-SEARCH.
```

### 62. SEARCH vs SEARCH ALL - difference in COBOL
- SEARCH - serial search
  - The table doesn't need sorting
  - Requires SET statement prior to SEARCH
  - May use multiple WHENs
- SEARCH ALL - binary search
  - Table must be sorted. Ascending or descending.
  - Cannot use multiple WHENs

### 63. 6. Demo Class: How to define and use the one-dimensional table?

### 64. 7. Demo Class: How to define and use the two-dimensional table in COBOL program?

### 65. 8. Demo Class: How to define and use variable length table

## Section 9: COBOL Data Fields: Definition, Movement, and Usage Clause

### 66. Move Statement in COBOL
- Transfers data from one area of storage to one or more other storage areas
- If the receiving field is larger than sending field, zeroes (numeric) or spaces (alphabetic) are filled
- If the receiving filed is smaller than sending field, then it will be truncated
```cobol
MOVE EMP-NO TO WS-EMP-NO.
MOVE 1233   TO WS-EMP-SAL.
MOVE 'COMP' TO WS-EMP-DEPT.
```

### 67. Numeric Move Rules and Example
- When Liter moves and sending/receiving fields have the same PIC
  - Right to left
- When receiving field is larger than sending
  - 123 -> 0123
  - or 123 -> 1230
- The lecture is not clear. Asked in Q&A section

### 68. Nonnumeric or Alphanumeric Move Rules and Example
- Filling as from left to right 
- When receiving field is larger, then space is added

### 69. COBOL USAGE and Data Format
- How to store data in the form of binary digits
  - Default is DISPLAY and may not perform well
  - `A PIC 9(01) VALUE 4.` stores ASCII 4, which is 00110100
- USAGE IS
  - BINARY
  - COMPUTATIONAL
  - COMP
  - INDEX
  - PACKED-DECIMAL
  - DISPLAY

### 70. COBOL Usage is Display
- Despite being computationaly inefficient, some benefits are:
  - Data can be directly output to a screen
  - Improved portability

### 71. COBOL Usage is Packed-decimal (COMP-3)
- Instead of using a single binary number to represent a value, each digit is tored in a nibble (a half byte) using binary values
- Stores two digits in each storage position
- Not readable and shouldn't be used for printing

### 72. COBOL Usage is Computational (COMP)

### 73. COBOL Comp Sync Clause
- SYNCHRONIZED clause is used with USAGE IS COMP or USAGE IS INDEX
- Optimizes the speed of processing at the expense of increased storage requirements
- Explictily aligns COMP and INDEX items along their natural word boundaries. Without SYNCHRONIZED, data items are algined on byte boundaries
- Slack bytes: bytes inserted b/w data items or records to ensure correct alignment

### 74. COBOL Blank When Zero Clause
- Editing
  - Insertion: 100 => $100
  - Suppression or replacement of Zeroes: 0.99 => Z.99

### 75. Defining Numeric Data - Sign Clause
- S in the PICTURE Clause
- Numeric data can be stored in 5 different formats in IBM mainframe
  - Zone-decimal (Display), packed-decimal(COMP-3), and binary (COMP) are common usage

### 76. 9. Demo Class: Mastering COBOL Data Fields: Definition, Movement, and Usage

## Section 10: COBOL Arithmetic Operation/Calculation

### 77. COBOL Arithmetic Operation Introduction
- Binary: +, -, *, /, **
- Unary: +, -

### 78. Calculate the size of receiving field
- How to avoid truncation
- Resultant field must be one position larger than the largest field in addition
- Resultant field must be as large as the minuend in subtraction
- Resultant field must be equal to the sum of the lengths of the operands in multiplication
- Resultant field must be equal to the sum of the number of digits in the divisor and dividend

### 79. Addition Statement in COBOL
```cobol
ADD WS-BONUS TO WS-MNTH-SALARY.
ADD WS-BONUS TO WS-MNTH-SALARY, TOT-YR-BONUS.
ADD +100 TO WS-MNTH-SALARY,TOT-YR-BONUS.
ADD WS-FRQ-BONUS, WS-SEC-BONUS GIVING TOT-YR-BONUS.
ADD +100, WS-FREQ-BONOUS, WS-SEC-BONUS GIVING TOT-YR-BONUS.
```

### 80. Subtract Statement in COBOL
```cobol
SUB WS-BONUS FROM WS-MNTH-SALARY.
SUB WS-BONUS FROM WS-MNTH-SALARY, TOT-YR-BONUS.
SUB WS-FRQ-BONUS, WS-ARR-AMT FROM TOT-MTN-SAL GIVING TOT-CR-AMT.
SUB CORRESPONDING 100 FROM WS-MNTH-SALARY, TOT-YR-BONUS.
```

### 81. Divide Statement in COBOL
```COBOL
DIVIDE 2 INTO WS-NUM-1.
DIVIDE 60 INTO SECONDS GIVING MINUTES.
DIVIDE HOUR BY 60 GIVING MINUTES.
DIVIDE NUM-1 INTO NUM-2 GIVING NUM-3 REMAINDER NUM-4.
```

### 82. Multiply Statement in COBOL
```COBOL
MULTIPLY 1.5 BY WS-BONUS.
MULTIPLY WS-RATE BY WS-MNTH-EMI.
MULTIPLY 60 BY WS-HOURS GIVING MINUTES.
MULTIPLY 1.8 BY WS-CEL-TEMP GTIVING TEMP.
```

### 83. Compute Statement in COBOL
```COBOL
COMPUTE EMP-SAL = WS-MNTH-SAL + WS-QTR-BONUS.
COMPUTE DAILY-SALES = QTY * UNIT-PRICE / 5.
COMPUTE DAILY-SALES = QTY * UNIT-PRICE/5 
        ON SIZE ERROR  PERFORM A00-CALL-ABEND-RTN
COMPUTE SALES = WS-TOTAL-SALES
```

### 84. ON SIZE ERROR in COBOL
- When data overflow happens
- Division by zero
```COBOL
ADD AMT1 AMT2 TO AMT3
  GIVING TOTAL-OUT
  ON SIZE ERROR MOVE ZERO TO TOTAL-OUT
END-ADD.
DIVIDE 60 INTO MINUTES
  GIVING HOURS
  ON SIZE ERROR MOVE 'INVALID DIVIDE' TO ERROR-MESSAGE
END-DIVIDE.
COMPUTE DAILY-SALES = QTY * UNIT-PRICE / 5
  ON SIZE ERROR PERFORM A00-CALL-ABEND-RTN
END-COMPUTE.
```

### 85. Rounded Option in COBOL
```COBOL
ADD WS-BONUS,WS-SALARY GIVING TOT-EMP-SAL ROUNDED.
COMPUTE TOT-EMP-SAL ROUNDED = WS-BONUS + WS-SALARY.
DIVIDE UNITS-OF-ITEM INTO WS-TOTAL ROUNDED.
```

### 86. 10. Demo Class: Mastering COBOL Arithmetic Operations

### 87. 11. Demo Class: Mastering Numeric Operations and Exception Handling in COBOL

## Section 11: File Management in COBOL

### 88. COBOL File and Database Handling overview

### 89. FILE Organization and Access Mode in COBOL

### 90. Select Statement in COBOL

### 91. File Description (FD) Entries in COBOL

### 92. Open Statement in COBOL

### 93. Close Statement in COBOL

### 94. Start Statement in COBOL

### 95. Read Statement in COBOL

### 96. How to work with Alternate Index?

### 97. Write Statement in COBOL

### 98. Rewrite Statement in COBOL

### 99. COBOL Delete Statement

### 100. 12. Demo Class - How to process Sequential File

### 101. 13. Demo Class - How to process VSAM File or Indexed Sequential File

## Section 12: File Sorting and Merging in COBOL

## Section 13: Compiling and Debugging COBOL program

## Section 14: Web Services Interface in COBOL

## Section 15: How to design and structure a COBOL Program

## Section 16: Exception Handling in COBOL

## Section 17: Examples: COBOL Integration

## Section 18: COBOL Interview Questions

## Section 19: Project: Employee Report Generation using COBOL Program

## Section 20: Conclusion
