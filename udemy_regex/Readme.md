## The Complete Regular Expressions Course with Exercises 2020
- Instructor: Jazeb Akram

7. Basic syntax
- enclose in literas with `/` at PCRE2
- test at regex101.com
```
>>> import re
>>> p = re.compile('abcde')
>>> p.match("abcde")
<_sre.SRE_Match object; span=(0, 5), match='abcde'>
```

9. Mode
-  After ending `/`
- standard : `/regx/`
- global : `/regx/g` - finds multiple times
- single line: : `/regx/s`
- case insensitive : `/regx/i`
- multi line : `/regx/m`
- Modes can be aggregated like `/gi`

11. wild card
- `.`: any character except new line
    - Ruby and Javascript
    - PCRE2 may find new line as well
- To match exact `.`, use `\.`

12. character set
- `/[cd]ash/`: cash or dash

13. character range
- `/[a-z]ash/` : aash, bash, ..., yash, zash
- `/[a-zA-Z]ash`: aash, ..., zash, Aash, ..., Zash
- `/[0-9]ash/`: 0ash, ..., 9ash

14. restricing RE
- Negation: `^`
- `/[^cd]ash[^12]/`: will exclude cash1, cash3, dash2, fash1, ...

15. escaping meta character
- To search `/`, use `\/`
- To search `\`, use `\\`
- To search `.`, use `\.`

17.
- `\w` : as same as `/[a-zA-Z0-9_]/`
- `\W` : as same as `/[^a-zA-Z0-9_]/`
- `\s` : as same as `/[\t]/`, tab and white space
- `\S` : as same as `/[^\t]/`
- `\d` : as same as `/[0-9]/`

18. Quantifiers and repetitions
- Quantifier: wildcard that specifies how many times to match
    - `*`: zero or more of previous character
    - `+`: one or more of previous character
    - `?`: one or zero of previous character
- `/flavou*r/` finds flavor or flavour or flavouuuur
- `/flavou+r/` finds flavour or flavouuuur
- `/flavou?r/` finds flavor or flavour

19. Limiting repetitions
- Use curly braces to avoid infinite repetitions
    - `{exact_number}` or `{min,max}`
    - `{min,}` will search more than min repetitions
- `/flavou{2,4}r/` searches flavouur, flavouuur, flavouuuur only

20. Greedy expression
- Using quantifier `*+?` as many times as possible
- Matches the longest possible
- Backtracks the string to match as many as possible
- Ex) for `/".+"/`
    - `earth has "mountains" and many "seas" to explore`
    - `-----------^` first matching of `"`
    - `--------------------^ or ------^` is not matches as the 2nd `"` in the expression
    - `------------------------------------^` last matching of `"`

21. Lazy expression
- Matches the smallest possible
- Add `?` into the quantifier like `*?`, `+?`, `??`
- Ex) for `/".+?"/`
    - `earth has "mountains" and many "seas" to explore`
    - `-----------^` first matching of `"`
    - `--------------------^` last matching of `"`

22. Greedy vs lazy

23. Groups
- Use `()`
- cannot be used in set []

24. Alternation
- Use `|`
- `/I think (abc|def|xy) would do/` will get 
    - `I think abc would do`
    - `I think def would do`
    - `I think xy would do`

25. Nested alternation
- `((abc|def) would be | (xy|vw) might be)`
- Use nested `()`

26. Anchors
- `^`: outside of set[], this is not negate but anchors the start position of the string
    - `^a`: the first character is `a`
- `$`: ending of the string. Or before `\n` a the end of the string or a line
    - `a$`: the end character is `a`

28. Other anchors
- Depends on language:.NET, Python, PHP, Ruby, Perl, Java
    - `\A` : Beginning of the string. No multiline
    - `\Z` : end of the string or before `\n` at the end of the string
    - `\z` : end of string

29. Word boundaries
- word for `\w` as same as `/[a-zA-Z0-9_]/`
- `\b` : on a word boundary
    - `\b\w+\b` will find words in the strings
    - No symbol as `\w` doesn't allow symbols
- `\B` : not on word boundary
- they are zero length

30. Back-references
- Can use captured group as a reference
- `(ab)(cd)\1\2\1` will capture `abcdabcdab`

31. Ex of back reference
- `\b(\w+)\b \1` : may find repeating words (but not triple)

32. Application
- `\s` : whitespace character

33. Non-capturing groups
- Increases speed    
- Use `?:`
    - Will not be stored for reference
- `(ab)(?:cd)\1` : `\2` will not work

35. Positive Look ahead assertion
- matches a pattern or a string before the match position or current match position
- Can be used as an extra filter
- Use `?=`
- `long(?=island)` will find `long` from `longisland`
- `(?=longisland)long` will find `long` from `longisland`

36. Example of positive look ahead assertion
- `\b\w+\b\.` captures the end word of a string, including `.`
- `\b\w+\b(?=\.)` captures the end word of a string, excluding `.`

38. Negative look ahead assertion
- When captures something not followed by something else
- Use `?!`
- `long(?!island)` will find `long` from `longixland`

39. Look behind assertion
- Not supported in Javascript
- Use `?<=` for positive look behind assertion
- Use `?<!` for negative look behind assertion
- `(?<=long)island` finds `island` from `longisland`

40. unicodes
- Use `\u` + 4 digit hexadecimal : `\u011B`

41. Projects

42. Names

44. email address
- `^[\w.\-#$%^]{1,35}@(\w+\.)+\w{1,10}$`

48. IP addresses
- `^((\d){1,3}\.){3}(\d{1,3})`
    - But this allows a number larger than 255
- `^([10]?[0-9][0-9]?|2[0-4][0-9]|25[0-5])\.([10]?[0-9][0-9]?|2[0-4][0-9]|25[0-5])\.([10]?[0-9][0-9]?|2[0-4][0-9]|25[0-5])\.([10]?[0-9][0-9]?|2[0-4][0-9]|25[0-5])$`

50. dates
- `^[01]?\d(\-|\/)([012]?\d|3[01])(\-|\/)\d{1,4}`
- Can read:
```
04-10-2011
11-4-1991
02-28-1999
07-31-1922
5-5-2013
```

## Some sample
- For txt like `some constraints { ... \n ACTION_LABEL 'MyAction' { ...] \n`, to find out `MyAction` from Python regex:
```py
import re
...
txt = "some constraints { ... \n ACTION_LABEL 'MyAction' { ...] \n"
m = re.findall(r".*(ACTION_LABEL)\s*\'(.+?)\'",txt)
if len(m) > 0:
    print(m[0][1])
```
