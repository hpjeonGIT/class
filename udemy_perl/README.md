## Title: Learn Perl 5 By Doing It
- Instructor: John Purcell

## Section 1: Basic Perl: Getting Started

### 1. Installing Perl and Some Great Free Editors

### 2. Hello World
```pl
use strict;
use warnings;

sub main {
  print "Hello World\n";
}
main()
```
- To run, `perl test.pl`

### 3. Downloading Text and Images - Updated
- https://github.com/caveofprogramming/learn-perl
- learn-perl/Tutorial3
- download_html.pl:
```pl
use strict;
use warnings;
use LWP::UserAgent;
use IO::Socket::SSL;

my $ua = LWP::UserAgent->new(cookie_jar=>{});

$ua->ssl_opts(
    'SSL_verify_mode' => IO::Socket::SSL::SSL_VERIFY_NONE, 
    'verify_hostname' => 0
);

my $request = new HTTP::Request('GET', 'https://caveofpython.com/');

my $response = $ua->request($request);

unless($response->is_success()) {
    die $response->status_line();
}

my $content = $response->decoded_content();

print($content);

print("Completed")
```
- download_image.pl:
```pl
use strict;
use warnings;
use LWP::UserAgent;
use IO::Socket::SSL;

my $ua = LWP::UserAgent->new(cookie_jar=>{});

$ua->ssl_opts(
    'SSL_verify_mode' => IO::Socket::SSL::SSL_VERIFY_NONE, 
    'verify_hostname' => 0
);

my $link = 'https://i0.wp.com/caveofpython.com/wp-content/uploads/2023/03/robot.png?w=996&ssl=1';

my $response = $ua->mirror($link, './robot.png');

unless($response->is_success()) {
    die $response->status_line();
}

print("\nCompleted")
```


### 4. Downloading Text and Images with Perl (Old version)

### 5. Arrays and Checking Whether Files exist
- Single quote '': as literal
- Double quote"": when special characters are used
- `-f` check if file exists or not
- Variable with `$`
- Array with `@`
  - Comma after the last item
  - For loop using array items
    - `foreach my $file (@files) {...}`
- To turn off output beffering, Use `$|=1;`
```pl
use strict;
use warnings;
# comment
$|=1;
sub main {
  my $file = './serval.png';
  my @files = ($file, './serval2.png',);
  foreach my $f0(@files) {  # @f0 will iterate over items in an array @files
    if( -f $f0) {  # -f returns true if $f0 exists
      print "Found file $f0\n";
    }
    else {
      print "File $f0 not foud\n";
    }
  }
}
main();
```

### 6. Reading Files and Beginning Regular Expressions
- `die;` let the program die
  - `die "some text";` prints the message
- `<INPUT>`: reads one line from INPUT
- `=~ /REGEX/`: check if REGEX expression matches or not
```pl
use strict;
use warnings;
# comment
$|=1;
sub main {
  my $file = './myfile.txt';
  open(INPUT, $file) or die("Input file $file not found\n");
  while(my $line = <INPUT>) {
    if($line =~ /ello/) {  # if any 'ello' is found in the line
      print $line;
    }
  }
  close(INPUT);
}
main();
```

### 7. Writing Files and Replacing Text
- `my $output = 'output.txt';` yields an error. Use `my $output = '>output.txt';` or use concatenation in open()
  - `open(OUTPUT,'>'.$output)`: overwrites when writes
  - `open(OUTPUT,'>>'.$output)`: appends when writes
- `$line =~ s/ello/ELLO/ig;`: in $line, replace 'ello' with 'ELLO'
```pl
use strict;
use warnings;
# comment
$|=1;
sub main {
  my $output = 'output.txt';
  open(OUTPUT,'>>'.$output) or die "Can't create $output";
  my $file = './myfile.txt';
  open(INPUT, $file) or die("Input file $file not found\n");
  while(my $line = <INPUT>) {
   $line =~ s/ello/ELLO/ig;
   print OUTPUT $line;
  }
  close(OUTPUT);
}
main();
```

### 8. Wildcards in Regular Expressions
- `/ h.s /` will look for has, his, hss, hxs, ... surrounded with a space

### 9. Groups: Finding Out What You Actually Matched
- Grouping using `()`
- Each group can be called as `$1`, `$2`, ...
```pl
use strict;
use warnings;
# comment
$|=1;
sub main {
  my $file = './myfile.txt';
  open(INPUT, $file) or die("Input file $file not found\n");
  while(my $line = <INPUT>) {
    if($line =~ /(.llo.)(...)/) { # (hello )(wor)
      print "first group $1, second group $2\n";
    }
  }
  close(INPUT);
}
main();
```

### 10. Quantifiers: Greedy vs. Non-Greedy
- '+': greedy quantifier. `l+` means l or ll or lll ...
- '+?': non-greedy quantifier. As few as possible
- '*': greedy quantifier
  - `/(.l.*o)/` will capture **'ello wo'**, not 'ello' from 'hello world'
  - Will try to capture as many as possible
- '*?': non-greedy quantifer
  - `/(.l.*?o)/` will capture 'ello' from 'hello world'
  - Will try to capture as few as possible

### 11. Escape Sequences
- `\d`: digit
- `\s`: space
- `\S`: non-space character
- `\w`: alphanumeric. [a-zA-Z0-9]
```pl
use strict;
use warnings;
# comment
$|=1;
sub main {
 my $text = 'I am 117 years old tomorrow.';
 if ($text =~ /(\d+)/) {
   print "$1\n" # prints 117
 }
}
main();
```

### 12. Numeric Quantifiers
- `\d{5}`: \d for 5 times
- `\d{3,6}`: \d for min 3 and max 6 times
- `\d{3,}`: \d for min 3 times or more
```pl
use strict;
use warnings;
# comment
$|=1;
sub main {
 my $text = 'DE$12345';
 if ($text =~ /(DE\$\d{3,6})/) {
   print "$1\n" # prints DE$12345
 }
}
main();
```

### 13. Test your Perl and Regex Knowledge - First Test
```pl
use strict;
use warnings;
$|=1;
sub main {
  my @emails = (
   'john@caveofprogramming.com',
   'hello',
   '@alpha.com',
   'ijd@somewhre.com',
   'ijk@7788.',
  );
  for my $email (@emails) {
    if ($email =~ /(\w+\@\w+\.\w+)/) {
      print "Found valid email $email\n";
    } else {
      print "Found invalid email $email\n";
    }
  }
}
main();
```

## Section 2: More on Reading Files Line by Line: Tips, Tricks and Vital Knowledge

### 14. Split and Reading CSV Files
- `while(<INPUT>) {...}`: Reads file handler into `$_`
  - Better practice: Use `while (my $line = <INPUT>) {...}` then every line is captured at `$line`
```pl
use strict;
use warnings;
sub main {
  my $input = 'test.csv';
  unless(open(INPUT,$input)) {
    die "\n Cannot open $input \n";
  }
  <INPUT>; # reads the first line (header)
  while(my $line = <INPUT>) {
    my @values = split ',', $line; # split by comma
    print $values[2]. "\n"; # print [2] element of values array
  }
  close(INPUT)
}
main();
```

### 15. Join and Viewing Data Using Data::Dumper
- `print join '|', @arrays; ` will print the join of array values
```pl
use strict;
use warnings;
use Data::Dumper;
$|=1;
sub main {
  my $input = 'test.csv';
  unless(open(INPUT,$input)) {
    die "\n Cannot open $input \n";
  }
  <INPUT>; # reads the first line (header)
  while(my $line = <INPUT>) {
    my @values = split ',', $line;
    #print join '|', @values;
    print Dumper(@values);
  }
  close(INPUT)
}
main();
```
- Running:
```bash
$ perl ch15.pl 
$VAR1 = 'Isaac Newton';
$VAR2 = '99.10';
$VAR3 = '15051999
';
$VAR1 = 'Albert Einstein';
$VAR2 = '13.20';
$VAR3 = '11062012
';
$VAR1 = 'Carl Scheele';
$VAR2 = '66.23';
$VAR3 = '01012000
';
$VAR1 = 'Rene Descartes';
$VAR2 = '0.57';
$VAR3 = '10072033
';
```
- $VAR3 yields '\n'. See next section to remove it

### 16. Chomp and Removing Spaces in Splits
- Q: chomp is not working as shown by the instructor
- `my @values = split /\s*,\s*/, $line;` : can use REGEX to split 

### 17. "Pushing" Onto Arrays
- `push @arrays, @var;` will push value var to arrays in the end
```pl
use strict;
use warnings;
use Data::Dumper;
$|=1;
sub main {
  my @arr;
  @arr[0] = 'abc'; # will append or allocate element
  @arr[1] = 'def'; 
  push @arr, 'hello'; # push to the end
  for my $el(@arr) {
    print $el."\n";
  }
}
main();
```

### 18. Arrays of Arrays
- When use `push`, use `\` for appending array
```pl
use strict;
use warnings;
use Data::Dumper;

my @animals = ('dog','cat','rabbit');
my @fruits = ('apple','banana','oragen');
my @val1;
my @val2;
push @val1, @animals;
push @val1, @fruits;
push @val2, \@animals; # entire @animals as a single. Array of array
push @val2, \@fruits;
print Dumper(@val1);
print Dumper(@val2);
```
- Running 
```bash
$ perl ch18_2.pl 
$VAR1 = 'dog';
$VAR2 = 'cat';
$VAR3 = 'rabbit';
$VAR4 = 'apple';
$VAR5 = 'banana';
$VAR6 = 'oragen';
$VAR1 = [
          'dog',
          'cat',
          'rabbit'
        ];
$VAR2 = [
          'apple',
          'banana',
          'oragen'
        ];
```
- Access through index
```pl
use strict;
use warnings;
use Data::Dumper;
$|=1;
sub main {
  my $input = 'test.csv';
  unless(open(INPUT,$input)) {
    die "\n Cannot open $input \n";
  }
  <INPUT>; # reads the first line (header)
  my @lines;
  while(my $line = <INPUT>) {
    my @val = split ',', $line;
    push @lines, \@val;
  }
  close(INPUT);
  print $lines[1][2]; # prints 11062012
  foreach my $line(@lines) {
    #print $line, "\n";
    print $line->[0];  # prints names
  }
}
main();
```

### 19. Hashes: Lookup Tables in Perl
- Symbols in Perl
  - Ref: https://www.cs.unc.edu/~jbs/resources/perl/perl-basics.html
  - `$name`: scalar variable
  - `@name()`: array
  - `%name{}`: hashes
- Not-ordered by keys
```pl
use strict;
use warnings;
use Data::Dumper;
$|=1;
sub main {
  my %months = (
    "Jan" => 1,
    "Feb" => 2,
    "Mar" => 3,
    "Apr" => 4,
  );
  print $months{"Jan"}."\n"; 
  my @months = keys %months; # array months. Gets keys of %months{}
  foreach my $month(@months) {
    print "$month: $months{$month}\n"
  }
  while (my ($key,$val) = each %months) { # key/val pair using each
    print "$key: $val\n"
  }
}
main();
```

### 20. Iterating Over hashes
- While loop is shown above
```pl
use strict;
use warnings;
use Data::Dumper;
$|=1;
sub main {
  my %foods = (
    "mice" => "cheese",
    "cats" => "birds",
    "dogs" => "bones",
  );
  foreach my $key(sort keys %foods) { # sort along keys
    my $val = $foods{$key};
    print "$key = $val\n";
  }
}
main();
```                           

### 21. Arrays of Hashes
- Use push
  - Will flatten as default
  - Use `\` to have the shape of keys/values (dereferencing)
```pl
use strict;
use warnings;
use Data::Dumper;
$|=1;
sub main {
  my %foods = (
    "mice" => "cheese",
    "cats" => "birds",
    "dogs" => "bones",
  );
  my @arr;
  push @arr, %foods;
  print Dumper(@arr);
  my @arr2;
  push @arr2, \%foods;
  print Dumper(@arr2);
}
main();
```
- Running:
```bash
$ perl ch21.pl 
$VAR1 = 'cats';
$VAR2 = 'birds';
$VAR3 = 'dogs';
$VAR4 = 'bones';
$VAR5 = 'mice';
$VAR6 = 'cheese';
$VAR1 = {
          'cats' => 'birds',
          'dogs' => 'bones',
          'mice' => 'cheese'
        };
```

### 22. Storing CSV Data in a Data Structure
```pl
use strict;
use warnings;
use Data::Dumper;
$|=1;
sub main {
  my $input = 'test.csv';
  unless(open(INPUT,$input)) {
    die "\n Cannot open $input \n";
  }
  my @data;
  <INPUT>; # reads the first line (header)
  while(my $line = <INPUT>) {
    my ($name, $payment, $date) = split ',', $line;
    my %values = (
       "Name" => $name,
       "Payment" => $payment,
       "Date" => $date,
    );
    push @data, \%values;
  }
  close(INPUT);
  foreach my $line(@data) {
    #print Dumper($line);
    print $line->{"Payment"}."\n";
  }
  print "Descartes: ". $data[3]{"Name"}."\n";
}
main();
```

### 23. Validating CSV Data
- How to skip empty line:
```pl
while(my $line = <INPUT>) {
  $line =~ /\S+/ or next; # skip if regex doesn't find any non-space character, jump to next loop
}
```
- Compare the size of array:
```pl
my @val = split ',', $line;
if (scalar(@val) < 3) { ... }
```    

### 24. Cleaning CSV Data
- How to remove redundant spaces
- Ref: https://www.geeksforgeeks.org/perl-removing-leading-and-trailing-white-spaces-trim/
- Left Trim (`~ s/^\s+//`): Removes extra spaces from leftmost side of the string till the actual text starts. From the leftmost side the string takes 1 or more white spaces (\s+) and replaces it with nothing.
- Right Trim (`~ s/\s+$//`): Removes extra spaces from rightmost side of the string till the actual text end is reached. From the rightmost side the string takes 1 or more white spaces (\s+) and replaces it with nothing.
- Trim (`~ s/^\s+|\s+$//`): It removes extra space from both sides of the string.
- `$line =~ s/\?|\!//g;`: replace all ? or ! symbol with ''

### 25. Test your Perl and Regex Knowledge - Second Test
- Reading following csv file:
```bash
Name,Payment,Date
Isaac Newton,$99.10,15051999
Albert Einstein,approx 13.20,11062012

Carl Scheele?,66.23,01012000
P.G Wodehouse,70.32
Dr. Who,, 16121978
    Rene Descartes,0.57,10072033
```
- Read complete 3columns data only, removing symbols
```pl
use strict;
use warnings;
use Data::Dumper;
$|=1;
sub main {
  my $input = 'ex.csv';
  unless(open(INPUT,$input)) {
    die "\n Cannot open $input \n";
  }
  my @data;
  my $line = <INPUT>; # skip header line
  chomp $line;
  my ($header1,$header2,$header3) = split /\s*,\s*/, $line;
  
  LINE: while(my $line = <INPUT>) { # LINE: as label
    chomp $line;
    $line =~ /\S+/ or next;
    $line =~ s/^\s*|\s*$//g;
    my @val = split /\s*,\s*/, $line;
    if (@val< 3) {
      next;
    }
    foreach my $value(@val) {
      if(length($value) ==0) {
        next LINE;               # next to the label LINE:
      }
    }
    my ($name,$payment,$date) = @val;
    my %data = (
      $header1 => $name,
      $header2 => $date,
      $header3 => $payment,      
    );
    push @data, \%data
  }
  close(INPUT);
  print Dumper(@data);
}
main();
```

## Section 3: Web Scraping and More Regular Expressions

### 26. Basic Web Scraping
- For web-scraping, it is better to use `Web::Scraper;`
- `defined()` will check if the argument has some content or not
- In REGEX, 
  - `m'` to find a single quote
  - `i` for case-insenstive search
```pl
use strict;
use warnings;
use LWP::Simple;
$|=1;
sub main{
  my $content = get("https://www.caveofprogramming.com/");
  unless(defined($content)) {
    die "Unreachable url\n";
  }
  if($content =~ m'<a class="mainlink" href=".+?">(.+?)</a>'i) {
    my $title = $1;
    print "Title: $title\n";
  }
  else {
    print "\nTitle not found\n";
  }
}
main();
```

### 27. Character Classes
- Matching a single or more characters using []
```pl
use strict;
use warnings;
$|=1;
sub main{
  my $content = "The 39 steps";
  if ($content =~ /([0-9]+)/) {
    print "matched '$1'\n"; # prints 39
  }
  else {
    print "No match\n";
  }
}
main();
```
- [0-9] any number
- [A-Z] any uppercase letter (in the English alphabet)
- [\=\%] - simply list alternatives. Backslash any character that might have a special meaning in regex
- [A-Za-z_0-9] -- specify alternatives just by listing them; can include ranges.
- [^0-9T\s] ^ Match anything EXCEPT the specified characters.

### 28. Matching Repeatedly
- Using `g` in REGEX
```pl
use strict;
use warnings;
use LWP::Simple;
$| = 1;
sub main {
	my $content = get("http://www.caveofprogramming.com");
	unless ( defined($content) ) {
		die "Unreachable url\n";
	}
	# <a href="http://news.bbc.co.uk">BBC News</a>
	# []<>
	while (
		$content =~
m| # Use a pipe character as the quote, since we don't need to use it inside the regex.
		<\s*a\s+ # Match the opening <a, with or without space around the 'a'
		[^>]* # Match any amount of gumpf, as long as it's not the closing angle bracket quote		
		href\s*=\s* # Match href=, with or without space around the '='		
		['"] # Match either a single or double quote		
		([^>"']+) # Match any text, as long as it doesn't include a closing '>' bracket or a quote		
		['"] # Close the quote		
		[^>]*> # Match anything other than the closing bracket, followed by the closing bracket.	
	\s*([^<>]*)\s*</ # Match the hyperlinked text; any characters other than angle brackets
	|sigx # s: match across new lines; i: case insensitive match; g: global (repeated) match; x: allow whitespace and comments 
	  )
	{
		print "$2: $1\n";
	}
}
main();
```

### 29. Collecting Repeated Matches All At Once
- Finding class from the webpage
```pl
use strict;
use warnings;
use LWP::Simple;
$| = 1;
sub main {
	my $content = get("http://www.caveofprogramming.com");
	unless ( defined($content) ) {
		die "Unreachable url\n";
	}
	my @classes = $content =~ m|class="([^"']*?)"|ig;

	if(@classes == 0) {
		print "No matches\n";
	}
	else {
		foreach my $class(@classes) {
			print "$class\n";
		}
	}
}
main();
```

## Section 4: Building a Complete Program: Command Line Options

### 30. Getting Command Line Options
- How to get argument using options
```pl
use strict;
use warnings;
use Data::Dumper;
use Getopt::Std;
$| = 1;
sub main {
	my %opts;	
	getopts('af:c', \%opts);	
	print Dumper(%opts);	
	my $file = $opts{'f'};	
	print "File: $file\n"
}
main();
```
- Using `print $ARGV[0]."\n";` might be better (?)

### 31. Subroutines and Returning Values
- No return type in subroutines of perl
  - true/false not allowed. Return 0 for false and 1 for true
- `return` may not be necessary. The last line will be returned
```pl
use strict;
use warnings;
use Data::Dumper;
use Getopt::Std;
$|=1;
sub main {
  if (checkusage()) {
    usage();
  }
}
sub checkusage {
  return 1;
}
sub usage {
  print "Correct options\n";
}
main();
```

### 32. Multi-Line Strings and Commands
```pl
use strict;
use warnings;
$|=1;
sub main {
  my $var = <<USAGE;

This is the results of multi line
strings

USAGE
  die $var;
}
main();
```
- Demo:
```bash
$ perl ch32.pl 

This is the results of multi line
strings

```
- Or can use `print <<USAGE; .... USAGE` block. USAGE is arbitrary keyword and you may name it as you want
```pl
sub main {
  print <<JEONB;

This is the results of multi line
strings

JEONB
}
```

### 33. Passing Arguments to Subroutines
- Regardless of type and number of arguments, they can be transferred using `@_`
```pl
use strict;
use warnings;
$|=1;
sub main {
  usage("Hello", 123);
}
sub usage() {
  my ($greet, $count) = @_;
  print $greet."\n". $count."\n";
}
main();
```
- The local variables above can be written as:
```pl
  my $greet = shift;
  my $count = shift;
```

### 34. References to Hashes
- How to pass Hashes as function arguments
- Sending a hash with dereferencing
```pl
use strict;
use warnings;
use Data::Dumper;
$|=1;
sub main {
    my %months = (
    "Jan" => 1,
    "Feb" => 2,
    "Mar" => 3,
    "Apr" => 4,
  );
  usage(\%months);
}
sub usage() {
  my $mymonths = shift;
  print $mymonths->{"Jan"}."\n";
}
main();
```

### 35. Checking Values in Hashes
- Multiline comments in perl
```pl
=pod
 comment here
 comment here
=cut
```
- Final sample code from the instructor:
```pl
use strict;
use warnings;
use Data::Dumper;
use Getopt::Std;
$| = 1;
=pod
	This is ACME XML parser version 1.0
	Use with care.	
=cut
sub main {
	my %opts;	
	# Get command line options
	getopts('af:r', \%opts);	
	if(!checkusage(\%opts)) {
		usage();
	} 	
=pod
	perl parse.pl -a -f test.xml -r	
	a => 1
	r => 1
	f => test.xml
=cut
}
sub checkusage {
	my $opts = shift;	
	my $a = $opts->{"a"};
	my $r = $opts->{"r"};
	my $f = $opts->{"f"};	
	# Image a is optional; don't really need to refer to it here at all.	
	# r is mandatory: it means "run the program"
	# f is mandatory.	
	unless(defined($r) and defined($f)) {
		return 0;
	}	
	unless($f =~ /\.xml$/i) {
		print "Input file must have the extension .xml\n";
		return 0;
	}	
	return 1;
}
sub usage {
	print <<USAGE;	
usage: perl main.pl <options>
	-f <file name>	specify XML file name to parse
	-a	turn off error checking
	-r run the program
example usage:
	perl main.pl -r -f test.xml -a	
USAGE	
	exit();
}
main();
```

## Section 5: Parsing XML and Complex Data Structures

### 36. Finding All Files in a Directory and Filtering Arrays
- For a directory, use opendir(), readdir(), closedir()
```pl
use strict;
use warnings;
use Data::Dumper;
$| = 1;
sub readD {
  my $read_dir = shift;
  unless(opendir(INPUTDIR, $read_dir)) {
    die "Cannot open $read_dir\n:";
  }
  my @files = readdir(INPUTDIR);
  closedir(INPUTDIR);
  return @files;
}
sub main() {
  my $target = ".";
  my @files = readD($target);
  print Dumper(\@files);
}
main();
```

### 37. Processing Files One By One
- Note that we use `@$files` in foreach loop
```pl
use strict;
use warnings;
use Data::Dumper;
$| = 1;
sub readD {
  my $read_dir = shift;
  unless(opendir(INPUTDIR, $read_dir)) {
    die "Cannot open $read_dir\n:";
  }
  my @files = readdir(INPUTDIR);
  closedir(INPUTDIR);
  return @files;
}
sub process_files {
  my $files = shift;
  #print Dumper($files);
  foreach my $file (@$files) {
    print "$file","\n";
  }
}
sub main() {
  my $target = ".";
  my @files = readD($target);
  #print Dumper(\@files);
  process_files(\@files);
}
```

### 38. Parsing XML with Regular Expressions
- Instead of line separating using new line ("\n"), separate parsed texts using `</entry>`
  - `$/ = "</entry>";` will configure the parsing separator globally

### 39. Using XML::Simple, and Extracting Data from Complex Structures
- `undef $/;` disable separator when read file. The entire text will be read
  -  Default would be `$/ = \n;`
```pl
	open(INPUTFILE, $filepath) or die "Unable to open $filepath\n";
	undef $/;
	my $content = <INPUTFILE>;	 # parse the entire text
	close(INPUTFILE);	
	print $content;	
	my $parser = new XML::Simple;	
	my $dom = $parser->XMLin($content);	
	print Dumper($dom);
```

### 40. Extracting Data from Complex Structures: A Complete Example
- Use `ForceArray=>1` as an extra argument
```pl
sub process_file {
	my ($file, $input_dir) = @_;	
	print "Processing $file in $input_dir ... \n";	
	my $filepath = "$input_dir/$file";	
	open(INPUTFILE, $filepath) or die "Unable to open $filepath\n";	
	undef $/;	
	my $content = <INPUTFILE>;	
	close(INPUTFILE);	
	print $content;
	my $parser = new XML::Simple;	
	my $dom = $parser->XMLin($content, ForceArray => 1);	
	print Dumper($dom);	
	foreach my $band(@{$dom->{"entry"}}) {
		my $band_name = $band->{"band"}->[0];		
		print "\n\n$band_name\n";
		print "============\n";		
		my $albums = $band->{"album"};		
		foreach my $album(@$albums) {
			my $album_name = $album->{"name"}->[0];
			my $chartposition =  $album->{"chartposition"}->[0];			
			print "$album_name: $chartposition\n";
		}
	}
}
```

### 41. Building Complex Data Structures

## Section 6: Working with Databases

### 42. Free Databaes to Use with Perl

### 43. Creating Databases with MySQL

### 44. Connecting to a Database
```pl
use strict;
use warnings;
use DBI;
$| = 1;
sub main {	
	my $dbh = DBI->connect("dbi:mysql:bands", "john", "letmein");	
	unless(defined($dbh)) {
		die "Cannot connect to database.\n";
	}	
	print "Connected\n";	
	$dbh->disconnect();
}
```

### 45. Inserting Data into a Database
```pl
sub add_to_database {
	my $data = shift;	
	my $dbh = DBI->connect("dbi:mysql:bands", "john", "letmein");	
	unless(defined($dbh)) {
		die "Cannot connect to database.\n";
	}	
	print "Connected to database.\n";	
	my $sth = $dbh->prepare('insert into bands (name) values (?)');	# using ? is a way to avoid SQL injection attack
	unless($sth) {
		die "Error preparing SQL\n";
	}	
	foreach my $data(@$data) {
		my $band_name = $data->{"name"};		
		print "Inserting $band_name into database ...\n";		
		unless($sth->execute($band_name)) {
			die "Error executing SQL\n";
		}
	}	
	$sth->finish();	
	$dbh->disconnect();	
	print "Completed.\n";
}
```

### 46. Deleting Data and Executing Dataless SQL  Commands
```pl
sub add_to_database {
	my $data = shift;	
	my $dbh = DBI->connect("dbi:mysql:bands", "john", "letmein");	
	unless(defined($dbh)) {
		die "Cannot connect to database.\n";
	}	
	print "Connected to database.\n";	
	my $sth = $dbh->prepare('insert into bands (name) values (?)');	
	unless($sth) {
		die "Error preparing SQL\n";
	}	
	$dbh->do('delete from bands') or die "Can't clean bands table.\n";
	$dbh->do('delete from albums') or die "Can't clean bands table.\n";	
	foreach my $data(@$data) {
		my $band_name = $data->{"name"};		
		print "Inserting $band_name into database ...\n";		
		unless($sth->execute($band_name)) {
			die "Error executing SQL\n";
		}
	}	
	$sth->finish();	
	$dbh->disconnect();	
	print "Completed.\n";
}
```

### 47. Getting the IDs of Records You've just Inserted
```pl
sub add_to_database {
	my $data = shift;	
	my $dbh = DBI->connect("dbi:mysql:bands", "john", "letmein");	
	unless(defined($dbh)) {
		die "Cannot connect to database.\n";
	}	
	print "Connected to database.\n";	
	my $sth_bands = $dbh->prepare('insert into bands (name) values (?)');
	my $sth_albums = $dbh->prepare('insert into albums (name, position, band_id) values (?, ?, ?)');	
	unless($sth_bands) {
		die "Error preparing band insert SQL\n";
	}	
	unless($sth_albums) {
		die "Error preparing album insert SQL\n";
	}	
	$dbh->do('delete from albums') or die "Can't clean bands table.\n";
	$dbh->do('delete from bands') or die "Can't clean bands table.\n";	
	foreach my $data(@$data) {
		my $band_name = $data->{"name"};
		my $albums = $data->{"albums"};		
		print "Inserting $band_name into database ...\n";		
		unless($sth_bands->execute($band_name)) {
			die "Error executing SQL\n";
		}		
		my $band_id = $sth_bands->{'mysql_insertid'};		
		foreach my $album(@$albums) {
			my $album_name = $album->{"name"};
			my $album_position = $album->{"position"};			
			# print "$album_name, $album_position\n";			
			unless($sth_albums->execute($album_name, $album_position, $band_id)) {
				die "Unlable to execute albums insert.\n";
			}
		}	
	}	
	$sth_bands->finish();
	$sth_albums->finish();	
	$dbh->disconnect();	
	print "Completed.\n";
}
```

### 48. Querying Databases
```pl
sub export_from_database {
	my $dbh = shift;
	print "Exporting ...\n";	
	my $sql = 'select b.id as band_id, b.name as band_name, a.id as album_id, ' .
		'a.name as album_name, a.position as position  ' .
		'from bands b join albums a on a.band_id=b.id';	
	my $sth = $dbh->prepare($sql);	
	unless(defined($sth)) {
		die "Unable to prepare export query.\n";
	}	
	unless($sth->execute()) {
		die "Unable to execute export query.\n";
	}	
	while(my $row = $sth->fetchrow_hashref()) {
		my $band_id = $row->{"band_id"};
		my $band_name = $row->{"band_name"};
		my $album_id = $row->{"album_id"};
		my $album_name = $row->{"album_name"};
		my $position = $row->{"position"};		
		print "$band_id, $band_name, $album_id, $album_name, $position\n";		
	}	
	$sth->finish();
}
```

### 49. Exporting Data
- Export to CSV. Just print to OUTPUT, which would be opened in the earlier stage

## Section 7: Perl One-Liners

### 50. Running One-Line Perl Programs
- `perl -e 'print "Hello world\n"'`

### 51. Replacing Text in Files
- `perl -pe 's/OLD/NEW/gi' sample.txt`

### 52. Editing Files in Place
- `perl -i.orig -pe 's/OLD/NEW/gi' sample.txt`

## Section 8: Modules and OO Perl

### 53. Modules
- File extension: *.pm
- Speak.pm 
```pl
package Speak;
sub test { print "Hello world\n"; }
1; # returns 1
```
- ch53.pl 
```pl
use strict;
use warnings;
use Speak;
$!=1;
sub main() {
  Speak::test();
}
main();
```
- Perl may not find the module in the current working directory. Run as: `perl -I . ch53.pl`
- `qw(txt1 txt2 tx3)`: array without quotation marks
  - Equivalent to `('txt1','txt2','txt3')`
- In the module file, using Exporter, @EXPORT_OK will allow to use subroutine names without namespace
```pl
use Exporter qw(subname1);
use Exporter qw(subname2);
@EXPORT_OK = qw(subname1 subname2);
```

### 54. Package and Directories
- Comm/Speak.pm 
```pl
package Comm::Speak;
sub test { print "Hello world\n"; }
1;
```
- ch54.pl
```pl
use strict;
use warnings;
use Comm::Speak;
$!=1;
sub main() {
  Comm::Speak::test();
}
main();
```
- Run as `perl -I . ch54.pl`
- Or inject `use lib '/..../';` into the ch54.pl, pointing the location of Comm/Speak.pm

### 55. Object Orientation: A Brief Introduction

### 56. Implementing OO in Perl
- Data/Person.pm
```pl
package Data::Person;
sub new {
	my $class = shift;	
	my $self = {
		"name" => shift,
		"age" => shift,
	};	
	bless($self, $class);	# Ref: https://stackoverflow.com/questions/392135/what-exactly-does-perls-bless-do
  # bless() associates an object with a class
	return $self;
}
sub greet {
	my ($self, $other) = @_;	
	print "Hello $other; my name is " . $self->{"name"} . "; I am " . $self->{"age"} . " years old.\n";
}
1;
```
- main.pl
```pl
use strict;
use warnings;
use Data::Person;
$|=1;
sub main {	
	my $person1 = new Data::Person("Bob", 45);
	$person1->greet("Sue");	
	my $person2 = new Data::Person("Mike", 55);
	$person2->greet("Rogriguez");
}
main();
```

## Section 9: Web Application Basics

### 57. Installing the Apache HTTP server

### 58. A Hello World Web App
- test.cgi:
```pl
#!/opt/local/bin/perl
use strict;
use warnings;
sub main {
	print "Content-type: text/html\n\n"; # this line is consumed by a browser
	print "Hello world";
}
main();
```

### 59. The CGI.pm Module

### 60. Using URL Parameters
```pl
#!/opt/local/bin/perl
use strict;
use warnings;
use CGI;
my $CGI = new CGI();
sub main {
	print $CGI->header();
	my $user = $CGI->param("user");
	my $password = $CGI->param("pass");
print<<HTML;
	<html>
	<b>Hello world</b>
	User: $user, Pass: $password
	</html>
HTML
}
main();
```

### 61. Website Forms
```pl
#!/opt/local/bin/perl
use strict;
use warnings;
use CGI;
my $CGI = new CGI();
sub main {
	print $CGI->header();
	my @query = $CGI->param();
	@query = map($_ . ": " . $CGI->param($_), @query);
	my $query = join(', ', @query);
print<<HTML;
	<html>
	<form action="test4.cgi" method="post">
	<input type="text" name="query" />
	<input type="hidden" name="go" value="true" />
	<input type="submit" name="submit" value="Go" /> 
	</form>
	<p>Last submitted: $query</p>
  </html>
HTML
}
main();
```

## Section 10: Basic Sysadmin Tasks

### 62. Moving, Copying, and Deletig Files
```pl
use strict;
use warnings;
use File::Copy;
$|=1;
sub main {
	if(move( 
	'./serval', 'serval2.png')){
		print "One file moved.\n";
	}
	else {
		print "Unable to move file\n";
	}
	unlink('serval2.png'); # deletes the file
}
main();
```

### 63. Executing System Commands
```pl
use strict;
use warnings;
$|=1;
sub main {	
	my $command = 'cd ..; ls -l';
	my @output = `$command`;	
	print join('', @output);
}
main();
```

## Section 11: Conclusion

### 64. Where to Find Documentation and More Modules
- https://www.cpan.org/
- https://perldoc.perl.org/

## Section 12: Appendix 1: Example Data

### 65. XML Files

## Section 13: Appendix 2: Alternate Systems

### 66. Running Perl in Unix, Linux, Mac and Cygwin

## Section 14: Extras

### 67. Arrays and Hashes Review

### 68. References to Hashes and Arrays Review
```pl
use strict;
use warnings;
$|=1;
sub main {	
	my @fruits = ("apple", "banana", "orange");	
	my %months = (
		"Jan" => 1,
		"Feb" => 2,
	);	
	print $fruits[0] . "\n";
	$fruits[3] = "kiwi";	
	print $months{"Jan"}. "\n";	
	$months{"Mar"} = 3;	
	my $fruits_ref = \@fruits;
	print $fruits_ref->[0] . "\n";	
	my $months_ref = \%months;
	print $months_ref->{"Jan"}. "\n";		
	foreach my $fruit(@$fruits_ref) {
		print "$fruit\n";
	}	
	while( my ($key, $value) = each %$months_ref) {
		print "$key - $value\n";
	}	
}
main();
```
